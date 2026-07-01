import torch  
import os
import numpy as np
import hydra
from pathlib import Path
from collections import deque

import yaml
from datetime import datetime
import importlib
import dill
from argparse import ArgumentParser
import cv2

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

target_size = 308
patch_size = 14

def get_target_shape(width, height, mode="crop"):
    """
    Calculate target shape based on input dimensions and mode.
    
    Args:
        width (int): Original image width
        height (int): Original image height
        mode (str): Either "crop" or "pad"
        
    Returns:
        tuple: (new_height, new_width)
    """
    if mode == "pad":
        # Make the largest dimension 518px while maintaining aspect ratio
        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / patch_size) * patch_size
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / patch_size) * patch_size
    else:  # mode == "crop"
        # Set width to 518px
        new_width = target_size
        # Calculate height maintaining aspect ratio, divisible by patch_size
        new_height = round(height * (new_width / width) / patch_size) * patch_size
        
        # Center crop height if it's larger than target_size
        if new_height > target_size:
            new_height = target_size
            
    return new_height, new_width

def preprocess_rgb(rgb, mode="crop"):
    """
    Preprocess RGB image with the following steps:
    1. Normalize to [0, 1]
    2. Resize to target shape (maintaining aspect ratio and divisible by patch_size)
    3. Center crop or pad if necessary
    4. Transpose to (C, H, W)
    
    Args:
        rgb (numpy.ndarray): Input RGB image with shape (H, W, C)
        mode (str): Either "crop" or "pad"
        
    Returns:
        numpy.ndarray: Preprocessed image with shape (C, H, W)
    """
    # Convert to float32 and normalize to [0, 1]
    rgb = rgb.astype(np.float32) / 255.0
    
    # Get original dimensions
    height, width = rgb.shape[:2]
    
    # Calculate target dimensions
    new_height, new_width = get_target_shape(width, height, mode)
    
    # Resize image
    rgb = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Handle cropping or padding
    if mode == "crop":
        # Center crop if height is larger than target_size
        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            rgb = rgb[start_y:start_y + target_size, :, :]
    else:  # mode == "pad"
        # Pad to make a square of target_size x target_size
        h_padding = target_size - rgb.shape[0]
        w_padding = target_size - rgb.shape[1]
        
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            
            # Pad with white (value=1.0)
            rgb = np.pad(rgb, 
                        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                        mode='constant',
                        constant_values=1.0)
    
    # Transpose to (C, H, W)
    rgb = np.transpose(rgb, (2, 0, 1))
    
    return rgb

class DPRunner:
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=300,
                 n_obs_steps=1,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 tqdm_interval_sec=5.0,
                 task_name=None,
    ):
        self.task_name = task_name
        self.eval_episodes = eval_episodes
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.obs = deque(maxlen=n_obs_steps+1)
        self.env = None

        self.obs_encoder_time_list = []
        self.action_pred_time_list = []

    def stack_last_n_obs(self, all_obs, n_steps):
        assert(len(all_obs) > 0)
        all_obs = list(all_obs)
        if isinstance(all_obs[0], np.ndarray):
            result = np.zeros((n_steps,) + all_obs[-1].shape, 
                dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = np.array(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        elif isinstance(all_obs[0], torch.Tensor):
            result = torch.zeros((n_steps,) + all_obs[-1].shape, 
                dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = torch.stack(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        else:
            raise RuntimeError(f'Unsupported obs type {type(all_obs[0])}')
        return result
    
    def reset_obs(self):
        length = min(len(self.obs_encoder_time_list), 100)
        self.obs_encoder_time_avg = np.mean(self.obs_encoder_time_list[-length:])
        self.action_pred_time_avg = np.mean(self.action_pred_time_list[-length:])
        self.obs.clear()

    def update_obs(self, current_obs):
        self.obs.append(current_obs)

    def get_n_steps_obs(self):
        assert(len(self.obs) > 0), 'no observation is recorded, please update obs first'

        result = dict()
        for key in self.obs[0].keys():
            result[key] = self.stack_last_n_obs(
                [obs[key] for obs in self.obs],
                self.n_obs_steps
            )

        return result

    def get_action(self, policy: BaseImagePolicy, observaton=None):
        device, dtype = policy.device, policy.dtype
        if observaton is not None:
            self.obs.append(observaton) # update
        obs = self.get_n_steps_obs()

        # create obs dict
        np_obs_dict = dict(obs)
        # separate VGGT-related fields
        vggt_obs_dict = {}
        vggt_obs_dict['head_cam'] = np_obs_dict.pop('vggt_head_cam')
        vggt_obs_dict['front_cam'] = np_obs_dict.pop('vggt_front_cam')

        # device transfer
        obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))

        # preprocess VGGT images
        if vggt_obs_dict:
            processed_vggt_obs = {'head_cam': [], 'front_cam': []}
            for i in range(self.n_obs_steps):
                processed_head_images = torch.from_numpy(preprocess_rgb(vggt_obs_dict['head_cam'][i])).to(device=device)
                processed_front_images = torch.from_numpy(preprocess_rgb(vggt_obs_dict['front_cam'][i])).to(device=device)
                processed_vggt_obs['head_cam'].append(processed_head_images)
                processed_vggt_obs['front_cam'].append(processed_front_images)
            processed_vggt_obs = {
                'head_cam': torch.stack(processed_vggt_obs['head_cam']),
                'front_cam': torch.stack(processed_vggt_obs['front_cam'])
            }
            vggt_obs_dict = processed_vggt_obs
        
        # run policy
        with torch.no_grad():
            obs_dict_input = {}  # flush unused keys
            obs_dict_input['head_cam'] = obs_dict['head_cam'].unsqueeze(0)
            obs_dict_input['front_cam'] = obs_dict['front_cam'].unsqueeze(0)
            obs_dict_input['left_cam'] = obs_dict['left_cam'].unsqueeze(0)
            obs_dict_input['right_cam'] = obs_dict['right_cam'].unsqueeze(0)
            obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
            vggt_obs_dict_input = {}
            vggt_obs_dict_input['head_cam'] = vggt_obs_dict['head_cam'].unsqueeze(0)
            vggt_obs_dict_input['front_cam'] = vggt_obs_dict['front_cam'].unsqueeze(0)
            
            action_dict = policy.predict_action(obs_dict_input,vggt_obs_dict_input)

        # device_transfer
        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy() if isinstance(x, torch.Tensor) else x)
        action = np_action_dict['action'].squeeze(0)
        return action

    def clear_state(self, policy: BaseImagePolicy):
        # print(f'Clear state')
        policy.obs_encoder.step = 0
        policy.obs_encoder.current_vggt_features = None

