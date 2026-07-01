import pickle, os
import numpy as np
import pdb
from copy import deepcopy
import zarr
import shutil
import argparse
import einops
import cv2
import torch
import psutil

target_size = 308
patch_size = 14
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_memory_usage():
    mem = psutil.virtual_memory()
    used_memory = mem.used / (1024 ** 2) # MB
    memory_percent = mem.percent
    return used_memory, memory_percent

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


def main():
    parser = argparse.ArgumentParser(description='Process some episodes.')
    parser.add_argument('task_name', type=str, default='block_hammer_beat',
                        help='The name of the task (e.g., block_hammer_beat)')
    parser.add_argument('head_camera_type', type=str)
    parser.add_argument('expert_data_num', type=int, default=50,
                        help='Number of episodes to process (e.g., 50)')
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    head_camera_type = args.head_camera_type
    load_dir = f'./data/{task_name}_{head_camera_type}_pkl'
    print(f'load_dir: {load_dir}')

    total_count = 0

    save_dir = f'./policy/R3DP/data/{task_name}_{head_camera_type}_{num}.zarr'
    print(f'save_dir: {save_dir}')
    
    if os.path.exists(load_dir):
        print('load_dir valid')

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    current_ep = 0

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    head_camera_arrays, front_camera_arrays, vggt_head_camera_arrays, vggt_front_camera_arrays = [], [], [], []
    episode_ends_arrays, action_arrays, state_arrays, joint_action_arrays = [], [], [], []
    
    while os.path.isdir(load_dir+f'/episode{current_ep}') and current_ep < num:
        used_memory, memory_percent = get_memory_usage()
        print(f'processing episode: {current_ep + 1} / {num} used memory: {used_memory:.2f}MB memory percent: {memory_percent}%', end='\r')
        file_num = 0
        
        while os.path.exists(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl'):
            with open(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl', 'rb') as file:
                data = pickle.load(file)
            
            head_img = data['observation']['head_camera']['rgb']
            front_img = data['observation']['front_camera']['rgb']
            vggt_head_img = preprocess_rgb(data['observation']['head_camera']['rgb'], mode="crop")
            vggt_front_img = preprocess_rgb(data['observation']['front_camera']['rgb'], mode="crop")
            action = data['endpose']
            joint_action = data['joint_action']

            head_camera_arrays.append(head_img)
            front_camera_arrays.append(front_img)
            vggt_head_camera_arrays.append(vggt_head_img)
            vggt_front_camera_arrays.append(vggt_front_img)
            action_arrays.append(action)
            state_arrays.append(joint_action)
            joint_action_arrays.append(joint_action)

            file_num += 1
            total_count += 1
            
        current_ep += 1

        episode_ends_arrays.append(total_count)

    print()
    episode_ends_arrays = np.array(episode_ends_arrays)
    action_arrays = np.array(action_arrays)
    state_arrays = np.array(state_arrays)
    head_camera_arrays = np.array(head_camera_arrays)
    front_camera_arrays = np.array(front_camera_arrays)
    joint_action_arrays = np.array(joint_action_arrays)
    vggt_head_camera_arrays = np.array(vggt_head_camera_arrays)
    vggt_front_camera_arrays = np.array(vggt_front_camera_arrays)
    head_camera_arrays = np.moveaxis(head_camera_arrays, -1, 1)  # NHWC -> NCHW
    front_camera_arrays = np.moveaxis(front_camera_arrays, -1, 1)  # NHWC -> NCHW

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    action_chunk_size = (100, action_arrays.shape[1])
    state_chunk_size = (100, state_arrays.shape[1])
    joint_chunk_size = (100, joint_action_arrays.shape[1])
    head_camera_chunk_size = (100, *head_camera_arrays.shape[1:])
    front_camera_chunk_size = (100, *front_camera_arrays.shape[1:])
    vggt_head_camera_chunk_size = (100, *vggt_head_camera_arrays.shape[1:])
    vggt_front_camera_chunk_size = (100, *vggt_front_camera_arrays.shape[1:])
    zarr_data.create_dataset('head_camera', data=head_camera_arrays, chunks=head_camera_chunk_size, overwrite=True, compressor=compressor)
    zarr_data.create_dataset('front_camera', data=front_camera_arrays, chunks=front_camera_chunk_size, overwrite=True, compressor=compressor)
    zarr_data.create_dataset('vggt_head_camera', data=vggt_head_camera_arrays, chunks=vggt_head_camera_chunk_size, overwrite=True, compressor=compressor)
    zarr_data.create_dataset('vggt_front_camera', data=vggt_front_camera_arrays, chunks=vggt_front_camera_chunk_size, overwrite=True, compressor=compressor)
    zarr_data.create_dataset('tcp_action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=joint_action_arrays, chunks=joint_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

if __name__ == '__main__':
    main()

# python script/pkl2zarr_r3dp.py block_hammer_beat L515 100
