from typing import Dict
import numba
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class RobotImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            batch_size=16,
            batch_combined=2,
            max_train_episodes=None
            ):
        
        super().__init__()
        print(f'[INFO] Loading Data from {zarr_path}')
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            # keys=['head_camera', 'front_camera', 'left_camera', 'right_camera', 'state', 'action'],
            keys=['head_camera', 'front_camera', 'state', 'action', 'vggt_head_camera', 'vggt_front_camera']
        )
        print(f'[INFO] Finish Loading Data')
            
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)


        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        batch_size = batch_size * batch_combined
        self.batch_size = batch_size
        self.batch_combined = batch_combined
        # print(f'batch_size: {self.batch_size}')
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        print("self.buffers['head_camera'].shape: ", self.buffers['head_camera'].shape)
        print("self.buffers_torch['head_camera'].shape: ", self.buffers_torch['head_camera'].shape)

        for v in self.buffers_torch.values():
            v.pin_memory()

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['head_cam'] = get_image_range_normalizer()
        normalizer['front_cam'] = get_image_range_normalizer()
        normalizer['left_cam'] = get_image_range_normalizer()
        normalizer['right_cam'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'].astype(np.float32) # (agent_posx2, block_posex3)
        head_cam = np.moveaxis(sample['head_camera'],-1,1)
        front_cam = np.moveaxis(sample['front_camera'],-1,1)
        # left_cam = np.moveaxis(sample['left_camera'],-1,1)/255
        # right_cam = np.moveaxis(sample['right_camera'],-1,1)/255

        data = {
            'obs': {
                'head_cam': head_cam, # T, 3, H, W
                'front_cam': front_cam, # T, 3, H, W
                # 'left_cam': left_cam, # T, 3, H, W
                # 'right_cam': right_cam, # T, 3, H, W
                'agent_pos': agent_pos, # T, D
            },
            'action': sample['action'].astype(np.float32) # T, D
        }
        return data

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # print(f'idx: {idx}')
        if isinstance(idx, slice):
            raise NotImplementedError  # Specialized
        elif isinstance(idx, int):
            sample = self.sampler.sample_sequence(idx)
            sample = dict_apply(sample, torch.from_numpy)
            return sample
        elif isinstance(idx, np.ndarray):
            assert len(idx) == self.batch_size
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(self.buffers[k], v, self.sampler.indices, idx, self.sampler.sequence_length)
            return self.buffers_torch
        else:
            raise ValueError(idx)
            
    def postprocess(self, samples):
        agent_pos = samples['state'].to(non_blocking=True)
        head_cam = samples['head_camera'].to(non_blocking=True) / 255.0
        front_cam = samples['front_camera'].to(non_blocking=True) / 255.0
        vggt_head_cam = samples['vggt_head_camera'].to(non_blocking=True)
        vggt_front_cam = samples['vggt_front_camera'].to(non_blocking=True)
        # left_cam = samples['left_camera'].to(device, non_blocking=True) / 255.0
        # right_cam = samples['right_camera'].to(device, non_blocking=True) / 255.0
        action = samples['action'].to(non_blocking=True)
        return {
            'obs': {
                'head_cam': head_cam , # B, T, 3, H, W
                'front_cam': front_cam, # B, T, 3, H, W
                'vggt_head_cam': vggt_head_cam, # B, T, 3, H, W
                'vggt_front_cam': vggt_front_cam, # B, T, 3, H, W
                # 'left_cam': left_cam, # B, T, 3, H, W
                # 'right_cam': right_cam, # B, T, 3, H, W
                'agent_pos': agent_pos, # B, T, D
            },
            'action': action # B, T, D
        }

def _batch_sample_sequence(data: np.ndarray, input_arr: np.ndarray, indices: np.ndarray, idx: np.ndarray, sequence_length: int):
    for i in numba.prange(len(idx)):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx, episode_idx = indices[idx[i]] # sdp_modify:episode_idx
        data[i, sample_start_idx:sample_end_idx] = input_arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = data[i, sample_start_idx]
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = data[i, sample_end_idx - 1]

_batch_sample_sequence_sequential = numba.jit(_batch_sample_sequence, nopython=True, parallel=False)
_batch_sample_sequence_parallel = numba.jit(_batch_sample_sequence, nopython=True, parallel=True)

def batch_sample_sequence(data: np.ndarray, input_arr: np.ndarray, indices: np.ndarray, idx: np.ndarray, sequence_length: int):
    batch_size = len(idx)
    # print(f'data shape: {data.shape}')
    # print(f'res: {batch_size, sequence_length, *input_arr.shape[1:]}')
    assert data.shape == (batch_size, sequence_length, *input_arr.shape[1:])
    if batch_size >= 16 and data.nbytes // batch_size >= 2 ** 16:
        _batch_sample_sequence_parallel(data, input_arr, indices, idx, sequence_length)
    else:
        _batch_sample_sequence_sequential(data, input_arr, indices, idx, sequence_length)
