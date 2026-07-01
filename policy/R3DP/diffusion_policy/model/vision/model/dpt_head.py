import torch
import torch.nn as nn
import sys
import os
from typing import List, Dict, Tuple, Union
import numpy as np
from vggt.heads.dpt_head import DPTHead
from .vggt_encoder import VGGTEncoder
from PIL import Image
import torchvision.transforms.functional as TF


def _vggt_preprocess_image(rgb_image: np.ndarray, vggt_mode: str = "crop") -> torch.Tensor:
    target_size = 518
    
    img = Image.fromarray(rgb_image)

    # If there's an alpha channel, blend onto white background:
    if img.mode == "RGBA":
        # Create white background
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        # Alpha composite onto the white background
        img = Image.alpha_composite(background, img)

    # Now convert to "RGB" (this step assigns white for transparent areas)
    img = img.convert("RGB")

    height, width = rgb_image.shape[:2]
    
    if vggt_mode == "pad":
        if height < width:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14
        else:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
    else:  # mode == "crop"
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
    
    # Resize
    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    img = TF.to_tensor(img)
    
    # Center crop height if necessary (crop mode only)
    if vggt_mode == "crop" and new_height > target_size:
        start_y = (new_height - target_size) // 2
        img = img[:, start_y : start_y + target_size, :]
    
    # Pad to square if necessary (pad mode only)
    if vggt_mode == "pad":
        h_padding = target_size - img.shape[1]
        w_padding = target_size - img.shape[2]
        
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            
            # Pad with white (value=1.0)
            img = torch.nn.functional.pad(
                img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
            )
    
    return img

class VGGTDepthHead(DPTHead):
    """
    VGGT Depth Head that only loads the depth head part of VGGT model.
    Inherits from DPTHead and filters out unnecessary weights.
    """
    
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, **kwargs):
        """
        Initialize the VGGT Depth Head.
        
        Args:
            img_size (int): Input image size. Default: 518
            patch_size (int): Patch size for the vision transformer. Default: 14
            embed_dim (int): Embedding dimension. Default: 1024
            **kwargs: Additional arguments passed to DPTHead
        """
        # Initialize with the same parameters as VGGT's depth head
        super().__init__(
            dim_in=2 * embed_dim,  # VGGT uses 2 * embed_dim for depth head
            patch_size=patch_size,
            output_dim=2,  # VGGT depth head outputs 2 channels (depth + confidence)
            activation="exp",  # VGGT uses "exp" activation for depth
            conf_activation="expp1",  # VGGT uses "expp1" for confidence
            **kwargs
        )
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
    
    def load_depth_head_weights(self, checkpoint_path: str):
        """
        Load only the depth head weights from a full VGGT checkpoint.
        
        Args:
            checkpoint_path (str): Path to the VGGT checkpoint file
        """
        # Load the full checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Filter out non-depth-head weights
        depth_head_state_dict = {}
        for key, value in checkpoint.items():
            # Only keep weights that belong to the depth head
            if key.startswith('depth_head.'):
                # Remove the 'depth_head.' prefix
                new_key = key[len('depth_head.'):]
                depth_head_state_dict[new_key] = value
        
        # Load the filtered weights
        missing_keys, unexpected_keys = self.load_state_dict(depth_head_state_dict, strict=False)
        
        print(f"Loaded {len(depth_head_state_dict)} depth head weights from {checkpoint_path}")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        
        return len(depth_head_state_dict) > 0
    
    def forward(self, aggregated_tokens_list: List[torch.Tensor], images: torch.Tensor, patch_start_idx: int):
        """
        Forward pass that returns depth predictions.
        
        Args:
            aggregated_tokens_list (List[torch.Tensor]): List of token tensors from different transformer layers.
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            patch_start_idx (int): Starting index for patch tokens in the token sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - depth: Predicted depth maps with shape [B, S, 1, H, W]
                - depth_conf: Confidence scores for depth predictions with shape [B, S, 1, H, W]
        """
        return super().forward(aggregated_tokens_list, images, patch_start_idx)
    
    @classmethod
    def from_vggt_checkpoint(cls, checkpoint_path: str, **kwargs):
        """
        Create a VGGTDepthHead instance and load weights from a VGGT checkpoint.
        
        Args:
            checkpoint_path (str): Path to the VGGT checkpoint file
            **kwargs: Additional arguments for initialization
            
        Returns:
            VGGTDepthHead: Initialized depth head with loaded weights
        """
        # Create instance
        depth_head = cls(**kwargs)
        
        # Load weights
        success = depth_head.load_depth_head_weights(checkpoint_path)
        if not success:
            print(f"Warning: No depth head weights found in {checkpoint_path}")
        
        return depth_head
    
    def get_depth_head_state_dict(self):
        """
        Get the state dict containing only depth head parameters.
        
        Returns:
            Dict[str, torch.Tensor]: State dictionary with depth head parameters
        """
        return self.state_dict()
    
    def save_depth_head_weights(self, save_path: str):
        """
        Save only the depth head weights to a file.
        
        Args:
            save_path (str): Path where to save the depth head weights
        """
        depth_head_state_dict = self.get_depth_head_state_dict()
        torch.save(depth_head_state_dict, save_path)
        print(f"Saved {len(depth_head_state_dict)} depth head weights to {save_path}")


def load_vggt_depth_head(checkpoint_path: str, **kwargs):
    """
    Convenience function to load a VGGT depth head from checkpoint.
    
    Args:
        checkpoint_path (str): Path to the VGGT checkpoint file
        **kwargs: Additional arguments for VGGTDepthHead initialization
        
    Returns:
        VGGTDepthHead: Initialized depth head with loaded weights
    """
    return VGGTDepthHead.from_vggt_checkpoint(checkpoint_path, **kwargs)
