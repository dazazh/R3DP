import torch
import torch.nn as nn
import sys
import os
from vggt.models.aggregator import Aggregator

class VGGTEncoder(Aggregator):
    """
    VGGT Encoder that only loads the aggregator part of VGGT model.
    Inherits from Aggregator and filters out unnecessary weights.
    """
    
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, **kwargs):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            **kwargs
        )
    
    def load_aggregator_weights(self, checkpoint_path):
        """
        Load only the aggregator weights from a full VGGT checkpoint.
        
        Args:
            checkpoint_path (str): Path to the VGGT checkpoint file
        """
        # Load the full checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Filter out non-aggregator weights
        aggregator_state_dict = {}
        for key, value in checkpoint.items():
            # Only keep weights that belong to the aggregator
            if key.startswith('aggregator.'):
                # Remove the 'aggregator.' prefix
                new_key = key[len('aggregator.'):]
                aggregator_state_dict[new_key] = value
        
        # Load the filtered weights
        missing_keys, unexpected_keys = self.load_state_dict(aggregator_state_dict, strict=False)
        
        print(f"Loaded {len(aggregator_state_dict)} aggregator weights from {checkpoint_path}")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
    
    def forward(self, images: torch.Tensor):
        """
        Forward pass that returns only the aggregator outputs.
        
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        return super().forward(images)