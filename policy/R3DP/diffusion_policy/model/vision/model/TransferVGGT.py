import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import sys
import os

from vggt.layers.vision_transformer import vit_small
from vggt.layers.block import Block
from vggt.layers.attention import Attention
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from .CrossAttentionBlock import CrossAttentionBlock

class TransferVGGT(nn.Module):
    """
    TransferVGGT model that transfers VGGT features across time.
    
    The model takes current images and historical VGGT features as input,
    and outputs current VGGT features by:
    1. Extracting 2D visual features using DINOv2-small
    2. Processing through AA blocks (frame + global attention) 
    3. Cross-attending to historical VGGT features at each layer
    4. Outputting 4 layers of features as current VGGT features
    """
    
    def __init__(
        self,
        img_size: int = 308,
        patch_size: int = 14,
        embed_dim: int = 1024,
        num_heads: int = 16,
        num_aa_blocks: int = 4,
        num_register_tokens: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        qk_norm: bool = True,
        rope_freq: int = 100,
        init_values: float = 0.01,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        dinov2_model_name: str = "dinov2_vits14_reg",
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_aa_blocks = num_aa_blocks
        self.num_register_tokens = num_register_tokens
        
        # Compute patch grid size
        self.patch_grid_size = img_size // patch_size
        self.num_patches = self.patch_grid_size ** 2
        
        # 1. DINOv2 backbone for 2D visual feature extraction
        self.dinov2_backbone = vit_small(
            img_size=img_size,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            block_chunks=0,  # Disable block chunking to match standard DINOv2 structure
        )
        
        # Load DINOv2 pretrained weights
        self._load_dinov2_weights()
        
        # Feature dimension adapter: DINOv2-small outputs 384-dim features, we need embed_dim
        self.dinov2_embed_dim = 384  # DINOv2-small embedding dimension
        self.feature_adapter = nn.Linear(self.dinov2_embed_dim, embed_dim) if self.dinov2_embed_dim != embed_dim else nn.Identity()
        
        # Freeze DINOv2 backbone initially (can be unfrozen later for fine-tuning)
        for param in self.dinov2_backbone.parameters():
            param.requires_grad = False
        
        # 2. Initialize rotary position embedding
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None
        
        # 3. AA blocks (frame and global attention)
        self.frame_blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                init_values=init_values,
                qk_norm=qk_norm,
                rope=self.rope,
                attn_drop=attn_drop,
                drop=proj_drop,
            ) for _ in range(num_aa_blocks)
        ])
        
        self.global_blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                init_values=init_values,
                qk_norm=qk_norm,
                rope=self.rope,
                attn_drop=attn_drop,
                drop=proj_drop,
            ) for _ in range(num_aa_blocks)
        ])
        
        # 4. Cross attention blocks for attending to historical features
        self.cross_attention_blocks = nn.ModuleList([
            CrossAttentionBlock(
                dim=2*embed_dim,  # Use 2*embed_dim to match concatenated features
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                qk_norm=qk_norm,
            ) for _ in range(num_aa_blocks)
        ])
        
        # 5. Special tokens (camera and register tokens)
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
        
        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens
        
        # Initialize special tokens with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)
        
        # Normalization constants for input images
        self.register_buffer("_resnet_mean", torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("_resnet_std", torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1), persistent=False)
    
    def _load_dinov2_weights(self):
        """
        Load pretrained DINOv2 weights from checkpoint file
        """
        r3dp_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        pretrain_path = os.path.join(r3dp_root, "tvggt/pretrain_ckpt/dinov2_vits14_reg4_pretrain.pth")

        if not os.path.exists(pretrain_path):
            print(f"Warning: DINOv2 pretrained weights not found at {pretrain_path}")
            print("Using randomly initialized weights for DINOv2 backbone")
            return

        try:
            # Load checkpoint
            checkpoint = torch.load(pretrain_path, map_location='cpu')
            
            # Extract state dict (handle different checkpoint formats)
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Filter out keys that don't match our model
            model_state_dict = self.dinov2_backbone.state_dict()
            # print("state_dict.keys():", state_dict.keys())
            # print("model_state_dict.keys():", model_state_dict.keys())
            filtered_state_dict = {}
            
            for key, value in state_dict.items():
                # Remove any prefix that might be present
                clean_key = key
                if clean_key.startswith('module.'):
                    clean_key = clean_key[7:]  # Remove 'module.' prefix
                if clean_key.startswith('backbone.'):
                    clean_key = clean_key[9:]  # Remove 'backbone.' prefix
                
                # Skip LayerScale parameters as our model doesn't have them
                if 'ls1.gamma' in clean_key or 'ls2.gamma' in clean_key:
                    continue
                
                if clean_key in model_state_dict:
                    if value.shape == model_state_dict[clean_key].shape:
                        filtered_state_dict[clean_key] = value
                    else:
                        print(f"Shape mismatch for {clean_key}: expected {model_state_dict[clean_key].shape}, got {value.shape}")
                else:
                    # Only print error for important keys, skip LayerScale
                    if 'ls1.gamma' not in key and 'ls2.gamma' not in key:
                        print(f"Key {clean_key} not found in model")
            
            # Load the filtered state dict
            missing_keys, unexpected_keys = self.dinov2_backbone.load_state_dict(filtered_state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys in DINOv2 backbone: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys in DINOv2 backbone: {unexpected_keys}")
            
            print(f"Successfully loaded DINOv2 pretrained weights from {pretrain_path}")
            print(f"Loaded {len(filtered_state_dict)} parameters")
            
        except Exception as e:
            print(f"Error loading DINOv2 pretrained weights: {e}")
            print("Using randomly initialized weights for DINOv2 backbone")
    
    def _extract_visual_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract 2D visual features using DINOv2 backbone
        
        Args:
            images: [B, S, 3, H, W] input images
            
        Returns:
            features: [B*S, num_patches, embed_dim] patch features
        """
        B, S, C, H, W = images.shape
        
        # Normalize images
        images = (images - self._resnet_mean) / self._resnet_std
        
        # Reshape for backbone processing
        images = images.view(B * S, C, H, W)
        
        # Extract features using DINOv2
        with torch.no_grad():  # DINOv2 is frozen
            features = self.dinov2_backbone(images)
            if isinstance(features, dict):
                features = features["x_norm_patchtokens"]
        
        # Adapt feature dimensions if necessary
        features = self.feature_adapter(features)
        
        return features
    
    def _add_special_tokens(self, patch_tokens: torch.Tensor, B: int, S: int) -> torch.Tensor:
        """
        Add camera and register tokens to patch tokens
        
        Args:
            patch_tokens: [B*S, num_patches, embed_dim]
            B: batch size
            S: sequence length
            
        Returns:
            tokens: [B*S, total_tokens, embed_dim]
        """
        # Expand camera and register tokens
        camera_token = self._slice_expand_and_flatten(self.camera_token, B, S)
        register_token = self._slice_expand_and_flatten(self.register_token, B, S)
        
        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        
        return tokens
    
    def _slice_expand_and_flatten(self, token_tensor: torch.Tensor, B: int, S: int) -> torch.Tensor:
        """
        Process specialized tokens for multi-frame processing
        Similar to the original VGGT implementation
        """
        # Use first position for first frame, second position for other frames
        query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
        others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
        combined = torch.cat([query, others], dim=1)
        
        # Flatten for processing
        combined = combined.view(B * S, *combined.shape[2:])
        return combined
    
    def _get_position_embeddings(self, B: int, S: int, H: int, W: int, device: torch.device) -> Optional[torch.Tensor]:
        """
        Get rotary position embeddings if enabled
        Similar to VGGT's Aggregator implementation
        """
        if self.rope is None:
            return None
        
        # Calculate patch grid size from image dimensions (like VGGT)
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size
        
        pos = self.position_getter(B * S, patch_h, patch_w, device=device)
        
        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos = pos.to(device)
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
        
        return pos
    
    def forward(
        self, 
        current_images: torch.Tensor, 
        historical_features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Forward pass of TransferVGGT
        
        Args:
            current_images: [B, S, 3, H, W] current frame images
            historical_features: List of 4 tensors [B, S, N, C] representing historical VGGT features
            
        Returns:
            output_features: List of 4 tensors [B, S, N, C] representing current VGGT features
        """
        B, S, C, H, W = current_images.shape
        
        # Validate inputs
        assert len(historical_features) == self.num_aa_blocks, f"Expected {self.num_aa_blocks} historical feature layers"
        
        # 1. Extract 2D visual features using DINOv2
        patch_tokens = self._extract_visual_features(current_images)  # [B*S, num_patches, embed_dim]
        
        # 2. Add special tokens
        tokens = self._add_special_tokens(patch_tokens, B, S)  # [B*S, total_tokens, embed_dim]
        
        # 3. Get position embeddings
        pos = self._get_position_embeddings(B, S, H, W, current_images.device)
        
        # Get token dimensions
        _, total_tokens, embed_dim = tokens.shape
        
        # 4. Process through AA blocks with cross attention
        output_features = []
        frame_idx = 0
        global_idx = 0
        
        for layer_idx in range(self.num_aa_blocks):
            # Frame attention
            tokens, frame_idx, frame_intermediates = self._process_frame_attention(tokens, B, S, total_tokens, embed_dim, frame_idx, pos)
            
            # Global attention 
            tokens, global_idx, global_intermediates = self._process_global_attention(tokens, B, S, total_tokens, embed_dim, global_idx, pos)
            
            # Process intermediates (like VGGT Aggregator)
            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                
                # Cross attention with historical features
                hist_feat = historical_features[layer_idx]  # [B, S, N_hist, 2C]
                
                # Apply cross attention (use full concat_inter with same dimension as historical features)
                enhanced_features = self.cross_attention_blocks[layer_idx](concat_inter, hist_feat)
                
                # Store the enhanced features directly (already 2C dimension)
                final_features = enhanced_features  # [B, S, P, 2C]
                
                # Store output for this layer
                output_features.append(final_features)
                
                # Update tokens for next iteration (use frame part of enhanced features)
                frame_part = enhanced_features[..., :embed_dim]  # [B, S, P, C]
                tokens = frame_part.view(B * S, total_tokens, embed_dim)
        
        return output_features
    
    def _process_frame_attention(
        self, 
        tokens: torch.Tensor, 
        B: int, S: int, total_tokens: int, embed_dim: int, 
        frame_idx: int, 
        pos: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        Similar to VGGT's Aggregator._process_frame_attention
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, total_tokens, embed_dim):
            tokens = tokens.view(B * S, total_tokens, embed_dim)

        if pos is not None and pos.shape != (B * S, total_tokens, 2):
            pos = pos.view(B * S, total_tokens, 2)

        intermediates = []

        # Process one block at a time (aa_block_size=1 equivalent)
        if self.training:
            from torch.utils.checkpoint import checkpoint
            tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=False)
        else:
            tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
        frame_idx += 1
        intermediates.append(tokens.view(B, S, total_tokens, embed_dim))

        return tokens, frame_idx, intermediates
    
    def _process_global_attention(
        self, 
        tokens: torch.Tensor, 
        B: int, S: int, total_tokens: int, embed_dim: int, 
        global_idx: int, 
        pos: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        Similar to VGGT's Aggregator._process_global_attention
        """
        if tokens.shape != (B, S * total_tokens, embed_dim):
            tokens = tokens.view(B, S, total_tokens, embed_dim).view(B, S * total_tokens, embed_dim)

        if pos is not None and pos.shape != (B, S * total_tokens, 2):
            pos = pos.view(B, S, total_tokens, 2).view(B, S * total_tokens, 2)

        intermediates = []

        # Process one block at a time (aa_block_size=1 equivalent)
        if self.training:
            from torch.utils.checkpoint import checkpoint
            tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=False)
        else:
            tokens = self.global_blocks[global_idx](tokens, pos=pos)
        global_idx += 1
        intermediates.append(tokens.view(B, S, total_tokens, embed_dim))

        return tokens, global_idx, intermediates


if __name__ == "__main__":
    # Test the model
    print("Testing TransferVGGT model...")
    
    # Create model
    model = TransferVGGT(
        img_size=308,
        patch_size=14,
        embed_dim=1024,
        num_heads=16,
        num_aa_blocks=4,
        num_register_tokens=4
    )
    model.train()
    # Create dummy inputs
    batch_size = 2
    seq_len = 5
    current_images = torch.randn(batch_size, seq_len, 3, 168, 308)
    
    # Create dummy historical features (4 layers)
    historical_features = []
    for i in range(4):
        # Each layer: [B, S, N, C] where N includes special tokens + patches
        feat = torch.randn(batch_size, seq_len, 37*37 + 5, 1024)  # +5 for special tokens
        historical_features.append(feat)
    
    # Test forward pass
    print("Running forward pass...")
    with torch.no_grad():
        output_features = model(current_images, historical_features)
    
    # Check outputs
    print(f"Number of output layers: {len(output_features)}")
    for i, feat in enumerate(output_features):
        print(f"Layer {i} shape: {feat.shape}")
        print(f"  Expected: [B={batch_size}, S={seq_len}, N=tokens, C=2*embed_dim={2*1024}]")
    
    print("✓ Model test completed successfully!")
