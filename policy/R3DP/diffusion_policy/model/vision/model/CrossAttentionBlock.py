import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import sys
import os

class CrossAttentionBlock(nn.Module):
    """
    Cross attention block for attending to historical VGGT features
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query projection (for current features)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        # Key and Value projections (for historical features)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Normalization layers
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        
        # Dropout and output projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Layer norm for residual connection
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP for post-attention processing
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(proj_drop)
        )
    
    def forward(self, current_features: torch.Tensor, historical_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            current_features: [B, S, N_current, C] - current frame features
            historical_features: [B, S, N_hist, C] - historical VGGT features
        
        Returns:
            updated_features: [B, S, N_current, C] - features after cross attention
        """
        B, S, N_current, C = current_features.shape
        B_hist, S_hist, N_hist, C_hist = historical_features.shape
        
        # Validate batch and sequence dimensions match
        assert B == B_hist and S == S_hist, f"Batch/sequence mismatch: current {(B, S)} vs historical {(B_hist, S_hist)}"
        assert C == C_hist, f"Feature dimension mismatch: current {C} vs historical {C_hist}"
        
        # print("historical_features shape: ", historical_features.shape)
        # print("current_features shape: ", current_features.shape)
        
        # Flatten spatial dimensions for attention
        current_flat = current_features.view(B * S, N_current, C)
        historical_flat = historical_features.view(B * S, N_hist, C)
        
        # Apply layer norm
        current_norm = self.norm1(current_flat)
        historical_norm = self.norm1(historical_flat)
        
        # Compute Q, K, V
        q = self.q_proj(current_norm).reshape(B * S, N_current, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(historical_norm).reshape(B * S, N_hist, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(historical_norm).reshape(B * S, N_hist, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Scaled dot-product attention
        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0
        )
        
        # Reshape and project
        attn = attn.transpose(1, 2).reshape(B * S, N_current, C)
        attn = self.proj(attn)
        attn = self.proj_drop(attn)
        
        # First residual connection
        current_flat = current_flat + attn
        
        # MLP with second residual connection
        mlp_out = self.mlp(self.norm2(current_flat))
        current_flat = current_flat + mlp_out
        
        # Reshape back
        return current_flat.view(B, S, N_current, C)
