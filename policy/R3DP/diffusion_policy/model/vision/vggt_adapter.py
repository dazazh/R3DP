import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

from .prope import PropeDotProductAttention

class VGGTAdapterGrouped(nn.Module):
    """
    Split [B,8,777,2048] into head (frames [0,2,4,6]) and front (frames [1,3,5,7]),
    project to D dims, and run token-level cross-attention per frame.
    Output: view_tokens [B, 2, frames, patches, D]
    """
    def __init__(self, input_dim=2048, proj_dim=512, num_heads=8):
        super().__init__()
        self.proj_dim = proj_dim

        # key/value projections (head & front separately)
        self.vggt_head_key_proj = nn.Linear(input_dim, proj_dim)
        self.vggt_head_value_proj = nn.Linear(input_dim, proj_dim)
        self.vggt_front_key_proj = nn.Linear(input_dim, proj_dim)
        self.vggt_front_value_proj = nn.Linear(input_dim, proj_dim)

        # query projections (global query per camera)
        self.head_proj = nn.Linear(proj_dim, proj_dim)
        self.front_proj = nn.Linear(proj_dim, proj_dim)

        # cross-attention, one output vector per token
        self.cross_attn_head = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn_front = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=num_heads, batch_first=True)

    def forward(self, head_feat, front_feat, vggt_feat):
        """
        head_feat:  [B, proj_dim]   global query for the head camera
        front_feat: [B, proj_dim]   global query for the front camera
        vggt_feat:  [B, 8, patches, input_dim]   (patches=777)
        Returns:
            view_tokens: [B, 2, frames, patches, D]   (frames fixed to 4)
        """
        B = vggt_feat.shape[0]
        _, n8, patches, in_dim = vggt_feat.shape
        assert n8 == 8, "Expect second dim==8 (4 frames per camera * 2 cameras)"

        # indices for frames per camera
        head_indices = [0, 2, 4, 6]
        front_indices = [1, 3, 5, 7]
        frames = len(head_indices)

        # extract per-frame token lists: each [B, patches, in_dim]
        head_frames = [vggt_feat[:, i, :, :] for i in head_indices]    # len=4
        front_frames = [vggt_feat[:, i, :, :] for i in front_indices]  # len=4

        proj_dim = self.proj_dim

        # process head frames
        head_view_tokens = []
        for f in range(frames):
            tokens = head_frames[f]  # [B, patches, in_dim]
            K = self.vggt_head_key_proj(tokens)    # [B, patches, D]
            V = self.vggt_head_value_proj(tokens)  # [B, patches, D]
            # query repeat to match patches -> q: [B, patches, D]
            q = self.head_proj(head_feat).unsqueeze(1).expand(-1, K.size(1), -1)
            out, _ = self.cross_attn_head(q, K, V)  # [B, patches, D]
            head_view_tokens.append(out.unsqueeze(1))  # [B,1,patches,D]

        # stack -> [B, frames, patches, D]
        head_view_tokens = torch.cat(head_view_tokens, dim=1)

        # process front frames
        front_view_tokens = []
        for f in range(frames):
            tokens = front_frames[f]
            K = self.vggt_front_key_proj(tokens)
            V = self.vggt_front_value_proj(tokens)
            q = self.front_proj(front_feat).unsqueeze(1).expand(-1, K.size(1), -1)
            out, _ = self.cross_attn_front(q, K, V)
            front_view_tokens.append(out.unsqueeze(1))

        front_view_tokens = torch.cat(front_view_tokens, dim=1)  # [B, frames, patches, D]

        # combine to [B, 2, frames, patches, D]
        view_tokens = torch.stack([head_view_tokens, front_view_tokens], dim=1)
        return view_tokens  # [B, 2, frames, patches, D]

def get_K_Tcw(batch_size, device):
    K_head = torch.tensor([
        [217.27922,   0.0,      160.0],
        [  0.0,     217.27922,   90.0],
        [  0.0,       0.0,        1.0],
    ], dtype=torch.float32, device=device)

    K_front = torch.tensor([
        [217.27922,   0.0,      160.0],
        [  0.0,     217.27922,   90.0],
        [  0.0,       0.0,        1.0],
    ], dtype=torch.float32, device=device)

    # extrinsics 3x4 -> 4x4 (world -> cam, i.e. T_cw)
    E_head_3x4 = torch.tensor([
        [ 1.0000000e+00,  1.1920929e-07, -5.9604645e-08,  3.2000184e-02],
        [ 5.9604645e-08, -8.0000007e-01, -5.9999990e-01,  4.5000005e-01],
        [-1.1920929e-07,  5.9999996e-01, -8.0000007e-01,  1.3500001e+00],
    ], dtype=torch.float32, device=device)

    E_front_3x4 = torch.tensor([
        [ 1.0000001e+00,  1.1920929e-07, -3.7252903e-09,  5.9604645e-08],
        [ 3.7252903e-09, -9.9503726e-02, -9.9503720e-01,  8.0100501e-01],
        [-1.1920929e-07,  9.9503726e-01, -9.9503726e-02,  5.3234494e-01],
    ], dtype=torch.float32, device=device)

    def to_4x4(E_3x4: torch.Tensor) -> torch.Tensor:
        T = torch.eye(4, dtype=E_3x4.dtype, device=device)
        T[:3, :4] = E_3x4
        return T

    Tcw_head = to_4x4(E_head_3x4)   # [4,4]
    Tcw_front = to_4x4(E_front_3x4) # [4,4]

    B = batch_size
    # shared intrinsics across frames: use frame dim 1, expanded to frames inside the module
    Ks = torch.stack([K_head, K_front], dim=0).unsqueeze(0).unsqueeze(2).expand(B, 2, 1, 3, 3).contiguous()
    Tcw = torch.stack([Tcw_head, Tcw_front], dim=0).unsqueeze(0).unsqueeze(2).expand(B, 2, 1, 4, 4).contiguous()

    return Ks, Tcw


class MultiViewFusionPRoPE(ModuleAttrMixin):
    """
    Cross-view fusion with PRoPE plus attention pooling.
    """
    def __init__(self,
                 embed_dim=512,
                 num_heads=8,
                 patches=264,
                 frames=4,
                 prope_cfg=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patches = patches
        self.frames = frames
        self.num_views = 2  # fixed: head/front

        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads

        # Q/K/V projection
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.prope_attn = PropeDotProductAttention(head_dim=self.head_dim, **prope_cfg)

        # attention pooling query per view
        self.pool_query = nn.Parameter(torch.randn(self.num_views, embed_dim))
        self.pool_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, view_tokens: torch.Tensor):
        """
        view_tokens: [B, 2, frames, patches, D]
        Returns:
            per_view_tokens: [B, 2, frames*patches, D]
            fused_views:     [B, 2, D]
        """
        B, N_view, F, P, D = view_tokens.shape
        assert N_view == 2 and F == self.frames and P == self.patches and D == self.embed_dim

        view_tokens = view_tokens.reshape(B, 2*F, P, D)

        Ks_frames, Tcw_frames = get_K_Tcw(B, self.device)

        # expand shared camera matrices across frames
        if Ks_frames.shape[2] == 1:
            Ks_frames = Ks_frames.expand(-1, -1, F, -1, -1)  # [B, 2, frames, 3, 3]
        if Tcw_frames.shape[2] == 1:
            Tcw_frames = Tcw_frames.expand(-1, -1, F, -1, -1)  # [B, 2, frames, 4, 4]

        # flatten (view, frame, patch) -> seq
        seqlen = N_view * F * P
        seq = view_tokens.reshape(B, seqlen, D)

        Q = self.q_proj(seq)
        K = self.k_proj(seq)
        V = self.v_proj(seq)

        def to_heads(x):
            b, s, d = x.shape
            return x.view(b, s, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous().to(self.device)
        qh = to_heads(Q)
        kh = to_heads(K)
        vh = to_heads(V)

        # expand camera matrices to token level
        Ks_flat = Ks_frames.reshape(B, N_view * F, 3, 3).to(self.device)
        Tcw_flat = Tcw_frames.reshape(B, N_view * F, 4, 4).to(self.device)

        # PRoPE attention
        prope_out_heads = self.prope_attn(qh, kh, vh, Ks=Ks_flat, viewmats=Tcw_flat)
        prope_out = prope_out_heads.permute(0, 2, 1, 3).reshape(B, seqlen, D)
        prope_out = self.out_proj(prope_out)

        # reshape to [B, 2, frames*patches, D] for per-view pooling
        per_view_tokens = prope_out.view(B, N_view, F * P, D)

        # attention pooling per view
        fused_views = []
        for v in range(N_view):
            # learnable query shared within a view: [B, 1, D]
            q = self.pool_query[v].unsqueeze(0).expand(B, -1).unsqueeze(1)  # [B,1,D]
            out, _ = self.pool_attn(q, per_view_tokens[:, v], per_view_tokens[:, v])
            fused_views.append(out.squeeze(1))  # [B,D]

        fused_views = torch.stack(fused_views, dim=1)  # [B,2,D]

        return per_view_tokens, fused_views
