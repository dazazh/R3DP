from typing import Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
import torchvision
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import replace_submodules
from .vggt_adapter import VGGTAdapterGrouped, MultiViewFusionPRoPE
from vggt.models.vggt import VGGT
import cv2
import numpy as np
import os
from .model.TransferVGGT import TransferVGGT
import logging

logger = logging.getLogger(__name__)

_R3DP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

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

dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

def extract_gt_features(vggt_model, img_tensor, device):
    vggt_model = vggt_model.to(device)
    with torch.no_grad():
        with torch.autocast(dtype=dtype, device_type='cuda'):
            img_tensor = img_tensor.to(device)
            aggregated_tokens_list, patch_start_idx = vggt_model(img_tensor) # [24] [B, V, N, D]
            intermediate_layer_idx = [4, 11, 17, 23]
            intermediate_tokens = [aggregated_tokens_list[intermediate_layer_idx[i]][:,:,patch_start_idx:] for i in range(len(intermediate_layer_idx))]
            intermediate_features = intermediate_tokens # [4] [B, V, N, D]
            current_features = intermediate_features
    return current_features, patch_start_idx

def extract_features(
    vggt_model, 
    transfer_model, 
    img_tensor, 
    batch_combined, 
    device, 
    dpt_head=None, 
    rank=None, 
    step=None): # img_tensor [B*seq, V, 3, H, W]
    
    vggt_model = vggt_model.to(device)
    _, V, C, H, W = img_tensor.shape
    seq = img_tensor.shape[0] // batch_combined
    img_tensor = img_tensor.view(batch_combined, seq, V, C, H, W).permute(1, 0, 2, 3, 4, 5) # [seq, B, V, 3, H, W]
    pred_sequence_features = []
    all_removed_tokens = []
    with torch.no_grad():
        with torch.autocast(dtype=dtype, device_type='cuda'):
            img_tensor = img_tensor.to(device)
            aggregated_tokens_list, patch_start_idx = vggt_model(img_tensor[0]) # [24] [B, V, N, D]
            intermediate_layer_idx = [4, 11, 17, 23]

            intermediate_tokens = [aggregated_tokens_list[intermediate_layer_idx[i]][:,:,patch_start_idx:] for i in range(len(intermediate_layer_idx))]
            intermediate_features = intermediate_tokens # [4] [B, V, N, D]
            current_features = intermediate_features
            for frame_idx in range(len(img_tensor)):
                frame_images = img_tensor[frame_idx]
                pred_features = transfer_model(frame_images, current_features)
                pred_features = [layer_feat[..., patch_start_idx:, :] for layer_feat in pred_features] # [4] [B, V, N, D]
                pred_sequence_features.append(torch.cat(pred_features, dim=1)) # append [B, 4*V, N, D]
                current_features = pred_features
            pred_sequence_features = torch.stack(pred_sequence_features, dim=0) # [seq, B, 4*V, N, D]
            pred_sequence_features = pred_sequence_features.permute(1, 0, 2, 3, 4) # [B, seq, 4*V, N, D]
            pred_sequence_features = pred_sequence_features.contiguous().view(batch_combined*seq, 4*V, pred_sequence_features.shape[3], pred_sequence_features.shape[4]) # [B*seq, 4*V, N, D]

    return pred_sequence_features

def load_vggt_model(ckpt_path=None):
    logger.info(f'Loading VGGT model...')
    model = VGGT()
    if ckpt_path is None:
        logger.warning(
            "vggt_path is None, skipping VGGT weight load. "
            "This is expected at eval (weights are restored from the checkpoint), "
            "but WILL break training if starting fresh.")
        return model
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(_R3DP_ROOT, ckpt_path)
    model.load_state_dict(torch.load(ckpt_path))
    return model

def load_transfer_model(model_cls, ckpt_path=None):
    model = TransferVGGT(
        img_size=308,
        patch_size=14,
        embed_dim=1024,
        num_heads=16,
        num_aa_blocks=4,
        num_register_tokens=4
    )

    if ckpt_path is None:
        logger.warning(
            "tvggt_path is None, skipping TransferVGGT (TFPNet) weight load. "
            "This is expected at eval (weights are restored from the checkpoint), "
            "but WILL break training if starting fresh.")
        return model
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(_R3DP_ROOT, ckpt_path)

    logger.info(f'Loading Transfer model {model_cls}')
    logger.info(f'Loading Transfer model {ckpt_path.split("/")[-3:]}')
    print(f'Loading Transfer model {model_cls}')
    print(f'Loading Transfer model {ckpt_path.split("/")[-3:]}')
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

class MultiImageObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            rgb_model: Union[nn.Module, Dict[str,nn.Module]],
            resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            random_crop: bool=True,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False,
            # spatial_reducer: nn.Module=None 
            device: str = None,
            batch_combined: int = 2,
            tau: int = 8, # run slow system while t % tau == 0
            model_cls: str = "feat_only",
            vggt_path: str = None,
            tvggt_path: str = None
        ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        Assumes features input: B,D
        """
        super().__init__()
        
        # Set device based on current process rank for DDP
        if device is not None:
            self._device = torch.device(device)
        else:
            # Let the model use the default device from ModuleAttrMixin
            # This will be properly set by accelerator.prepare() later
            self._device = None

        rgb_keys = list()
        low_dim_keys = list()
        feature_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map['rgb'] = rgb_model

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)
                
                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16, 
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = this_model
                
                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h,w)
                    )
                    input_shape = (shape[0],h,w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_normalizer = torchvision.transforms.CenterCrop(
                            size=(h,w)
                        )
                # configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                key_transform_map[key] = this_transform
            elif type == 'low_dim':
                low_dim_keys.append(key)
            elif type == 'features':
                feature_keys.append(key)
            elif type == 'original_rgb':
                pass
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        low_dim_keys = sorted(low_dim_keys)
        feature_keys = sorted(feature_keys)

        self.vggt_adapter = VGGTAdapterGrouped().to(self.device)
        prope_cfg = {
            "patches_x": 22,  
            "patches_y": 12,  
            "image_width": 308,
            "image_height": 168,
        }
        self.prope_adapter = MultiViewFusionPRoPE(prope_cfg=prope_cfg).to(self.device)
        self.vggt_model = load_vggt_model(vggt_path).to(self.device)
        self.vggt_model.eval()
        for param in self.vggt_model.parameters():
            param.requires_grad_(False) 
        self.current_vggt_features = None

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.feature_keys = feature_keys
        self.key_shape_map = key_shape_map
        self.visualization_counter = 0
        self.batch_combined = batch_combined

        self.state = None
        
        self.model_cls = model_cls
        self.transfer_model = load_transfer_model(self.model_cls, tvggt_path).to(self.device)
        self.transfer_model.eval()
        for param in self.transfer_model.parameters():
            param.requires_grad_(False) 

        self.step = 0
        self.tau = tau
        self.current_vggt_features = None
        self.patch_start_idx = None

        self.task_name = None
        
    def set_tau(self, tau):
        self.tau = tau
        print(f'[INFO] tau is set {tau}')

    def forward(self, obs_dict, vggt_obs_dict, mode=None):
        batch_size = None
        features = list()
        # process rgb input
        vggt_head_img = vggt_obs_dict['head_cam'] # (B*T, C, H, W)
        vggt_front_img = vggt_obs_dict['front_cam'] # (B*T, C, H, W)
        vggt_imgs = [vggt_head_img, vggt_front_img]
        
        rgb_features = {}
        for key in self.rgb_keys:
            img = obs_dict[key]
            if batch_size is None:
                batch_size = img.shape[0]
            else:
                assert batch_size == img.shape[0]
            assert img.shape[1:] == torch.Size(self.key_shape_map[key])
            img = self.key_transform_map[key](img)
            rgb_features[key] = self.key_model_map[key](img)
        
        # move all tensors to the correct device
        vggt_imgs = torch.stack(vggt_imgs, dim=1).to(self.device)
        if mode == 'inference':
            if self.step % self.tau == 0:
                self.current_vggt_features, self.patch_start_idx = extract_gt_features(self.vggt_model, vggt_imgs, self.device)
            vggt_features = self.transfer_model(vggt_imgs, self.current_vggt_features)
            self.current_vggt_features = vggt_features

            vggt_features = [vggt_features[i][:,:,self.patch_start_idx:] for i in range(4)] # [4] [B, V, N, D]
            vggt_features = torch.cat(vggt_features, dim=1) # [B, 4*V, N, D]
        else:
            vggt_features = extract_features(self.vggt_model, self.transfer_model, vggt_imgs, self.batch_combined, self.device)   # (B*T, view*4, 777, 2048)


        vggt_features = self.vggt_adapter(  # (B*T, view, 4, 777, 2048)
            rgb_features['head_cam'],
            rgb_features['front_cam'],
            vggt_features
        )

        vggt_features, vggt_features_compressed = self.prope_adapter(vggt_features) # (B*T, view, 4*264, 512) # (B*T, view, 4*264, 512) (B*T, view, 512)
        B, N_view, F = vggt_features_compressed.shape
        vggt_features_compressed = vggt_features_compressed.reshape(B, N_view*F) # (B*T, view*512)
        features.append(vggt_features_compressed)
        
        vggt_features = vggt_features.permute(0, 2, 1, 3).contiguous()         # (B*T, 4*264, V, 512)
        vggt_features = vggt_features.reshape(vggt_features.size(0), vggt_features.size(1), -1)  # (B*T, 4*264, V*512)
        
        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)
        
        # concatenate all features
        result = torch.cat(features, dim=-1) # (B*T, 1038)
        self.step += 1
        return result
    
    @torch.no_grad()
    def output_shape(self):
        # Don't manually move models to device here
        # Let accelerator.prepare() handle device assignment
        # This prevents models from being moved to wrong devices in DDP
        
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 8
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        vggt_obs_dict = {}
        vggt_obs_dict['head_cam'] = example_obs_dict['vggt_head_cam']
        vggt_obs_dict['front_cam'] = example_obs_dict['vggt_front_cam']
        example_output = self.forward(example_obs_dict, vggt_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape
    
    @property
    def device(self):
        if self._device is not None:
            return self._device
        else:
            # Fall back to ModuleAttrMixin's device property
            return super().device
