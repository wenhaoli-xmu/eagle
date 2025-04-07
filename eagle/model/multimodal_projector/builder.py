import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from typing import Optional, List
from torch.autograd import Function

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)
        
# def forward_sample_linear(self, x: torch.Tensor, sample_idx: Optional[List[int]] = None):
#     sample_weight_idx = [i for idx in sample_idx for i in range(idx*1024, (idx + 1)*1024)]
#     sampled_weight = self.weight[:, sample_weight_idx] # [output_channel, input_channel]
#     sampled_bias = self.bias
#     return F.linear(x, sampled_weight, sampled_bias)

# def forward_sample_linear(self, x: torch.Tensor, sample_idx: Optional[List[int]] = None):
#     if sample_idx is None:
#         return F.linear(x, self.weight, self.bias)
    
#     output_dim = self.weight.shape[0]
    
#     # Create a mask for the selected weights
#     mask = torch.zeros_like(self.weight, device=self.weight.device)
    
#     for idx in sample_idx:
#         mask[:, idx*1024:(idx+1)*1024] = 1.0
    
#     # Apply the mask to the weights
#     masked_weight = self.weight * mask
#     # Perform the linear operation with the masked weights
#     return F.linear(x, masked_weight, self.bias)

class SampleWeightFunction(Function):
    @staticmethod
    def forward(ctx, weight, sample_idx):
        mask = torch.zeros_like(weight)
        if sample_idx is not None:
            for idx in sample_idx:
                mask[:, idx*1024:(idx+1)*1024] = 1.0
        
        ctx.mask = mask
        return weight * mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.mask, None

class SampleLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.sample_idx = None

    def forward(self, x: torch.Tensor, sample_idx: Optional[List[int]] = None):
        self.sample_idx = sample_idx
        sampled_weight = SampleWeightFunction.apply(self.weight, self.sample_idx)
        return F.linear(x, sampled_weight, self.bias)


def build_vision_projector(config, delay_load=False, fpn_input_dim=[], **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        
        # if config.mm_vision_sample_feature:
        #     # bound_method = forward_sample_linear.__get__(modules[0], modules[0].__class__)
        #     # setattr(modules[0], 'forward', bound_method)
        #     modules = [SampleLinear(config.mm_hidden_size, config.hidden_size)]
        #     print("build samplelinear for projector")
        # else:
        #     modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]

        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
