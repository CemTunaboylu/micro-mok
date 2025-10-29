from enum import Enum, IntEnum
from conv import get_conv_for_dim
from defaulting import FirstEnumIsDefault
from functools import partial

from torch import nn
# For simplicity, regardless of method used, 
# downscaling is always assumed to halve the spatial dimensions
# AND double the number of channels. E.g: 
# Input: (B, C, H, W) -> Output: (B, 2C, H/2, W/2) 
# defaults to ConvStride (3x3 conv with stride 2) 
class DownScale(IntEnum, metaclass=FirstEnumIsDefault):
    # A.k.a. linear weighting with non-linearity OR projection with non-linearity
    # OR feature map pooling layer, which is 1x1 conv. with stride 2 learned pooling
    CrossChannelParametric = 0
    MaxPool = 1
    AvgPool = 2

    def get_layer_f(self):
        if self == DownScale.CrossChannelParametric:
            def ccp(in_channels:int, out_channels:int, spatial_dims:int):
                conv = get_conv_for_dim(spatial_dims)
                return partial(conv, in_channels=in_channels, out_channels=out_channels, stride=2)
            return ccp
        elif self == DownScale.MaxPool:
            def mp(in_channels:int, out_channels:int, spatial_dims:int):
                mps = [nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]
                return partial(mps[spatial_dims - 1], kernel_size=2, stride=2)
            return mp
        elif self == DownScale.AvgPool:
            def ap(in_channels:int, out_channels:int, spatial_dims:int):
                aps = [nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d]
                return partial(aps[spatial_dims - 1], kernel_size=2, stride=2)
            return ap
        else:
            raise ValueError(f"Unknown downscale method: {self}")
