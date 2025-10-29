from typing import Callable, List, Optional, Sequence, Union
from functools import partial

import torch
from torch import dropout_, nn
from torch import Tensor
from torch.nn import functional as F
import numpy as np
import einops

from adn import Activation, ADN, Normalization
from film import FiLMIntercept
from pooling import Pooling

def __get_conv_for_dim(spatial_dims:int, conv_opts:List)->nn.Module:
    if spatial_dims <= 0:
        raise ValueError(f'Spatial dimensions can only be positive but got {spatial_dims}')
    if len(conv_opts) < spatial_dims:
        raise NotImplementedError(f'Convolutions for {spatial_dims}-D is not implemented')
    return conv_opts[spatial_dims-1]

def get_conv_for_dim(spatial_dims:int)->nn.Module:
    conv_opts = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
    return __get_conv_for_dim(spatial_dims, conv_opts)

def get_f_conv_for_dim(spatial_dims:int)->nn.Module:
    conv_opts = [F.conv1d, F.conv2d, F.conv3d]
    return __get_conv_for_dim(spatial_dims, conv_opts)

def get_conv_for_dim_from(spatial_dims:int, from_module)->nn.Module:
    conv_opts : List   
    try:
        conv_opts = [from_module.Conv1d, from_module.Conv2d, from_module.Conv3d]
    except AttributeError:
        conv_opts = [from_module.conv1d, from_module.conv2d, from_module.conv3d]
    return __get_conv_for_dim(spatial_dims, conv_opts)

# stolen from monai.networks.layers.convutils
def same_padding(kernel_size: Sequence[int] | int, dilation: Sequence[int] | int = 1) -> tuple[int, ...] | int:
    """
    Return the padding value needed to ensure a convolution using the given kernel size produces an output of the same
    shape as the input for a stride of 1, otherwise ensure a shape of the input divided by the stride rounded down.

    Raises:
        NotImplementedError: When ``np.any((kernel_size - 1) * dilation % 2 == 1)``.
    """

    kernel_size_np = np.atleast_1d(kernel_size)
    dilation_np = np.atleast_1d(dilation)

    if np.any((kernel_size_np - 1) * dilation % 2 == 1):
        raise NotImplementedError(
            f"Same padding not available for kernel_size={kernel_size_np} and dilation={dilation_np}."
        )

    padding_np = (kernel_size_np - 1) / 2 * dilation_np
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]

def validate_interception_point(p:int, len:int):
    neg_len = -len
    if p >= len or p < neg_len:
        raise ValueError(f'Layer injection index must be within bounds, but {p} is given')
