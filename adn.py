from functools import partial
from typing import Optional, Tuple

from rust_enum import Case, enum
import torch
from torch import nn
F = nn.functional

from sys import path 

@enum
class Activation:
    ELU = Case(inplace=bool)
    LRELU = Case(negative_slope=float, inplace=bool) # Leaky ReLU
    PRELU = Case(num_parameters=int, init=float)
    RELU = Case(inplace=bool)
    SWISH = Case(inplace=bool) # SiLU
    MEMSWISH = Case(inplace=bool) # Memory-efficient SiLU

    @staticmethod
    def elu(inplace=True):
        return Activation.ELU(inplace=inplace)
    @staticmethod
    def lrelu(negative_slope=0.2, inplace=True):
        return Activation.LRELU(negative_slope=negative_slope, inplace=inplace)
    @staticmethod
    def prelu(num_parameters=1, init=0.25):
        return Activation.PRELU(num_parameters=num_parameters, init=init)
    @staticmethod
    def relu(inplace=True):
        return Activation.RELU(inplace=inplace)
    @staticmethod
    def swish(inplace=True):
        return Activation.SWISH(inplace=inplace)
    @staticmethod
    def memswish(inplace=True):
        return Activation.MEMSWISH(inplace=inplace)

    def get_layer(self)->nn.Module:
        a : nn.Module 
        match self:
            case Activation.ELU(inplace): 
                a = nn.ELU(inplace=inplace)
            case Activation.LRELU(negative_slope, inplace):
                a = nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
            case Activation.PRELU(num_parameters, init):
                a = nn.PReLU(num_parameters=num_parameters, init=init)
            case Activation.RELU(inplace):
                a = nn.ReLU(inplace=inplace)
            case Activation.SWISH(inplace):
                a = nn.SiLU(inplace=inplace)  # Swish activation
            case Activation.MEMSWISH(inplace):
                a = nn.Mish(inplace=inplace)
        return a

@enum
class Normalization:
    INSTANCE = Case(num_features=int, affine=bool)
    BATCH = Case(num_features=int)
    INSTANCE_NVFUSER = Case(num_features=int, affine=bool)
    LOCALRESPONSE = Case(size=int, alpha=float, beta=float, k=float) 
    LAYER = Case(normalized_shape=list, eps=float, elementwise_affine=bool)
    GROUP = Case(num_groups=int, num_channels=int, affine=bool)
    SYNCBATCH = Case(num_features=int, eps=float, momentum=float, affine=bool, track_running_stats=bool)

    @staticmethod
    def instance(num_features:int, affine=True):
        return Normalization.INSTANCE(num_features=num_features, affine=affine)
    @staticmethod
    def batch(num_features:int):
        return Normalization.BATCH(num_features=num_features)
    @staticmethod
    def instance_nvfuser(num_features:int, affine=True):
        return Normalization.INSTANCE_NVFUSER(num_features=num_features, affine=affine)
    @staticmethod
    def localresponse(size=5, alpha=0.0001, beta=0.75, k=2.0):
        return Normalization.LOCALRESPONSE(size=size, alpha=alpha, beta=beta, k=k)
    @staticmethod
    def layer(normalized_shape=[1], eps=1e-05, elementwise_affine=True):
        return Normalization.LAYER(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
    @staticmethod
    def group(num_groups=4, num_channels=1, affine=True):
        return Normalization.GROUP(num_groups=num_groups, num_channels=num_channels, affine=affine)
    @staticmethod
    def syncbatch(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        return Normalization.SYNCBATCH(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def get_layer(self)->nn.Module:
        n : nn.Module 
        match self:
            case Normalization.INSTANCE(num_features, affine):
                n =  nn.InstanceNorm1d(num_features=num_features, affine=affine)
            case Normalization.BATCH(num_features):
                n =  nn.BatchNorm1d(num_features=num_features)
            case Normalization.INSTANCE_NVFUSER(num_features, affine):
                n =  nn.InstanceNorm1d(num_features=num_features, affine=affine, track_running_stats=False)
            case Normalization.LOCALRESPONSE(size, alpha, beta, k):
                n =  nn.LocalResponseNorm(size=size, alpha=alpha, beta=beta, k=k)
            case Normalization.LAYER(normalized_shape, eps, elementwise_affine):
                n =  nn.LayerNorm(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
            case Normalization.GROUP(num_groups, num_channels, affine):
                n =  nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, affine=affine)
            case Normalization.SYNCBATCH(num_features, eps, momentum, affine, track_running_stats):
                n =  nn.SyncBatchNorm(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        return n

def get_dropout_layer(dim:int, p:Optional[float], inplace:bool=True)->nn.Dropout1d|nn.Dropout2d|nn.Dropout3d:
    allowed_dims = (1,2,3)
    if dim not in allowed_dims:
        raise ValueError(f"dim must be in {allowed_dims}.")
    dropout_layer = (nn.Dropout1d, nn.Dropout2d ,nn.Dropout3d)[dim-1]
    if p is not None:
        dropout_layer = partial(dropout_layer, p=p)
    return dropout_layer(inplace=inplace)

class ADN(nn.Sequential):
    def __init__(
            self,
            ordering='NDA',
            in_channels: int | None = None, # will be used as num_features or num_channels for normalization if not specified in norm args
            act: Activation = Activation.relu(),
            norm: Normalization | None = None,
            dropout_dim: int | None = None,
            dropout_prob: float | None = None,
            ):
        super().__init__()
        op_dict = {"A": act, "D": None, "N": None}
        if norm is not None:
            op_dict["N"] = norm.get_layer() # get_norm_layer(name=norm, spatial_dims=norm_dim or dropout_dim, channels=in_channels)

        if dropout_dim is not None:
            op_dict["D"] = get_dropout_layer(dropout_dim, dropout_prob)

        for item in ordering.upper():
            if item not in op_dict:
                raise ValueError(f"ordering must be a string of {op_dict}, got {item} in it.")
            if op_dict[item] is not None:
                self.add_module(item, op_dict[item])