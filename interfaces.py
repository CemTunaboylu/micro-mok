import abc 
from typing import Callable

from torch import nn

class IntoLayer(abc.ABC):
    @abc.abstractmethod
    def get_layer(self)->nn.Module:
        raise NotImplementedError()

# f(in_channels: int, out_channels: int, spatial_dims: int) -> nn.Module
type IntoLayerC = Callable[[int, int, int], nn.Module]

class IntoLayerF(abc.ABC):
    @abc.abstractmethod
    def get_layer_f(self)->IntoLayerC:
        raise NotImplementedError()
