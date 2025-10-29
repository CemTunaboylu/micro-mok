from typing import List, Union
from rust_enum import Case, enum

from torch import nn
from adn import ADN, Activation, Normalization, get_dropout_layer 

@enum
class Aggregation:
    # Iterative is binary, i.e. expects 2 operands 
    # (blocks, stages etc. of same channels, and spatial dimension)
    Iterative = Case()
    # Hierarchical is n-ary, i.e. expects n operands where n = 2^depth
    # and aggregated blocks/stages may not have same channels and spatial dimensions
    Hierarchical = Case(depth=int)

# ! Will be switched with injector i.e. takes a Block or Stage and injects
# connections in btw. arbitrary layers for aggregation
class Block(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 spatial_dims: int,
                 kernel_sizes: int | list[int] = 3,
                 batch_norms: bool | list[bool] = True,
                 activations: bool | list[bool] = True,
                 dropouts: bool | list[bool] = False,
                 dropout_probs: float | list[float] = 0.1,
                 has_residual: bool = True,
                 ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dims = spatial_dims
        self.kernel_sizes = [kernel_sizes] if isinstance(kernel_sizes, int) else kernel_sizes

        self.num_convs = len(self.kernel_sizes)

        layers = []
        for i in range(self.num_convs):
            conv_in_channels = in_channels if i == 0 else out_channels
            layers.append(
                nn.Conv2d(
                    in_channels=conv_in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)

# The left-most child has shape of (B,C,H,W) 
# others are already downscaled (B,2C,H/2,W/2)
# since they are within the same stage
class AggregationNode:
    def __init__(self,
                 aggregation: Aggregation,
                 children: List[Union[Block, 'AggregationNode']],
                 ):
        self.aggregation = aggregation
        self.children = children

