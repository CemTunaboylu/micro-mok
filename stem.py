from functools import partial

from torch import nn

from adn import ADN, Activation, Normalization
from conv import get_conv_for_dim
from pooling import Pooling

class Stem(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = 64,
                 kernel_size: int = 7,
                 stride : int = 2,
                 spatial_dims : int = 2,
                 act = Activation.relu(),
                 norm = Normalization.batch,
                ):
        conv = get_conv_for_dim(spatial_dims)
        norm = norm(dim=spatial_dims, num_features=out_channels)

        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.bna = ADN(in_channels=out_channels, act=act, norm=norm)
        self.pool = Pooling.max_f(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.bna(self.conv(x))
        out = self.pool(out)
        return out