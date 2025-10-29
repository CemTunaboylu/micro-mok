from functools import partial
from typing import Iterable

from torch import nn

from adn import ADN, Activation, Normalization
from conv import get_conv_for_dim
from scale import DownScale
from stage import Stage
from stem import Stem

class ResidualBlock(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 stride=1,
                 spatial_dims=2,
                 adn=partial(ADN, ordering='NA'),
                 ):
        super().__init__()

        conv = get_conv_for_dim(spatial_dims)
        batch : Normalization = Normalization.batch(dim=spatial_dims, num_features=out_ch)
        act : Activation = Activation.relu() 

        self.out_ch = out_ch
        self.in_ch = in_ch
        self.spatial_dims = spatial_dims

        self.conv1 = conv(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bna = adn(in_channels=in_ch, act=act, norm=batch)
        self.conv2 = conv(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = batch.get_layer()
        self.relu = act.get_layer()

        # Adjust input if shape mismatch (channels or stride)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                conv(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.bna(self.conv1(x))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)
        
class BottleneckBlock(nn.Module):
    def __init__(self,
                 in_ch,
                 mid_ch,
                 out_ch,
                 stride=1,
                 spatial_dims=2,
                 adn=partial(ADN, ordering='NA'),
                 ):
        super().__init__()

        self.out_ch = out_ch
        self.in_ch = in_ch
        self.spatial_dims = spatial_dims
        
        conv = get_conv_for_dim(spatial_dims)
        act: Activation = Activation.relu()
        mid_norm: Normalization = Normalization.batch(dim=spatial_dims, num_features=mid_ch)
        out_norm: Normalization = Normalization.batch(dim=spatial_dims, num_features=out_ch)

        # keep them separate (not nn.Sequential) since
        # DLA nodes will need to access them individually
        self.conv1 = conv(in_ch, mid_ch, 1, bias=False)
        self.bna1 = adn(in_channels=mid_ch, norm=mid_norm, act=act)
        self.conv2 = conv(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bna2 = adn(in_channels=mid_ch, norm=mid_norm, act=act)
        self.conv3 = conv(mid_ch, out_ch, 1, bias=False)
        self.bn3 = out_norm.get_layer()
        self.relu = act.get_layer()

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                conv(in_ch, out_ch, 1, stride=stride, bias=False),
                out_norm.get_layer()
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.bna1(self.conv1(x))
        out = self.bna2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return self.relu(out)

class ResNet34(nn.Module):
    def __init__(self, 
                 num_blocks: Iterable[int] = (3, 4, 6, 3),
                 downscales: Iterable[DownScale] = (None, DownScale(), DownScale(), DownScale()),
                 in_channels: int = 64,
                 num_classes: int = 1000
                 ):
        super().__init__()
        self.stem = Stem(in_channels=in_channels, out_channels=64, spatial_dims=2)

        self.stages : list[Stage] = [None] * 4
        for ix, (block_size, downscale) in enumerate(zip(num_blocks, downscales)):
            s = Stage(
                    index=ix,
                    block=ResidualBlock(64, 64),
                    num=block_size,
                    downscale=downscale
                    )
            self.stages[ix] = s

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x