from functools import partial
from typing import Iterable

from torch import nn

from adn import ADN, Activation, Normalization
from conv import get_conv_for_dim
from scale import DownScale
from stage import Stage
from stem import Stem

class ResNeXtBlock(nn.Module):
    def __init__(self,
                 in_ch,
                 mid_ch,
                 out_ch,
                 stride=1,
                 cardinality=32,
                 spatial_dims=2,
                 adn=partial(ADN, ordering='NA'),
                 ):
        super().__init__()

        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch
        self.spatial_dims = spatial_dims

        conv = get_conv_for_dim(spatial_dims)
        batch : Normalization = Normalization.batch(dim=spatial_dims, num_features=mid_ch)
        act: Activation = Activation.relu()

        self.conv1 = conv(in_ch, mid_ch, kernel_size=1, bias=False)
        self.bna1 = adn(in_channels=mid_ch, act=act, norm=batch)

        self.conv2 = conv(
            mid_ch, mid_ch, kernel_size=3, stride=stride, padding=1,
            groups=cardinality, bias=False
        )
        self.bna2 = adn(in_channels=mid_ch, act=act, norm=batch)

        self.conv3 = conv(mid_ch, out_ch, kernel_size=1, bias=False)
        self.bn3 = batch.get_layer()

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                conv(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = act.get_layer()

    def forward(self, x):
        out = self.bna1(self.conv1(x))
        out = self.bna2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return self.relu(out)
        
class ResNeXt34(nn.Module):
    def __init__(self,
                 in_channels=3,
                 num_blocks: Iterable[int] =(3, 4, 6, 3),
                 downscales: Iterable[DownScale] =tuple([DownScale()]*4),
                 cardinality=32,
                 spatial_dims=2,
                 ):
        super().__init__()
        self.stem = Stem(in_channels=in_channels, out_channels=64, spatial_dims=spatial_dims)

        self.stages : list[Stage] = [None] * 4
        for ix, (block_size, downscale) in enumerate(zip(num_blocks, downscales)):
            mid_ch = 128 * (2 ** ix)
            out_ch = mid_ch * 4
            block = partial(ResNeXtBlock,
                            mid_ch=mid_ch,
                            out_ch=out_ch,
                            cardinality=cardinality,
                            spatial_dims=spatial_dims)
            in_ch = 64 if ix == 0 else out_ch // 2
            s = Stage(
                      index=ix,
                      block=block(in_ch=in_ch),
                      num=block_size,
                      downscale=downscale)
            self.stages[ix] = s

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x