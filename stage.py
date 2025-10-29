from torch import nn

from interfaces import IntoLayerF
from scale import DownScale

class Stage(nn.Sequential):
    def __init__(self,
                 index: int,
                 block: nn.Module,
                 num: int,
                 downscale: None | IntoLayerF = None,
                 ):
        self.index = index
        self.in_ch = block.in_ch
        self.out_ch = block.out_ch
        self.spatial_dims = block.spatial_dims

        blocks = [block] * num
        if downscale:
            f = downscale.get_layer_f()
            in_ch = blocks[-1].out_ch 
            out_ch = in_ch * 2
            spatial_dims = blocks[-1].spatial_dims
            downscale = f(in_ch, out_ch, spatial_dims)
            blocks.append(downscale)

        super().__init__(*blocks)