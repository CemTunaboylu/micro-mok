from functools import partial
from parameterized import parameterized
import unittest

import torch

from dla.dla import DownScale, Aggregation

class TestDownScale(unittest.TestCase):
    def test_default_down_scale(self):
        assert DownScale() is DownScale.CrossChannelParametric

class AggregationEnumTest(unittest.TestCase):
    def test_iterative_aggregation(self):
        pass
    @parameterized.expand([
       ("single_depth", -1.5, -2.0),
       ("double_depth", 1, 1.0),
       ("triple_depth", 1.6, 1),
   ])
    def test_hierarchical_aggregation(self, name, input_channels, input_spatial_dims, expected):
        pass
        # for cls in model_classes: 
        #     model = cls(
        #             spatial_dims = 3, 
        #             in_channels = channel_size, 
        #             out_channels = channel_size, 
        #             features = features
        #             )

        #     model.train()
        #     i= torch.ones([batch,channel_size,d,h,w])
        #     o : torch.Tensor
        #     if cls == UNet3D:
        #         o = model.forward(i)
        #     else:
        #         meta = torch.ones([batch,meta_size])
        #         o = model.forward(i,meta)
        #     self.assertEqual(o.shape, (batch, channel_size, d, h, w))
        

if __name__ == "__main__":
    unittest.main()