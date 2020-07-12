#
#
"""
"""
import torch
import torch.nn as nn
from nintorch.model import BaseNet, Conv2dSame 


class NetTest(BaseNet):
    def __init__(self):
        super(NetTest, self).__init__()
        self.l0 = nn.Conv2d(1, 6, 3)
        self.l1 = nn.BatchNorm2d(6)
        self.l2 = nn.Dropout2d(0.3)
        self.l3 = nn.Linear(100, 10)
        self.l4 = nn.MaxPool2d(2)


def test_flatten():
   x = torch.zeros(1, 2, 3, 4)
   flat_tensor = BaseNet().flatten(x)
   assert (flat_tensor == torch.zeros(1, 24)).all()


def test_conv2dsame():
    mock_input = torch.zeros(128, 1, 28, 28)
    conv_shape = Conv2dSame(1, 6, 5)(mock_input).shape
    assert conv_shape == (128, 6, 28, 28)


