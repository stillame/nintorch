#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collection of function for model.
"""
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = [
    'Conv2dSame',
    'BaseNet',]


class Conv2dSame(nn.Conv2d):
    """Convolutional layer with same padding.
    """
    def __init__(self, *args, **kwargs):
        super(Conv2dSame, self).__init__(*args, **kwargs)

    def _conv_forward(self, input, weight):
        kernel_w = weight.shape[-2]
        kernel_h = weight.shape[-1]
        self.padding = (math.floor(kernel_w/2), math.floor(kernel_h/2))
        return F.conv2d(
            input, weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self._conv_forward(input, self.weight)


class BaseNet(nn.Module):
    """BaseNet: basic model class.
    TODO: load_pretrain weight, save model, load model.
    """
    def __init__(self, *args, **kwargs):
        super(BaseNet, self).__init__(*args, **kwargs)

    def save_all(self, path: str):
        torch.save(self, path)

    @staticmethod
    def load_all(path: str):
        model = torch.load(path)
        return BaseNet(model)

    def save_state_dict(self):
        return

    @staticmethod
    def flatten(f_map: torch.tensor):
        """For converting the output activation from convolutional layer
        to the dense layer.
        """
        assert hasattr(f_map, 'shape')
        num_f = reduce(lambda x, y: x*y, f_map.shape[1:])
        return f_map.view(-1, num_f)

    def init_weight_bias(self, init_funct=None) -> None:
        """Provided init_funct to apply to model.
        Modified from: https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py
        TODO: need more support layers.
        """
        if init_funct is None:
            def init_funct(module):
                """Default weight and bias initization.
                """
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out')
                    nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=1e-3)
                    nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
        self.apply(init_funct)

    def track_stat(self, writer, global_step: int):
        """Given the tensorboard writer to tracking the statistical analysis.
        Return std and mean of each layers.
        Return mean and std from each layer that contains parameters.
        """
        for name_l, layer in self.named_children():
            params = list(layer.parameters())
            if not len(params) == 0:
                w, b = params[0], params[1]
                writer.add_scalar(f'{name_l}_w_std', w.std(), global_step)
                writer.add_scalar(f'{name_l}_b_std', b.std(), global_step)
                writer.add_scalar(f'{name_l}_w_mean', w.mean(), global_step)
                writer.add_scalar(f'{name_l}_b_mean', b.mean(), global_step)

    def track_dist_weight_bias(self, writer, global_step: int):
        """Using of tensorboard to tracking distributions for each
        weight and bias in each layer of model.
        """
        for name_l, layer in self.named_children():
            params = list(layer.parameters())
            if not len(params) == 0:
                writer.add_histogram(f'{name_l}_w', params[0], global_step)
                writer.add_histogram(f'{name_l}_b', params[1], global_step)
