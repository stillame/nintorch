#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collection of model defination.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..model import BaseNet, Conv2dSame


__all__ = [
    'LeNet5', 
    'VGG16BN']

class LeNet5(BaseNet):
    def __init__(self, in_chl: int = 1, *args, **kwargs) -> None:
        super(LeNet5, self).__init__(*args, **kwargs)
        assert isinstance(in_chl, int)
        if in_chl == 1:
            # In case MNIST like.
            FEAT_SIZE: int = 4
        elif in_chl == 3:
            # In case CIFAR10 like.
            FEAT_SIZE: int = 5
        else:
            # In case wanting to use other datasets, 
            # please replace self.features, self.linear or both of them.
            raise NotImplementedError(
                f'Supported only in_chl 1 or 3, your input: {in_chl}')
        
        self.features = nn.Sequential(
            nn.Conv2d(in_chl, 6, 5),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True))
        
        self.classifier = nn.Sequential(
            nn.Linear(FEAT_SIZE*FEAT_SIZE*16, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10))

        
    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=-1)
        return x


class VGG16BN(BaseNet):
    def __init__(self, in_chl: int = 3, *args, **kwargs) -> None:
        assert isinstance(in_chl, int)
        super(VGG16BN, self).__init__(*args, **kwargs)
        self.features = nn.Sequential(
            Conv2dSame(in_chl, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Conv2dSame(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=False),

            Conv2dSame(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Conv2dSame(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=False),

            Conv2dSame(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            Conv2dSame(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            Conv2dSame(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=False),

            Conv2dSame(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            Conv2dSame(512, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            Conv2dSame(512, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=False),

            Conv2dSame(512, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            Conv2dSame(512, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            Conv2dSame(512, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=False)
            )
        self.classifier = nn.Linear(1*1*512, 10)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=-1)
        return x

