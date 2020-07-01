#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..model import BaseNet


class LeNet5(BaseNet):
    def __init__(self, in_chl: int = 3):
        super(LeNet5, self).__init__()
        self.l0 = nn.Conv2d(in_chl, 6, 5)
        self.l1 = nn.MaxPool2d(2)

        self.l2 = nn.Conv2d(6, 16, 5)
        self.l3 = nn.MaxPool2d(2)

        self.l4 = nn.Linear(4*4*16, 120)
        self.l5 = nn.Linear(120, 84)
        self.l6 = nn.Linear(84, 10)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.l0(x)
        x = F.relu(x)
        x = self.l1(x)

        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)

        x = self.flatten(x)
        x = self.l4(x)
        x = F.relu(x)

        x = self.l5(x)
        x = F.relu(x)

        x = self.l6(x)
        x = F.log_softmax(x)
        return x


class VGG16BN(BaseNet):
    def __init__(self, in_chl: int = 3, *args, **kwargs):
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
    
