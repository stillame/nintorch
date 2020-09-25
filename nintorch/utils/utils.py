# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The collection of modules, which does not belong to any other sections.
"""
import os
import random
import requests
from io import BytesIO

import cv2
from PIL import Image
from loguru import logger

import torch
import torch.nn as nn
import numpy as np

__all__ = [
    'seed',
    'torch_cpu_or_gpu',
    'get_lr_from_optim',
    'AvgMeter',
    'init_logger',
    'get_device_from_module',
    ]

def seed(seed: int = 2020) -> None:
    r"""Initialize a random seed in every possible places.
    backend might cause the model to reduce the accuracy but incresing the reproductivity. .
    From: https://github.com/pytorch/pytorch/issues/11278
    From: https://pytorch.org/docs/stable/notes/randomness.html
    From: https://github.com/NVIDIA/apex/tree/master/examples/imagenet
    """
    assert isinstance(seed, int)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # TODO: checking speed of cudnn False
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    log_exists = 'logger' in locals() or 'logger' in globals()
    if log_exists:
        logger.info(f'Plant the random seed: {seed}.')
    else:
        print(f'Plant the random seed: {seed}.')


def torch_cpu_or_gpu() -> None:
    r"""Return torch device either `cuda` or `cpu`.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def get_lr_from_optim(optim):
    r"""Note that the lr in this case is not changed with scheduler.
    Using scheduler.get_lr()[0] instead in this case.
    """
    return [param_group['lr'] for param_group in optim.param_groups][0]


class AvgMeter(object):
    r"""For using with loss or accuracy.
    """
    def __init__(self) -> None:
        self.sum = 0.0
        self.num = 0
        self.idx = 0

    def update(self, val: int, batch_size: int) -> None:
        self.sum += val
        self.num += batch_size
        self.idx += 1

    def clear(self) -> None:
        self.sum = 0.0
        self.num = 0
        self.idx = 0

    @property
    def avg(self) -> float:
        return self.sum/self.num

    def __call__(self, val: int, batch_size: int):
        self.update(val, batch_size)

    def __repr__(self) -> str:
        return (f'Accumuation: {self.sum},' 
                f' with number of elements {self.num},'
                f' for {self.idx} times.')


def init_logger(name_log: str = __file__, rm_exist: bool = False):
    r"""Setting logger with my basic setting.
    Args:
        name_log (str): a txt file name for logging go to.
        rm_exist (bool): remove the existing txt file or not.
    """
    if rm_exist and os.path.isfile(name_log):
        os.remove(name_log)
    logger.add(
        name_log, format='{time} | {level} | {message}',
        backtrace=True, diagnose=True, level='INFO', colorize=False)


def get_device_from_module(model: nn.Module) -> torch.device:
    r"""Return torch.device from nn.Module.
    """
    return next(model.parameters()).device


def get_img_from_url(url: str) -> np.array: 
    """Download image from url. If .png 4 channels convert to rgb automatically.
    Ex: get_img_from_url(url=" https://i.imgur.com/Bvro0YD.png")
    """
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = np.array(img)
    if img.shape[-1] == 4:
        # If download .png file with extra-channel, alpha that convert to 3 channels.
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


def is_inrange(x: np.ndarray, low, high) -> bool:
    """Support only in [].
    TODO: () (] [) []? and for torch? 
    Ex:
        is_inrange(np.random.random(), 0, 1) -> True
    """
    return np.logical_and(x <= high, x >=  low).all()