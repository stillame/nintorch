# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import os
import random
from loguru import logger
import torch

__all__ = [
    'seed_torch',
    'torch_cpu_or_gpu',
    'get_lr_from_optim',
    'AvgMeter',
    'init_logger']


def seed_torch(seed: int = 0, backend: bool = True):
    """Initialize a random seed in every possible places.
    backend might cause the model to reduce the accuracy but incresing the reproductivity. .
    From: https://github.com/pytorch/pytorch/issues/11278
    From: https://pytorch.org/docs/stable/notes/randomness.html
    From: https://github.com/NVIDIA/apex/tree/master/examples/imagenet
    """
    assert isinstance(seed, int)
    assert isinstance(backend, bool)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = backend
    torch.backends.cudnn.benchmark = not(backend)
    try:
        logger.info(f'Plant the random seed: {seed}.')
    except NameError:
        print(f'Plant the random seed: {seed}.')


def torch_cpu_or_gpu():
    """Return torch device either `cuda` or `cpu`.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def get_lr_from_optim(optim):
    """Note that the lr in this case is not changed with scheduler.
    Using scheduler.get_lr()[0] instead in this case.
    """
    return [param_group['lr'] for param_group in optim.param_groups][0]


class AvgMeter(object):
    """For using with loss or accuracy.
    """
    def __init__(self):
        self.sum = 0.0
        self.num = 0

    def update(self, val: int, batch_size: int):
        self.sum += val
        self.num += batch_size

    def clear(self):
        self.sum = 0.0
        self.num = 0

    @property
    def avg(self):
        return self.sum/self.num

    def __call__(self, val: int, batch_size: int):
        self.update(val, batch_size)


def init_logger(name_log: str = __file__, rm_exist: bool = False):
    """Setting logger with my basic setting.
    Args:
        name_log (str): a txt file name for logging go to.
        rm_exist (bool): remove the existing txt file or not.
    """
    if rm_exist and os.path.isfile(name_log):
        os.remove(name_log)
    logger.add(
        name_log, format='{time} | {level} | {message}',
        backtrace=True, diagnose=True, level='INFO', colorize=False)
