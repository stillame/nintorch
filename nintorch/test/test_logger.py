#
#
"""
"""
import os
from loguru import logger
import pytest
from nintorch import init_logger


def test_init_logger():
    init_logger('test.log')
    logger.info('test')
    assert os.path.isfile('test.log')
    if os.path.isfile('test.log'):
        os.remove('test.log')

