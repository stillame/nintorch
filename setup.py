#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name='nintorch',
    version='0.1',
    author='Ninnart Fuengfusin',
    author_email='ninnart.fuengfusin@yahoo.com',
    description='A wrapper of Pytorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ninfueng/nintorch',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
    ],
    install_requires=[
        'torch == 1.5',
        'torchvision == 0.6.0',
        'apex == 0.1',
        'ninstd == 0.4',
        'pandas == 1.0.3',
        'matplotlib == 3.1.3',
        'loguru == 0.5.0',
        'pytest == 5.4.3',
    ],
    python_requires='>=3.7',
)
