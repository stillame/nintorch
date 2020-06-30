#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import setuptools

def get_version():
    return

with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name='ninstd',
    version='0.4',
    author='Ninnart Fuengfusin',
    author_email='ninnart.fuengfusin@yahoo.com',
    description='A wrapper of Python using only standard libraries.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ninfueng/ninstd',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.7',
)

