#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Inspired from https://github.com/kennethreitz/setup.py

from pathlib import Path

from setuptools import setup

NAME = 'diffq'
DESCRIPTION = ('Differentiable quantization.')
URL = 'anonymized.com'
EMAIL = 'anon@anon.com'
AUTHOR = 'Anonymous author'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = "0.0.0"

HERE = Path(__file__).parent

try:
    with open(HERE / "README.md", encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=['diffq'],
    install_requires=['torch'],
    extras_require={'dev': ['coverage', 'pdoc3']},
    include_package_data=True,
    license='',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
