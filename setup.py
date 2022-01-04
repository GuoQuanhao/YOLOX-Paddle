#!/usr/bin/env python
# Copyright (c) Megvii, Inc. and its affiliates. All Rights Reserved

import re
import setuptools
import glob
from os import path
import paddle
from paddle.utils.cpp_extension import CppExtension, setup
# from paddle.utils.cpp_extension import CppExtension
# from setuptools import Extension, setup


paddle_ver = [int(x) for x in paddle.__version__.split(".")[:2]]
assert paddle_ver >= [2, 0], "Requires Paddle >= 2.0"

def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "yolox", "layers", "csrc")

    main_source = path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))

    sources = [main_source] + sources
    return sources


with open("yolox/__init__.py", "r") as f:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        f.read(), re.MULTILINE
    ).group(1)


setup(
    name="yolox",
    version=version,
    author="GuoQuanhao",
    python_requires=">=3.7",
    ext_modules=CppExtension(sources=get_extensions()),
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    packages=setuptools.find_packages(),
)

