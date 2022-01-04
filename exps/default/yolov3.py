#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import paddle.nn as nn

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_model(self, sublinear=False):
        def init_yolo(M):
            for m in M.sublayer():
                if isinstance(m, nn.BatchNorm2D):
                    m._epsilon = 1e-3
                    m._momentum = 0.97
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOFPN, YOLOXHead
            backbone = YOLOFPN()
            head = YOLOXHead(self.num_classes, self.width, in_channels=[128, 256, 512], act="lrelu")
            self.model = YOLOX(backbone, head)
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

        return self.model
