#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import math
from copy import deepcopy

import paddle

__all__ = ["ModelEMA", "is_parallel"]


def is_parallel(model):
    """check if model is in parallel mode."""
    return isinstance(model, paddle.DataParallel)


class ModelEMA:
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        """
        Args:
            model (paddle.nn.Layer): model to apply EMA.
            decay (float): ema decay reate.
            updates (int): counter of EMA updates.
        """
        # Create EMA(FP32)
        self.ema = deepcopy(model._layers if is_parallel(model) else model)
        self.ema.eval()
        self.updates = updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.stop_gradient = False

    def update(self, model):
        state_dict = {}
        # Update EMA parameters
        with paddle.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype in [paddle.float16, paddle.float32, paddle.float64]:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()
                    state_dict[k] = v
                state_dict[k] = v
        self.ema.set_state_dict(state_dict)
