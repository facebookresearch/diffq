# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import unittest

import torch
from torch import nn


def _cached_setup():
    model_small = nn.Sequential(nn.Conv1d(4, 4, 8, bias=False),
                                nn.ReLU(), nn.Conv1d(4, 4, 1, bias=False))
    model_big = nn.LSTM(256, 256, 2)

    def setUp(self):
        self.model_small = deepcopy(model_small)
        self.model_big = deepcopy(model_big)
        self.model_small_big = nn.ModuleList(
            [deepcopy(self.model_small), deepcopy(self.model_big)])
        self.models = [self.model_small, self.model_big, self.model_small_big]

    return setUp


class QuantizeTest(unittest.TestCase):
    setUp = _cached_setup()

    def _test_save_restore_state(self, factory):
        for model in self.models:
            model = deepcopy(model)
            params = list(model.parameters())

            zeroed = deepcopy(model)
            other_params = list(zeroed.parameters())
            for p in other_params:
                p.data.zero_()

            quantizer = factory(model)
            quantizer.quantize()
            state = quantizer.get_quantized_state()

            for p, q in zip(params, other_params):
                self.assertNotAlmostEqual(torch.norm(p.data - q.data).item(), 0)

            quantizer = factory(zeroed)
            quantizer.restore_quantized_state(state)

            for p, q in zip(params, other_params):
                self.assertAlmostEqual(torch.norm(p.data - q.data).item(), 0)

    def _test_true_model_size(self, factory):
        for model in [self.model_big, self.model_small_big]:
            model = deepcopy(model)
            quantizer = factory(model)
            compressed = quantizer.compressed_model_size(2, 1)
            true = quantizer.true_model_size()
            packed = quantizer.packed_model_size()
            estimate = quantizer.model_size()
            self.assertLessEqual(compressed, 1.5 * true, msg=repr(factory))
            self.assertLessEqual(compressed, 1.5 * estimate, msg=repr(factory))
            self.assertLessEqual(estimate, 1.5 * true, msg=repr(factory))
            self.assertLessEqual(true, 1.5 * estimate, msg=repr(factory))
            self.assertLessEqual(true, 1.1 * packed, msg=repr(factory))
            self.assertLessEqual(packed, 1.1 * true, msg=repr(factory))

            for num_workers in [1, 2]:
                other = quantizer.compressed_model_size(2, num_workers=num_workers)
                self.assertLessEqual(compressed, 1.1 * other, msg=repr(factory))
                self.assertLessEqual(other, 1.1 * compressed, msg=repr(factory))

    def _test_exclude(self, factory):
        model = nn.Conv1d(3, 64, 4, 1)
        quantizer = factory(model, min_size=0, exclude=["bias"])
        self.assertEqual(len(quantizer._qparams), 1)

        model = nn.Conv1d(3, 64, 4, 1)
        quantizer = factory(model, min_size=0, exclude=["bias", "weight"])
        self.assertEqual(len(quantizer._qparams), 0)

        model = nn.Sequential(*[nn.Conv1d(32, 32, 4, 1) for _ in range(6)])
        quantizer = factory(model, min_size=0, exclude=["[024].*"])
        self.assertEqual(len(quantizer._qparams), 6)
