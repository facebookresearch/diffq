# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch

from diffq.tests.base import QuantizeTest
from diffq.lsq import LSQ


class TestLSQ(QuantizeTest):
    def test_save_restore_state(self):
        self._test_save_restore_state(partial(LSQ, min_size=0))
        self._test_save_restore_state(partial(LSQ, min_size=0.1))
        self._test_save_restore_state(partial(LSQ, min_size=0.1, float16=True))

    def test_float16(self):
        for model in self.models:
            quantizer = LSQ(model, min_size=100000, float16=True)
            params = list(model.parameters())
            refs = [p.data.clone() for p in params]
            quantizer.quantize()
            for p, ref in zip(params, refs):
                self.assertAlmostEqual(torch.norm(ref.half().float() - p.data), 0)

    def test_true_model_size(self):
        factories = [
            partial(LSQ, bits=15, float16=False, min_size=0),
            partial(LSQ, bits=8, float16=False, min_size=0),
            partial(LSQ, bits=5, float16=False, min_size=0),
            partial(LSQ, bits=8, float16=False, min_size=0),
            partial(LSQ, bits=8, float16=False, min_size=0.5),
            partial(LSQ, bits=8, float16=True, min_size=0.5),
        ]
        for factory in factories:
            self._test_true_model_size(factory)

    def test_exclude(self):
        self._test_exclude(LSQ)

    def test_forward(self):
        model = torch.nn.Linear(16, 16, bias=False)
        x = torch.randn(4, 16)
        y = model(x)

        quant = LSQ(model, min_size=0)
        with self.assertRaises(RuntimeError):
            y2 = model(x)

        quant.no_optimizer()
        y2 = model(x)
        model.eval()
        y3 = model(x)

        self.assertNotAlmostEqual(torch.norm(y - y2).item(), 0)
        self.assertAlmostEqual(torch.norm(y2 - y3).item(), 0, places=6)
