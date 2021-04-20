# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch
from torch import nn

from diffq.tests.base import QuantizeTest
from diffq.uniform import UniformQuantizer


class TestUniform(QuantizeTest):
    def test_save_restore_state(self):
        self._test_save_restore_state(partial(UniformQuantizer, min_size=0))
        self._test_save_restore_state(partial(UniformQuantizer, min_size=0.1))
        self._test_save_restore_state(partial(UniformQuantizer, min_size=0.1, float16=True))

    def test_float16(self):
        for model in self.models:
            quantizer = UniformQuantizer(model, min_size=100000, float16=True)
            params = list(model.parameters())
            refs = [p.data.clone() for p in params]
            quantizer.quantize()
            for p, ref in zip(params, refs):
                self.assertAlmostEqual(torch.norm(ref.half().float() - p.data), 0)

    def test_delta(self):
        for model in self.models:
            for bits in [1, 4, 8, 9.6, 15]:
                params = list(model.parameters())
                refs = [p.data.clone() for p in params]
                quantizer = UniformQuantizer(model, bits=bits, min_size=0)
                quantizer.quantize()
                for p, ref in zip(params, refs):
                    scale = ref.max() - ref.min()
                    delta = scale / (int(2 ** bits) - 1)
                    self.assertLessEqual((ref - p).abs().max(), delta)

    def test_qat(self):
        x = torch.randn(1, 16)

        model = nn.Sequential(nn.Linear(16, 1024), nn.Linear(1024, 16))
        params = list(model.parameters())
        refs = [p.data.clone() for p in params]
        ref_y = model(x)

        _ = UniformQuantizer(model, min_size=0, qat=True)
        for p, ref in zip(params, refs):
            self.assertAlmostEqual(torch.norm(p.data - ref).item(), 0)
        y = model(x)
        self.assertNotAlmostEqual(torch.norm(y - ref_y).item(), 0)
        for p, ref in zip(params, refs):
            self.assertAlmostEqual(torch.norm(p.data - ref).item(), 0)

    def test_lstm(self):
        x = torch.randn(4, 5, 16)
        lstm = nn.LSTM(16, 32, 2)
        ref_y = lstm(x)[0]
        _ = UniformQuantizer(lstm, min_size=0, qat=True)
        y = lstm(x)[0]
        self.assertNotAlmostEqual(torch.norm(y - ref_y).item(), 0)

    def test_true_model_size(self):
        factories = [
            partial(UniformQuantizer, bits=15, float16=False, min_size=0),
            partial(UniformQuantizer, bits=8, float16=False, min_size=0),
            partial(UniformQuantizer, bits=5, float16=False, min_size=0),
            partial(UniformQuantizer, bits=8, float16=False, min_size=0),
            partial(UniformQuantizer, bits=8, float16=False, min_size=0.5),
            partial(UniformQuantizer, bits=8, float16=True, min_size=0.5),
        ]
        for factory in factories:
            self._test_true_model_size(factory)

    def test_resume_training(self):
        m = nn.Linear(16, 32)
        x = torch.randn(4, 16)
        _ = UniformQuantizer(m, min_size=0)

        y_train = m(x)
        m.eval()
        y_test = m(x)
        m.train()
        y_train_post = m(x)

        self.assertNotAlmostEqual(torch.norm(y_train - y_test).item(), 0)
        self.assertAlmostEqual(torch.norm(y_train - y_train_post).item(), 0)

    def test_exclude(self):
        self._test_exclude(UniformQuantizer)

    def test_bound(self):
        one = nn.Linear(16, 16)
        two = nn.ModuleList([nn.Linear(16, 16), nn.Linear(16, 16)])
        two[1].weight = two[0].weight
        two[1].bias = two[0].bias

        quant_one = UniformQuantizer(one, min_size=0, detect_bound=False)
        quant_two = UniformQuantizer(two, min_size=0, detect_bound=False)
        quant_two_detect = UniformQuantizer(two, min_size=0, detect_bound=True)

        self.assertEqual(quant_one.true_model_size(), quant_two_detect.true_model_size())
        self.assertEqual(quant_two.true_model_size(), 2 * quant_one.true_model_size())
