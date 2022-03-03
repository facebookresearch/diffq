# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from functools import partial
import os
from tempfile import NamedTemporaryFile

import torch
from torch import nn

import diffq
from diffq.tests.base import QuantizeTest
from diffq.diffq import DiffQuantizer
from diffq import ts_export


class TestDiffQ(QuantizeTest):
    def test_save_restore_state(self):
        self._test_save_restore_state(partial(DiffQuantizer, min_size=0))
        self._test_save_restore_state(partial(DiffQuantizer, min_size=0.1))
        self._test_save_restore_state(partial(DiffQuantizer, min_size=0.1, float16=True))

    def test_float16(self):
        for model in self.models:
            quantizer = DiffQuantizer(model, min_size=100000, float16=True)
            params = list(model.parameters())
            refs = [p.data.clone() for p in params]
            quantizer.quantize()
            for p, ref in zip(params, refs):
                self.assertAlmostEqual(torch.norm(ref.half().float() - p.data), 0)

    def test_init(self):
        model = self.models[0]
        quantizer = DiffQuantizer(model, init_bits=5, min_size=0)
        bits = quantizer._get_bits(quantizer._qparams[0].logit)
        self.assertTrue((bits == 5).all())

    def test_does_something(self):
        for ref_model in self.models:
            for group_size in [0, 1, 4, 8]:
                model = deepcopy(ref_model)
                params = list(model.parameters())
                refs = [p.data.clone() for p in params]
                quantizer = DiffQuantizer(model, group_size=group_size, min_size=0)
                quantizer.quantize()
                for p, ref in zip(params, refs):
                    self.assertNotAlmostEqual(torch.norm(p.data - ref).item(), 0)

    def test_detach(self):
        model = self.models[0]
        quantizer = DiffQuantizer(model, min_size=0)

        with self.assertRaises(RuntimeError):
            _ = DiffQuantizer(model, min_size=0)

        quantizer.detach()
        _ = DiffQuantizer(model, min_size=0)

    def test_setup_optimizer(self):
        model = nn.Linear(36, 4)

        quantizer = DiffQuantizer(model, min_size=0)
        opt = torch.optim.Adam(model.parameters())
        with self.assertRaises(RuntimeError):
            quantizer.setup_optimizer(opt)
        quantizer.detach()

        opt = torch.optim.Adam(model.parameters())
        quantizer = DiffQuantizer(model, init_bits=9, min_size=0)
        quantizer.setup_optimizer(opt)

        x = torch.randn(4, 36)
        model(x).sum().backward()
        opt.step()
        bits = quantizer._get_bits(quantizer._qparams[0].logit)
        self.assertNotAlmostEqual(torch.norm(bits - 9).item(), 0)

    def test_quantize_on_eval(self):
        x = torch.randn(1, 16)
        model = nn.Sequential(nn.Linear(16, 1024), nn.Linear(1024, 16))

        params = list(model.parameters())
        refs = [p.data.clone() for p in params]
        quant = DiffQuantizer(model, min_size=0)
        quant.no_optimizer()

        for p, ref in zip(params, refs):
            self.assertAlmostEqual(torch.norm(p.data - ref).item(), 0)
        model(x)
        # At train time, nothing should happen on the original param.
        for p, ref in zip(params, refs):
            self.assertAlmostEqual(torch.norm(p.data - ref).item(), 0)
        model.eval()
        model(x)
        for p, ref in zip(params, refs):
            self.assertAlmostEqual(torch.norm(p.data - ref).item(), 0)
        quant.quantize()
        for p, ref in zip(params, refs):
            self.assertNotAlmostEqual(torch.norm(p.data - ref).item(), 0)
        model.train()
        model(x)
        # Back to train, going back to the same params
        for p, ref in zip(params, refs):
            self.assertAlmostEqual(torch.norm(p.data - ref).item(), 0)

    def test_forward(self):
        x = torch.randn(1, 16)
        model = nn.Sequential(nn.Linear(16, 1024), nn.Linear(1024, 16))

        y = model(x)
        y2 = model(x)
        self.assertAlmostEqual(torch.norm(y - y2).item(), 0)

        DiffQuantizer(model, min_size=0).no_optimizer()
        model.train()
        # In training mode, two evals will be different because of the injected noise.
        y = model(x)
        y2 = model(x)
        self.assertNotAlmostEqual(torch.norm(y - y2).item(), 0)

        model.eval()
        # But at eval, no noise is injected.
        y = model(x)
        y2 = model(x)
        self.assertAlmostEqual(torch.norm(y - y2).item(), 0)

    def test_lstm(self):
        x = torch.randn(4, 5, 16)
        lstm = nn.LSTM(16, 32, 2)
        ref_y = lstm(x)[0]
        DiffQuantizer(lstm, min_size=0).no_optimizer()
        y = lstm(x)[0]
        self.assertNotAlmostEqual(torch.norm(y - ref_y).item(), 0)

    def test_model_size(self):
        for model in self.models:
            quantizer = DiffQuantizer(model, min_size=0)
            model_size = quantizer.model_size()
            model_size.backward()

    def test_true_model_size(self):
        factories = [
            partial(DiffQuantizer, init_bits=9, float16=False, min_size=0, group_size=8),
            partial(DiffQuantizer, init_bits=9, float16=False, min_size=0, group_size=0),
            partial(DiffQuantizer, init_bits=9, float16=False, min_size=0),
            partial(DiffQuantizer, init_bits=5, float16=False, min_size=0),
        ]
        for factory in factories:
            self._test_true_model_size(factory)

    def test_exclude(self):
        self._test_exclude(DiffQuantizer)

    def test_bound(self):
        one = nn.Linear(16, 16)
        two = _TestBound()
        two_detect = _TestBound()

        quant_one = DiffQuantizer(one, min_size=0, detect_bound=False)
        quant_two = DiffQuantizer(two, min_size=0, detect_bound=False)
        quant_two_detect = DiffQuantizer(two_detect, min_size=0, detect_bound=True)

        self.assertEqual(quant_one.true_model_size(), quant_two_detect.true_model_size())
        self.assertEqual(quant_two.true_model_size(), 2 * quant_one.true_model_size())

        qps = quant_two._qparams
        self.assertIsNot(qps[0].logit, qps[2].logit)

        qps = quant_two_detect._qparams
        self.assertIs(qps[0].logit, qps[2].logit)

        x = torch.randn(1, 16)

        quant_two.no_optimizer()
        ya, yb = two(x)
        self.assertNotAlmostEqual(torch.norm(ya - yb).item(), 0)

        quant_two_detect.no_optimizer()
        ya, yb = two_detect(x)
        self.assertAlmostEqual(torch.norm(ya - yb).item(), 0)
        (ya + yb).sum().backward()

    def test_restore(self):
        model = nn.Linear(53, 8, bias=False)
        quant = DiffQuantizer(model, min_size=0, group_size=8)
        for n, p in model.named_parameters():
            if n.endswith(quant.suffix):
                p.data.uniform_(-2, 2)
        state = quant.get_quantized_state()
        state_torch_pack = quant.get_quantized_state(torch_pack=True)

        model2 = nn.Linear(53, 8, bias=False)
        diffq.restore_quantized_state(model2, state)

        with quant.enter_quantize():
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                self.assertAlmostEqual(torch.norm(p1 - p2).item(), 0)

        model2 = nn.Linear(53, 8, bias=False)
        diffq.restore_quantized_state(model2, state_torch_pack)

        with quant.enter_quantize():
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                self.assertAlmostEqual(torch.norm(p1 - p2).item(), 0)

    def test_repr(self):
        model = nn.Linear(53, 8, bias=False)
        quant = DiffQuantizer(model, min_size=0, group_size=8)
        r = repr(quant)
        self.assertTrue("min_size=0" in r)
        self.assertTrue("group_size=8" in r)

    def test_torchscript_export(self):
        from torchvision import models
        model = models.resnet18(pretrained=False)
        quant = DiffQuantizer(model)
        for qp in quant._qparams:
            if qp.other is None:
                qp.logit.data.uniform_(-1, 1)
        ms = quant.true_model_size()
        model.eval()
        x = torch.randn(1, 3, 256, 256)
        y = model(x)
        with NamedTemporaryFile('wb') as file:
            ts = ts_export.export(quant, file.name)
            size = os.stat(file.name).st_size / 2**20
            self.assertLessEqual(size, 1.2 * ms, (size, ms))
            ts.eval()
            y2 = ts(x)
            self.assertAlmostEqual(torch.norm(y - y2).item(), 0)
            ts = torch.jit.load(file.name)
            ts.eval()
            y2 = ts(x)
            self.assertAlmostEqual(torch.norm(y - y2).item(), 0)


class _TestBound(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(16, 16)
        self.b = nn.Linear(16, 16)

        self.b.weight = self.a.weight
        self.b.bias = self.a.bias

    def forward(self, x):
        return self.a(x), self.b(x)
