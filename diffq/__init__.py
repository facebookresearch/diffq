# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# flake8: noqa
"""
This package implements different quantization strategies:

- `diffq.uniform.UniformQuantizer`: classic uniform quantization over n bits.
- `diffq.diffq.DiffQuantizer`: differentiable quantizer based on scaled noise injection.
- `diffq.lsq.LSQ`: Learnt Step size Quantizer based on [Esser et al. 2019] https://arxiv.org/abs/1902.08153
- `diffq.bitpack`: efficient CPU bit-packing for returning quantized states.

Also, do check `diffq.base.BaseQuantizer` for the common methods of all Quantizers.
"""

from .uniform import UniformQuantizer
from .diffq import DiffQuantizer
from .lsq import LSQ
from .base import restore_quantized_state