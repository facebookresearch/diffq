# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import sys


def find_diffq():
    diffq_path = Path(__file__).parent.parent.parent.parent
    sys.path.append(str(diffq_path))
