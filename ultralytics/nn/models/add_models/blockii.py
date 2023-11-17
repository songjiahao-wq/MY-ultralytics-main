"""
Core of BiFormer, Bi-Level Routing Attention.
To be refactored.
author: ZHU Lei
github: https://github.com/rayleizhu
email: ray.leizhu@outlook.com
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


