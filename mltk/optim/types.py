from typing import Union, List, Tuple

import torch as th
from torch.optim.optimizer import Optimizer

# Parameters input type
ParamsInput = Union[
    th.Tensor,
    List[th.Tensor],
    Tuple[th.Tensor],
    Optimizer
]
