from typing import Optional
from torch.types import _size

import torch as th
import torch.distributions as thd

__all__ = [
    "MultivariateNormalDiag",
    "OneHotCategorical"
]

class MultivariateNormalDiag(thd.LowRankMultivariateNormal):
    """
    Create a multivariate normal distribution with diagonal covariance matrix.
    (i.e. zero correlation between any dimensions)
    """
    def __init__(self, loc: th.Tensor, scale_diag: th.Tensor,
        validate_args: Optional[bool] = None):
        # "Dummy" covariance factor
        cov_factor = th.zeros_like(scale_diag).unsqueeze(-1)
        cov_diag = scale_diag.square()

        super().__init__(loc, cov_factor, cov_diag, validate_args=validate_args)

    def detach(self):
        new = MultivariateNormalDiag.__new__(MultivariateNormalDiag)
        super(MultivariateNormalDiag, new).__init__(
            self.loc.detach(),
            self.cov_factor,
            self.cov_diag.detach(),
            validate_args=self._validate_args
        )
        return new

    @property
    def scale_diag(self) -> th.Tensor:
        return self.cov_diag.sqrt()

class OneHotCategorical(thd.OneHotCategorical):
    has_rsample = True

    @th.no_grad()
    def sample(self, sample_shape: _size = th.Size()) -> th.Tensor:
        return super().sample(sample_shape)

    def rsample(self, sample_shape: _size = th.Size()) -> th.Tensor:
        samples = super().sample(sample_shape)
        probs = self.probs
        
        # Trick to make samples differentiable with regard to probabilities
        samples += probs-probs.detach()

        return samples
