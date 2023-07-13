import torch
from torch.distributions.bernoulli import Bernoulli

from .likelihood_base import FishLikelihoodBase

__all__ = [
    "BernoulliLikelihood",
]


class BernoulliLikelihood(FishLikelihoodBase):
    r"""
    The Bernoulli likelihood used for classification.
    Using the standard Normal CDF :math:`\Phi(x)`) and the identity
    :math:`\Phi(-x) = 1-\Phi(x)`, we can write the likelihood as:

    .. math::
        p(y|f(x))=\Phi(yf(x))

    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def nll(self, preds: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        bce = torch.sum(preds * (1.0 - observations) + torch.nn.Softplus()(-preds))
        return bce / preds.shape[0]

    def draw(self, preds: torch.Tensor) -> torch.Tensor:
        return Bernoulli(logits=preds).sample()
