import torch
from typing import List
from torch.nn import Parameter

from .likelihood_base import FishLikelihoodBase

__all__ = [
    "GaussianLikelihood",
]


class GaussianLikelihood(FishLikelihoodBase):
    """
    The standard likelihood for regression,
    but assuming fixed heteroscedastic noise.

    .. math::
        p(y | f(x)) = f(x) + \epsilon, \:\:\:\: \epsilon \sim N(0,\sigma^{2})

    :param `torch.Tensor` sigma: standard deviation for each example;
                    also to be learned during training.
    """

    def __init__(self, sigma: torch.Tensor, device: str = "cpu") -> None:
        self.device = device
        self.sigma = Parameter(torch.tensor(sigma))
        self.sigma.to(self.device)

    def nll(self, preds: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        logsigma2 = torch.log(torch.square(self.sigma))
        return (
            0.5
            / preds.shape[0]
            * torch.sum(logsigma2 + torch.square((observations - preds) / self.sigma))
        )

    def draw(self, preds: torch.Tensor) -> torch.Tensor:
        return preds + torch.normal(0, self.sigma.data, size=preds.shape).to(
            self.device
        )
