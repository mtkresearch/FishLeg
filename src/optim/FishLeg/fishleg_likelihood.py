import torch
import numpy as np
from typing import List
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from torch.nn.functional import one_hot, log_softmax
from torch.nn import Parameter


from abc import abstractmethod

__all__ = [
    "FishLikelihood",
    "FixedGaussianLikelihood",
    "GaussianLikelihood",
    "BernoulliLikelihood",
    "SoftMaxLikelihood",
]


class FishLikelihood:
    r"""
    A Likelihood in FishLeg specifies a probablistic modeling, which attributes
    the mapping from latent function values 
    :math:`f(\mathbf X)` to observed labels :math:`y`.

    For example, in the case of regression, 
    a Gaussian likelihood can be chosen, as

    .. math::
        y(\mathbf x) = f(\mathbf x) + \epsilon, \:\:\:\: \epsilon \sim N(0,\sigma^{2}_{n} \mathbf I)

    As for the case of classification, 
    a Bernoulli distribution can be chosen

    .. math::
            y(\mathbf x) = \begin{cases}
                1 & \text{w/ probability} \:\: \sigma(f(\mathbf x)) \\
                0 & \text{w/ probability} \:\: 1-\sigma(f(\mathbf x))
            \end{cases}
    
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def nll(self, observations, preds, **kwargs):
        r"""
        Computes the negative log-likelihood
        :math:`\ell(\theta, \mathcal D)=-\log p(\mathbf y|f(\mathbf x))`

        :param `torch.Tensor` observations: Values of :math:`y`.
        :param `torch.Tensor` preds: Predictions from model :math:`f(\mathbf x)`
        :rtype: `torch.Tensor`
        """
        raise NotImplementedError

    @abstractmethod
    def draw(self, preds, **kwargs):
        r"""
        Draw samples from the conditional distribution
        :math:`p(\mathbf y|f(\mathbf x))`

        :param `torch.Tensor` preds: Predictions from model :math:`f(\mathbf x)`
        """
        raise NotImplementedError

    def get_parameters(self) -> List:
        r"""
        return a list of learnable parameter.

        """
        return []

class FixedGaussianLikelihood(FishLikelihood):
    """
    The standard likelihood for regression,
    but assuming fixed heteroscedastic noise.

    .. math::
        p(y | f(x)) = f(x) + \epsilon, \:\:\:\: \epsilon \sim N(0,\sigma^{2})

    :param `torch.Tensor` sigma: Known observation
                            standard deviation for each example.

    """

    def __init__(self, sigma: torch.Tensor, device: str = "cpu") -> None:
        self.device = device
        self.sigma = torch.as_tensor(sigma).to(self.device)

    @property
    def get_variance(self) -> torch.Tensor:
        return self.sigma

    def nll(self, observations: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        logsigma2 = torch.log(torch.square(self.sigma))
        return 0.5 / preds.shape[0] * torch.sum(
                logsigma2 + torch.square((observations - preds) / self.sigma) 
            )  

    def draw(self, preds: torch.Tensor) -> torch.Tensor:
        return preds + torch.normal(0, self.sigma, size=preds.shape).to(self.device)

class GaussianLikelihood(FishLikelihood):
    """
    The standard likelihood for regression,
    but assuming fixed heteroscedastic noise.

    .. math::
        p(y | f(x)) = f(x) + \epsilon, \:\:\:\: \epsilon \sim N(0,\sigma^{2})

    :param `torch.Tensor` sigma: standard deviation for each example;
                    also to be learned during training.
    """
    def __init__(self, sigma: torch.Tensor, device: str = 'cpu') -> None:
        self.device = device
        self.sigma = Parameter(torch.tensor(sigma))
        self.sigma.to(self.device)

    def nll(self, observations: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        logsigma2 = torch.log(torch.square(self.sigma))
        return 0.5 / preds.shape[0] * torch.sum(
                logsigma2 + torch.square((observations - preds) / self.sigma) 
            )
    def draw(self, preds: torch.Tensor) -> torch.Tensor:
        return preds + torch.normal(0, self.sigma.data, size=preds.shape).to(self.device)

    def get_parameters(self) -> List:
        return [self.sigma,]

    def init_aux(self, init_scale) -> None:
        self.lam = Parameter(
                    torch.tensor(init_scale)
                )
        self.lam.to(self.device)
        self.order = ['lambda',]

    def get_aux_parameters(self) -> List:
        return [self.lam,]

    def Qv(self, v) -> List:
        return [torch.square(self.lam) * v[0],]

class BernoulliLikelihood(FishLikelihood):
    r"""
    The Bernoulli likelihood used for classification.
    Using the standard Normal CDF :math:`\Phi(x)`) and the identity
    :math:`\Phi(-x) = 1-\Phi(x)`, we can write the likelihood as:

    .. math::
        p(y|f(x))=\Phi(yf(x))

    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def nll(self, observations: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        bce = torch.sum(preds * (1. - observations) + torch.nn.Softplus()(-preds))
        return bce / preds.shape[0]

    def draw(self, preds: torch.Tensor) -> torch.Tensor:
        return Bernoulli(logits=preds).sample()


class SoftMaxLikelihood(FishLikelihood):
    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def nll(sef, observations: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        logits = log_softmax(preds, dim=1)
        return -torch.mean(torch.sum(logits * observations, dim=1))

    def draw(self, preds: torch.Tensor) -> torch.Tensor:
        logits = torch.log(preds)
        dense = Categorical(logits=logits).sample()
        return one_hot(dense, num_classes=logits.shape[-1])
