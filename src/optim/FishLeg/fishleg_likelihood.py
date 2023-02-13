import torch
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from torch.nn.functional import one_hot, log_softmax

from abc import abstractmethod

__all__ = [
    "FishLikelihood",
    "FixedGaussianLikelihood",
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
        return 0.5 * (
            torch.square((observations - preds) / self.sigma).sum()
        ) / preds.shape[0] + torch.log(self.sigma**2)

    def draw(self, preds: torch.Tensor) -> torch.Tensor:
        return preds + torch.normal(0, self.sigma, size=preds.shape)


class BernoulliLikelihood(FishLikelihood):
    r"""
    The Bernoulli likelihood used for classification.
    Using the standard Normal CDF :math:`\Phi(x)`) and the identity
    :math:`\Phi(-x) = 1-\Phi(x)`, we can write the likelihood as:

    .. math::
        p(y|f(x))=\Phi(yf(x))

    """

    def __init__(self) -> None:
        pass

    def nll(self, observations: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:

        result = (
            -torch.sum(
                torch.log(1 - preds + 1e-5) * (1.0 - observations)
                + torch.log(preds + 1e-5) * observations
            )
            / preds.shape[0]
        )
        return result

    def draw(self, preds: torch.Tensor) -> torch.Tensor:
        return Bernoulli(probs=preds).sample()


class SoftMaxLikelihood(FishLikelihood):
    def __init__(self) -> None:
        pass

    def nll(sef, observations: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        logits = log_softmax(preds, dim=1)
        return -torch.mean(torch.sum(logits * observations, dim=1))

    def draw(self, preds: torch.Tensor) -> torch.Tensor:
        logits = torch.log(preds)
        dense = Categorical(logits=logits).sample()
        return one_hot(dense, num_classes=logits.shape[-1])
