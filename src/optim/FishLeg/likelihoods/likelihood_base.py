from typing import List

from abc import abstractmethod

__all__ = [
    "FishLikelihoodBase",
]


class FishLikelihoodBase:
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
    def nll(self, preds, observations, **kwargs):
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

    def __call__(self, preds, observations, **kwargs):
        return self.nll(preds, observations, **kwargs)
