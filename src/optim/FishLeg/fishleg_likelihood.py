import torch
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import  Bernoulli
from torch.nn.functional import one_hot, log_softmax

from abc import abstractmethod

#####
# TODO: Add an abstract syntax class here for users to be able to create their own custom likelihoods.
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
    def __init__(self):
        pass

    @abstractmethod
    def nll(
        self, 
        observations, 
        preds, 
        **kwargs
    ):
        r"""
        Computes the negative log-likelihood 
        :math:`\ell(\theta, \mathcal D)=-\log p(\mathbf y|f(\mathbf x))`

        :param `torch.Tensor` observations: Values of :math:`y`.
        :param `torch.Tensor` preds: Predictions from model :math:`f(\mathbf x)`
        :rtype: `torch.Tensor`
        """
        raise NotImplementedError


    @abstractmethod
    def draw(
        self,
        preds,
        **kwargs
    ):
        r"""
        Draw samples from the conditional distribution 
        :math:`p(\mathbf y|f(\mathbf x))`

        :param `torch.Tensor` preds: Predictions from model :math:`f(\mathbf x)`
        """
        raise NotImplementedError


#####

# Note, need to check that the recuction of the nll is correct, default reducation is mean
class FixedGaussianLikelihood(FishLikelihood):
    """
    The standard likelihood for regression, 
    but assuming fixed heteroscedastic noise. 
    
    .. math::
        p(y | f(x)) = f(x) + \epsilon, \:\:\:\: \epsilon \sim N(0,\sigma^{2})

    :param `torch.Tensor` sigma_fixed: Known observation 
                            standard deviation for each example.

    """
    def __init__(self, sigma_fixed):
        self.sigma_fixed = torch.as_tensor(sigma_fixed)

    @property
    def get_variance(self):
        return self.sigma_fixed

    def nll(self, observations: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        return 0.5 * (
            torch.square((observations - preds) / self.sigma_fixed).mean(dim=0).sum()
            + torch.log(self.sigma_fixed**2)
        ) 

    def draw(self, preds: torch.Tensor) -> torch.Tensor:
        return preds + torch.normal(0, self.sigma_fixed, size=preds.shape)


class BernoulliLikelihood(FishLikelihood):
    r"""
    The Bernoulli likelihood used for classification. 
    Using the standard Normal CDF :math:`\Phi(x)`) and the identity
    :math:`\Phi(-x) = 1-\Phi(x)`, we can write the likelihood as:

    .. math::
        p(y|f(x))=\Phi(yf(x))

    """
    def __init__(self):
        pass


    def nll(self, observations: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        max_val = torch.clip(-preds, 0, None)
        return torch.sum(preds *(1.0- observations) + max_val + torch.log(torch.exp(-max_val) \
                      + torch.exp((-preds - max_val))))/preds.shape[0]

    def draw(self, preds: torch.Tensor) -> torch.Tensor:
        return Bernoulli(logits=preds).sample()


class SoftMaxLikelihood(FishLikelihood):
    def __init__(self):
        pass

    def nll(sef, observations: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        logits = log_softmax(preds, dim=1)
        return -torch.mean(torch.sum(logits * observations, dim=1))

    def draw(self, preds: torch.Tensor) -> torch.Tensor:
        logits = torch.log(preds)
        dense = Categorical(logits=logits).sample()
        return one_hot(dense, num_classes=logits.shape[-1])
