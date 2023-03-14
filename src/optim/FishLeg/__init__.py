from .fishleg import FishLeg
from .fishleg_layers import *
from .fishleg_likelihood import *

FISH_LAYERS = {"linear": FishLinear}
FISH_LIKELIHOODS = {
    "fixedgaussian": FixedGaussianLikelihood,
    "gaussian": GaussianLikelihood,
    "bernoulli": BernoulliLikelihood,
    "softmax": SoftMaxLikelihood,
}
