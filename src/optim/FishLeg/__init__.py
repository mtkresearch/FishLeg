from .fishleg import FishLeg
from .layers import *
from .fishleg_likelihood import *
from .utils import *

FISH_LAYERS = {"linear": FishLinear}
FISH_LIKELIHOODS = {
    "fixedgaussian": FixedGaussianLikelihood,
    "gaussian": GaussianLikelihood,
    "bernoulli": BernoulliLikelihood,
    "softmax": SoftMaxLikelihood,
}
