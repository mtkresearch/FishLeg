import torch
from torch.distributions.categorical import Categorical
from torch.nn.functional import one_hot, log_softmax

from .likelihood_base import FishLikelihoodBase

__all__ = [
    "SoftMaxLikelihood",
]


class SoftMaxLikelihood(FishLikelihoodBase):
    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def nll(sef, preds: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        logits = log_softmax(preds, dim=1)
        return -torch.mean(torch.sum(logits * observations, dim=1))

    def draw(self, preds: torch.Tensor) -> torch.Tensor:
        # logits = torch.log(preds)
        logits = log_softmax(preds, dim=1)
        dense = Categorical(logits=logits).sample()
        return one_hot(dense, num_classes=logits.shape[-1])
