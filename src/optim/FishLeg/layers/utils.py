import torch
from typing import Callable


# TODO: Is this already in torch somewhere? I can't remember if we had to define this ourselves or not?
def get_zero_grad_hook(mask: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    def hook(grad: torch.Tensor) -> torch.Tensor:
        if grad.get_device() >= 0:
            mask.to(grad.get_device())
        return grad * mask

    return hook
