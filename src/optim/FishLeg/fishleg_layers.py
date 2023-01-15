import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import ParameterDict, Parameter

from typing import Any, List, Dict, Tuple


class FishModel:
    def nll(self, data):
        data_x, data_y = data
        pred_y = self.forward(data_x)
        return self.likelihood.nll(None, pred_y, data_y)

    def sample(self, K):
        data_x = self.data[0][np.random.randint(0, self.N, K)]
        pred_y = self.forward(data_x)
        return (data_x, self.likelihood.sample(None, pred_y))


class FishModule(nn.Module):
    """Base class for all neural network modules in FishLeg to 
    1. Initialize auxiliary parameters, λ and its forms, Q(λ)
    2. Specify quick calculation of Q(λ)v products
    
    Args:
        fishleg_aux (ParameterDict): auxiliary parameters with their
                initialization, including an additional parameter, scale, η. 
                Make sure that 
                        - η Q(λ) gradient = - η_adam gradient
                is hold in the beginning of the optimization
        order (List): specify a name order of original parameter

    Methods:
        Qv (Dict, Tuple) -> Tuple: required for each module
    """


    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(FishModule, self).__init__()
        self.__setattr__('fishleg_aux', ParameterDict())
        self.__setattr__('order', List)
    
    @staticmethod
    def Qv(aux: Dict, v: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        raise NotImplementedError(f"Module is missing the required \"Qv\" function")


class FishLinear(nn.Linear, FishModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device = None,
        dtype = None,
    ) -> None:
        super(FishLinear, self).__init__(
            in_features, out_features, bias, device=device, dtype=dtype
        )
        self._layer_name = "Linear"
        self.fishleg_aux = ParameterDict(
            {
                "scale": Parameter(torch.ones(size=(1,))),
                "L": Parameter(torch.eye(in_features + 1)),
                "R": Parameter(torch.eye(out_features)),
            }
        )
        self.order = ["weight", "bias"]

    @property
    def name(self) -> str:
        return self._layer_name

    @staticmethod
    def Qv(aux: Dict, v: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        L, R = aux["fishleg_aux.L"], aux["fishleg_aux.R"]
        u = torch.cat([v[0], v[1][:, None]], dim=-1)
        z = aux["fishleg_aux.scale"] * torch.linalg.multi_dot((R, R.T, u, L, L.T))
        return (z[:, :-1], z[:, -1])

    def cuda(self, device) -> None:
        super.cuda(device)
        for p in self.fishleg_aux.values:
            p.to(device)


FISH_LAYERS = {
    "linear": FishLinear
}  # Perhaps this would be better constructed inside the __init__.py file?
