import numpy as np
import torch
import torch.nn as nn
from torch.nn import ParameterDict, Parameter


class FishModel:
    def nll(self, data):
        data_x, data_y = data
        pred_y = self.forward(data_x)
        return self.likelihood.nll(None, pred_y, data_y)

    def sample(self, K):
        data_x = self.data[0][np.random.randint(0, self.N, K)]
        pred_y = self.forward(data_x)
        return (data_x, self.likelihood.sample(None, pred_y))


class FishLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super(FishLinear, self).__init__(
            in_features, out_features, bias, device=device, dtype=dtype
        )
        self.fishleg_aux = ParameterDict(
            {
                "scale": Parameter(torch.ones(size=(1,))),
                "L": Parameter(torch.eye(in_features + 1)),
                "R": Parameter(torch.eye(out_features)),
            }
        )
        self.order = ["weight", "bias"]

    @staticmethod
    def Qv(aux: dict, v: list):
        L, R = aux["fishleg_aux.L"], aux["fishleg_aux.R"]
        u = torch.cat([v[0], v[1][:, None]], dim=-1)
        z = aux["fishleg_aux.scale"] * torch.linalg.multi_dot((R, R.T, u, L, L.T))
        return [z[:, :-1], z[:, -1]]

    def cuda(self, device):
        super.cuda(device)
        L, R = self.fishleg_aux
        L.to(device)
        R.to(device)


FISH_LAYERS = {"linear": FishLinear}
