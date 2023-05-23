import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch.nn import ParameterDict, Parameter

from .fish_base import FishModule
from .utils import get_zero_grad_hook  # TODO: Is this in torch? Let's upgrade?
from typing import Any, List, Dict, Tuple, Callable, Optional

__all__ = [
    "FishLinear",
]


class FishLinear(nn.Linear, FishModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super(FishLinear, self).__init__(
            in_features, out_features, bias, device=device, dtype=dtype
        )

        self._layer_name = "Linear"
        self.fishleg_aux = ParameterDict(
            {
                "L": Parameter(torch.eye(in_features + 1)),
                "R": Parameter(torch.eye(out_features)),
                "A": Parameter(torch.ones(out_features, in_features + 1)),
            }
        )
        mask_L = torch.tril(torch.ones_like(self.fishleg_aux["L"])).to(device)
        self.fishleg_aux["L"].register_hook(get_zero_grad_hook(mask_L))

        mask_R = torch.triu(torch.ones_like(self.fishleg_aux["R"])).to(device)
        self.fishleg_aux["R"].register_hook(get_zero_grad_hook(mask_R))

        self.order = ["weight", "bias"]
        self.device = device

    def warmup(
        self,
        v: Tuple[Tensor, Tensor] = None,
        batch_speedup: bool = False,
        init_scale: float = 1.0,
    ) -> None:
        out_features, in_features = self.weight.shape
        if v is None:
            if batch_speedup:
                self.fishleg_aux["R"].data.mul_(np.sqrt(init_scale))
                self.fishleg_aux["L"].data.mul_(np.sqrt(init_scale))
            else:
                self.fishleg_aux["A"].data.mul_(np.sqrt(init_scale))
        else:
            A = torch.cat([v[0], v[1][:, None]], dim=-1)
            if batch_speedup:
                # nearest Kronecker product, using SVD
                # TODO: Check the below! This was D instead of A!
                U, S, Vh = torch.linalg.svd(A, full_matrices=False)
                self.fishleg_aux["R"].data.copy_(
                    torch.sqrt(torch.diag(torch.sqrt(S[0]) * U[:, 0]))
                )
                self.fishleg_aux["L"].data.copy_(
                    torch.sqrt(torch.diag(torch.sqrt(S[0]) * Vh[0, :]))
                )
            else:
                self.fishleg_aux["A"].data.copy_(A)

    def Qv(self, v: Tuple[Tensor, Tensor], full: bool = False) -> Tuple[Tensor, Tensor]:
        """For fully-connected layers, the default structure of :math:`Q` as a
        block-diaglonal matrix is,
        .. math::
                    Q_l = (R_lR_l^T \otimes L_lL_l^T)
        where :math:`l` denotes the l-th layer. The matrix :math:`R_l` has size
        :math:`(N_{l-1} + 1) \\times (N_{l-1} + 1)` while the matrix :math:`L_l` has
        size :math:`N_l \\times N_l`. The auxiliarary parameters :math:`\lambda`
        are represented by the matrices :math:`L_l, R_l`. For a Kronecker form that
        introduces full inner and outer diagonal rescaling structure is,

        .. math::
                    Q_l = A_l(L_l \otimes R_l^T) D_l^2 (L_l^T \otimes R_l) A_l

        where :math:`A_l` and :math:`D_l` are two diagonal matrices of the
        appropriate size.
        """
        L = self.fishleg_aux["L"]
        R = self.fishleg_aux["R"]
        u = torch.cat([v[0], v[1][:, None]], dim=-1)

        A = self.fishleg_aux["A"]
        u = A * u
        u = torch.linalg.multi_dot((R.T, R, u, L, L.T))
        u = A * u
        return (u[:, :-1], u[:, -1])

    def Qg(self) -> Tuple[Tensor, Tensor]:
        """Speed up Qg product, when batch size is smaller than parameter size.
        By chain rule:

        .. math::
                    DW_i = g_i\hat{a}^T_{i-1}
        where :math:`DW_i` is gradient of parameter of the ith layer, :math:`g_i` is
        gradient w.r.t output of ith layer and :math:`\hat{a}_i` is input to ith layer,
        and output of (i-1)th layer.
        """

        L = self.fishleg_aux["L"]
        R = self.fishleg_aux["R"]
        lft = torch.linalg.multi_dot((R.T, R, self._g))
        rgt = torch.linalg.multi_dot((self._a, L, L.T))
        z = lft @ rgt
        return (z[:, :-1], z[:, -1])

    def save_layer_input(self, input_: List[Tensor]) -> None:
        a = input_[0].to(self.device).clone()
        a = a.view(-1, a.size(-1))
        if self.bias is not None:
            a = torch.cat([a, a.new_ones((*a.shape[:-1], 1))], dim=-1)
        self._a = a

    def save_layer_grad_output(
        self,
        grad_output: Tuple[Tensor, ...],
    ) -> None:
        g = grad_output[0].to(self.device)
        g = g.view(-1, g.size(-1))
        self._g = g.T

    def diagQ(self) -> Tuple:
        """The Q matrix defines the inverse fisher approximation as below:

        .. math::
                    Q_l = (R_lR_l^T \otimes L_lL_l^T)

        where :math:`l` denotes the l-th layer. The matrix :math:`R_l` has size
        :math:`(N_{l-1} + 1) \\times (N_{l-1} + 1)` while the matrix :math:`L_l` has
        size :math:`N_l \\times N_l`. The auxiliarary parameters :math:`\lambda`
        are represented by the matrices :math:`L_l, R_l`.

        The diagonal of this matrix is therefore calculated by

        .. math::
                    \\text{diag}(Q_l) = \\text{diag}(R_l R_l^T) \otimes \\text{diag}(L_l L_l^T)

        where :math:`\\text{diag}` involves summing over the columns of the and :math:`\otimes` remains as
        the Kronecker product.

        """
        L = self.fishleg_aux["L"]
        R = self.fishleg_aux["R"]
        diag = torch.kron(torch.sum(L * L, dim=1), torch.sum(R * R, dim=0))
        diag = diag * torch.square(self.fishleg_aux["A"].T).reshape(-1)

        diag = diag.reshape(L.shape[0], R.shape[0]).T
        return (diag[:, :-1], diag[:, -1])
