import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch.nn import ParameterDict, Parameter

from .fish_base import FishModule
from .utils import get_zero_grad_hook  # TODO: Is this in torch? Let's upgrade?
from typing import Tuple

__all__ = [
    "FishLinear",
]


class FishLinear(nn.Linear, FishModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_scale: int = 1.0,
        device: str or None = None,
        dtype: str or None = None,
    ) -> None:
        super(FishLinear, self).__init__(
            in_features, out_features, bias, device=device, dtype=dtype
        )

        self._layer_name = "Linear"
        self.fishleg_aux = ParameterDict(
            {
                "L": Parameter(torch.eye(in_features + 1)),
                "R": Parameter(torch.eye(out_features)),
                "A": Parameter(
                    torch.ones(out_features, in_features + 1).mul_(np.sqrt(init_scale))
                ),
            }
        )
        mask_L = torch.tril(torch.ones_like(self.fishleg_aux["L"])).to(device)
        self.fishleg_aux["L"].register_hook(get_zero_grad_hook(mask_L))

        mask_R = torch.triu(torch.ones_like(self.fishleg_aux["R"])).to(device)
        self.fishleg_aux["R"].register_hook(get_zero_grad_hook(mask_R))

        self.order = ["weight", "bias"]
        self.device = device

        self.warmup_state = torch.ones_like(self.fishleg_aux["A"]).to(device)

    def add_warmup_grad(
        self,
        grad: Tuple[Tensor, Tensor],
    ) -> None:
        # Add this into an overload of to() function?
        self.warmup_state = self.warmup_state.to(grad[0].device)

        self.warmup_state += torch.cat([grad[0], grad[1][:, None]], dim=-1)

    def finalise_warmup(self, damping: float, num_steps: int) -> None:
        self.fishleg_aux["A"].data.div_(self.warmup_state.div_(num_steps).add_(damping))

    def Qv(self, v: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        TODO: Check this...

        For fully-connected layers, the default structure of :math:`Q` as a
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

    def diagQ(self) -> Tuple:
        """
        TODO: Check this...

        The Q matrix defines the inverse fisher approximation as below:

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
