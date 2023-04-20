import torch
import sys
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch.nn import ParameterDict, Parameter
from abc import abstractmethod

from typing import Any, List, Dict, Tuple, Callable, Optional

__all__ = [
    "FishLinear",
    "FishConv2d"
]


class FishModule(nn.Module):
    """Base class for all neural network modules in FishLeg to

    #. Initialize auxiliary parameters, :math:`\lambda` and its forms, :math:`Q(\lambda)`.
    #. Specify quick calculation of :math:`Q(\lambda)v` products.

    :param torch.nn.ParameterDict fishleg_aux: auxiliary parameters
                with their initialization, including an additional parameter, scale,
                :math:`\eta`. Make sure that

                .. math::
                        - \eta_{init} Q(\lambda) grad = - \eta_{sgd} grad

                is hold in the beginning of the optimization
    :param List order: specify a name order of original parameter

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(FishModule, self).__init__(*args, **kwargs)
        self.__setattr__("fishleg_aux", ParameterDict())
        self.__setattr__("order", List)

    @property
    def name(self) -> str:
        return self._layer_name

    def cuda(self, device: str) -> None:
        super.cuda(device)
        for p in self.fishleg_aux.values:
            p.to(device)

    @abstractmethod
    def Qv(
        self, aux: Dict, v: Tuple[Tensor, ...], full: bool = False
    ) -> Tuple[Tensor, ...]:
        """:math:`Q(\lambda)` is a positive definite matrix which will effectively
        estimate the inverse damped Fisher Information Matrix. Appropriate choices
        for :math:`Q` should take into account the architecture of the model/module.
        It is usually parameterized as a positive definite Kronecker-factored
        block-diagonal matrix, with block sizes reflecting the layer structure of
        the neural networks.

        Args:
            aux: (Dict, required): auxiliary parameters,
                    :math:`\lambda`, a dictionary with keys, the name
                    of the auxiliary parameters, and values, the auxiliary parameters
                    of the module. These auxiliaray parameters will form :math:`Q(\lambda)`.
            v: (Tuple[Tensor, ...], required): Values of the original parameters,
                    in an order that align with `self.order`, to multiply with
                    :math:`Q(\lambda)`.
            full: (bool, optional), whether to use full inner and outer re-scaling
        Returns:
            Tuple[Tensor, ...]: The calculated :math:`Q(\lambda)v` products,
                    in same order with `self.order`.

        """
        raise NotImplementedError(f'Module is missing the required "Qv" function')


def get_zero_grad_hook(mask: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    def hook(grad: torch.Tensor) -> torch.Tensor:
        return grad * mask.to(grad.get_device())

    return hook


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
                "D": Parameter(torch.ones(out_features, in_features + 1)),
            }
        )
        mask_L = torch.triu(torch.ones_like(self.fishleg_aux["L"])).to(device)
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
                self.fishleg_aux["D"].data.mul_(np.sqrt(init_scale))
        else:
            D = torch.cat([v[0], v[1][:, None]], dim=-1)
            if batch_speedup:
                # nearest Kronecker product, using SVD
                U, S, Vh = torch.linalg.svd(D, full_matrices=False)
                self.fishleg_aux["R"].data.copy_(
                    torch.sqrt(torch.diag(torch.sqrt(S[0]) * U[:, 0]))
                )
                self.fishleg_aux["L"].data.copy_(
                    torch.sqrt(torch.diag(torch.sqrt(S[0]) * Vh[0, :]))
                )
            else:
                self.fishleg_aux["D"].data.copy_(D)

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

        if not full:
            u = torch.square(self.fishleg_aux["D"]) * u
            u = torch.linalg.multi_dot((R.T, R, u, L, L.T))
        else:
            A = self.fishleg_aux["A"]
            u = torch.linalg.multi_dot((R, (A * u), L))
            u = torch.square(self.fishleg_aux["D"]) * u
            u = A * torch.linalg.multi_dot((R.T, u, L.T))
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

    def diagQ(self) -> Tensor:
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
        diag = torch.kron(torch.sum(L * L, dim=0), torch.sum(R * R, dim=0))
        return (
            diag
            * torch.square(self.fishleg_aux["D"].T).reshape(-1)
            * torch.square(self.fishleg_aux["A"].T).reshape(-1)
        )

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
        super(FishConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self._layer_name = "Conv2d"

        self.k_size = self.kernel_size[0] * self.kernel_size[1]
        eff_scale = init_scale ** (1.0 / 3)
        self.fishleg_aux = ParameterDict(
            {
                "L_o": Parameter(torch.eye(out_channels, device=device) * eff_scale),
                "L_i": Parameter(torch.eye(in_channels, device=device) * eff_scale),
                "L_k": Parameter(torch.eye(self.k_size, device=device) * eff_scale),
                "L_b": Parameter(torch.eye(out_channels, device=device) * eff_scale),
            }
        )

        self.order = ["weight", "bias"] if bias else ["weight"]

    def Qv(self, v: Tuple[Tensor, Optional[Tensor]]) -> Tuple[Tensor, Optional[Tensor]]:
        """For fully-connected layers, the default structure of :math:`Q` as a
        block-diaglonal matrix is,
        .. math::
                    Q_l = (R_lR_l^T \otimes L_lL_l^T)
        where :math:`l` denotes the l-th layer. The matrix :math:`R_l` has size
        :math:`(N_{l-1} + 1) \\times (N_{l-1} + 1)` while the matrix :math:`L_l` has
        size :math:`N_l \\times N_l`. The auxiliarary parameters :math:`\lambda`
        are represented by the matrices :math:`L_l, R_l`.
        """

        # Qv product for the weights
        W = v[0]
        L_o = self.fishleg_aux["L_o"]
        L_i = self.fishleg_aux["L_i"]
        L_k = self.fishleg_aux["L_k"]
        L_b = self.fishleg_aux["L_b"]

        tmp = torch.reshape(W, (-1, self.k_size))
        tmp = torch.matmul(torch.matmul(tmp, L_k), L_k.T)
        tmp = torch.reshape(tmp, (self.out_channels, self.in_channels, self.k_size))
        tmp = torch.transpose(tmp, 1, 2)
        tmp = torch.reshape(tmp, (-1, self.in_channels))
        tmp = torch.matmul(torch.matmul(tmp, L_i), L_i.T)
        tmp = torch.reshape(tmp, (self.out_channels, self.k_size, self.in_channels))
        tmp = torch.transpose(tmp, 0, 2)
        tmp = torch.reshape(tmp, (-1, self.out_channels))
        tmp = torch.matmul(torch.matmul(tmp, L_o), L_o.T)
        tmp = torch.reshape(tmp, (self.in_channels, self.k_size, self.out_channels))
        tmp = tmp.permute((2, 0, 1))
        qvW = torch.reshape(
            tmp,
            (
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
            ),
        )

        if self.bias:
            b = v[1]
            bs = tuple(torch.size(bs))
            b = torch.reshape(b, (-1, 1))
            qvB = torch.matmul(L_b, torch.matmul(L_b.T, b))
            qvB = torch.reshape(qvB, bs)
            return (qvW, qvB)
        else:
            return (qvW,)
