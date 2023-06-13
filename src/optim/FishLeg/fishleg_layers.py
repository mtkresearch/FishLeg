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
    "FishEmbedding",
    "FishConv2d",
    "FishBatchNorm2d",
    "FishLayerNorm"
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
        for p in self.fishleg_aux.values():
            p.to(device)

    def warmup(
        self,
        v: Tuple[Tensor, Tensor] = None,
        batch_speedup: bool = False,
        init_scale: float = 1.0,
    ):
        pass

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
        if grad.get_device() >= 0:
            mask.to(grad.get_device())
        return grad * mask

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

        size = in_features + 1 if bias else in_features
        self.fishleg_aux = ParameterDict(
            {
                "L": Parameter(torch.eye(size)),
                "R": Parameter(torch.eye(out_features)),
                "scaleA": Parameter(torch.ones(out_features, size)),
            }
        )
        mask_L = torch.tril(torch.ones_like(self.fishleg_aux["L"])).to(device)
        self.fishleg_aux["L"].register_hook(get_zero_grad_hook(mask_L))

        mask_R = torch.triu(torch.ones_like(self.fishleg_aux["R"])).to(device)
        self.fishleg_aux["R"].register_hook(get_zero_grad_hook(mask_R))

        self.order = ["weight", "bias"] if bias else ["weight"]
        self.device = device
        self._bias = bias

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
                self.fishleg_aux["scaleA"].data.mul_(np.sqrt(init_scale))
        else:
            if self._bias:
                A = torch.cat([v[0], v[1][:, None]], dim=-1)
            else:
                A = v[0]
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
                self.fishleg_aux["scaleA"].data.copy_(A)

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
        if self._bias:
            u = torch.cat([v[0], v[1][:, None]], dim=-1)
        else:
            u = v[0]
        
        A = self.fishleg_aux["scaleA"]
        u = A * u
        u = torch.linalg.multi_dot((R.T, R, u, L, L.T))
        u = A * u
        return (u[:, :-1], u[:, -1]) if self._bias else (u,)

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
        return (z[:, :-1], z[:, -1]) if self._bias else (z,)

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
        diag = diag * \
                torch.square(self.fishleg_aux["scaleA"].T).reshape(-1)

        diag = diag.reshape(L.shape[0], R.shape[0]).T
        return (
            diag[:, :-1], diag[:, -1]
            ) if self._bias else (diag,)

class FishEmbedding(nn.Embedding, FishModule):
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        padding_idx: Optional[int] = None, 
        max_norm: Optional[float] = None, 
        norm_type: float = 2, 
        scale_grad_by_freq: bool = False, 
        sparse: bool = False, 
        _weight: Optional[Tensor] = None, 
        device=None, 
        dtype=None
    ) -> None:
        super().__init__(
            num_embeddings, 
            embedding_dim, 
            padding_idx, 
            max_norm, norm_type, scale_grad_by_freq, sparse, _weight, 
            device, dtype)
        
        self._layer_name = "Embedding"
        self.fishleg_aux = ParameterDict(
            {
                "L": Parameter(torch.eye(embedding_dim)),
                "R": Parameter(torch.ones(num_embeddings)),
                "A": Parameter(torch.ones(num_embeddings, embedding_dim)),
            }
        )

        self.order = ["weight",]
        self.device = device

    def warmup(
        self,
        v: Tuple[Tensor,] = None,
        init_scale: float = 1.0,
        batch_speedup: bool = False,
    ) -> None:
        if v is None:
            self.fishleg_aux["A"].data.mul_(np.sqrt(init_scale))
        else:
            self.fishleg_aux["A"].data.copy_(v[0])

    def Qv(self, v: Tuple[Tensor,], full: bool = False) -> Tuple[Tensor,]:

        L = self.fishleg_aux["L"]
        R = self.fishleg_aux["R"]
        u = v[0]

        
        A = self.fishleg_aux["A"]
        u = A * u
        u = torch.linalg.multi_dot((torch.diag(R), torch.diag(R), u, L, L.T))
        u = A * u
        return (u,)

    def diagQ(self) -> Tuple:
        L = self.fishleg_aux["L"]
        R = self.fishleg_aux["R"]
        diag = torch.kron(torch.sum(L * L, dim=1), R * R)
        diag = diag * \
                torch.square(self.fishleg_aux["A"].T).reshape(-1)

        diag = diag.reshape(L.shape[0], R.shape[0]).T
        return (diag,)


class FishConv2d(nn.Conv2d, FishModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
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
            device=device,
        )
        self._layer_name = "Conv2d"
        self.in_channels_eff = self.in_channels / self.groups
        self.k_size = self.kernel_size[0] * self.kernel_size[1]
        self.fishleg_aux = ParameterDict(
                {
                    "L": Parameter(torch.eye(int(bias) + self.in_channels_eff * self.k_size, device=device)),
                    "R": Parameter(torch.eye(out_channels, device=device)),
                    "scaleA": Parameter(torch.ones(out_channels, int(bias) + self.in_channels_eff * self.k_size, device=device)),
                }
            )
        self.order = ["weight", "bias"] if bias else ["weight"]
        self._bias = bias

    def warmup(
        self,
        v: Tuple[Tensor, Tensor] = None,
        init_scale: float = 1.0,
    ) -> None:
        if v is None:
            self.fishleg_aux["scaleA"].data.mul_(np.sqrt(init_scale))
        else:
            if self._bias:
                w, b = v
                w = torch.reshape(w, (self.out_channels, -1))
                b = torch.reshape(b, (self.out_channels, 1))
                A = torch.cat([w, b], dim=-1)
            else:
                w, = v
                A = torch.reshape(w, (self.out_channels, -1))
 
            self.fishleg_aux["scaleA"].data.copy_(A)


    def Qv(
        self, v: Tuple[Tensor, Optional[Tensor]], full: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Inspired by KFAC's conv2D layer by Grosse and Martens: Kronecker product of sizes (out_channels ⊗  (in_channels_eff * k_size))
        """

        L = self.fishleg_aux["L"]
        R = self.fishleg_aux["R"]
        A = self.fishleg_aux["scaleA"]

        if self._bias:
            w, b = v
            sw = w.shape
            sb = b.shape
            w = torch.reshape(w, (self.out_channels, -1))
            b = torch.reshape(b, (self.out_channels, 1))
            u = torch.cat([w, b], dim=-1)
        else:
            w, = v
            sw = w.shape
            u = torch.reshape(w, (self.out_channels, -1))

        # at this stage, u is out_channels * (in_channels_eff * k_size (perhaps +1))

        u = torch.linalg.multi_dot((R, (A * u), L.T))
        u = A * torch.linalg.multi_dot((R.T, u, L))

        if self._bias:
            return (torch.reshape(u[:, :-1], sw), torch.reshape(u[:, -1], sb))
        else:
            return (torch.reshape(u, sw), )


    def diagQ(self) -> Tensor:
        """Similar maths as the Linear layer"""

        L = self.fishleg_aux["L"]
        R = self.fishleg_aux["R"]
        A = self.fishleg_aux["scaleA"]

        diagA = torch.reshape(A.T, (-1))
        diag = diagA * torch.kron(torch.sum(torch.square(L), dim=0), torch.sum(torch.square(R), dim=0))

        if self._bias:
            w = diag[: -self.out_channels]
            b = diag[-self.out_channels :]
            w = torch.reshape(w, (self.in_channels_eff * self.k_size, self.out_channels))
            w = torch.reshape(w.T, (self.out_channels, self.in_channels_eff, self.kernel_size[0], self.kernel_size[1]))
            b = torch.reshape(b, (self.out_channels))
            return (w, b)
        else:
            w = torch.reshape(w, (self.in_channels_eff * self.k_size, self.out_channels))
            w = torch.reshape(w.T, (self.out_channels, self.in_channels_eff, self.kernel_size[0], self.kernel_size[1]))
            return (w, )


class FishBatchNorm2d(nn.BatchNorm2d, FishModule):

    def __init__(self, num_features: int, 
                       eps: float = 0.00001, 
                       momentum: float = 0.1, 
                       affine: bool = True, 
                       track_running_stats: bool = True, 
                       init_scale = 1.0,
                       device=None, 
                       dtype=None) -> None:

        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)()
        self._layer_name = "BatchNorm2d"
        if affine:
            self.fishleg_aux = ParameterDict(
                {
                "L_w": Parameter(torch.ones((num_features,), device=device) * np.sqrt(init_scale)),
                "L_b": Parameter(torch.ones((num_features,), device=device) * np.sqrt(init_scale)),
                }
            )

        self.order = ["weight", "bias"]

    def Qv(self, v: Tuple, full=False):

        return (
            torch.square(self.fishleg_aux['L_w']) * v[0],
            torch.square(self.fishleg_aux['L_b']) * v[1]
        )

    def diagQ(self):
        return (
            torch.square(self.fishleg_aux['L_w']),
            torch.square(self.fishleg_aux['L_b'])
        )

class FishLayerNorm(nn.LayerNorm, FishModule):

    def __init__(self, normalized_shape, 
                       eps: float = 0.00001, 
                       elementwise_affine: bool = True, 
                       init_scale = 1.0,
                       device=None, 
                       dtype=None) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine, device, dtype)

        self._layer_name = "LayerNorm"

        if elementwise_affine:
            self.fishleg_aux = ParameterDict(
                {
                "scalew": Parameter(torch.ones(normalized_shape, device=device)),
                "scaleb": Parameter(torch.ones(normalized_shape, device=device)),
                }
            )

        self.order = ["weight", "bias"]
    
    def warmup(
        self,
        v: Tuple[Tensor,] = None,
        init_scale: float = 1.0,
        batch_speedup: bool = False,
    ) -> None:
        if v is None:
            self.fishleg_aux["scalew"].data.mul_(np.sqrt(init_scale))
            self.fishleg_aux["scaleb"].data.mul_(np.sqrt(init_scale))
        else:
            self.fishleg_aux["scalew"].data.copy_(v[0])
            self.fishleg_aux["scaleb"].data.copy_(v[1])

    def Qv(self, v: Tuple, full=False):

        return (
            torch.square(self.fishleg_aux['scalew']) * v[0],
            torch.square(self.fishleg_aux['scaleb']) * v[1]
        )

    def diagQ(self):
        return (
            torch.square(self.fishleg_aux['scalew']),
            torch.square(self.fishleg_aux['scaleb'])
        )
