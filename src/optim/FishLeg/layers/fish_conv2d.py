import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch.nn import ParameterDict, Parameter

from .fish_base import FishModule
from typing import Tuple, Optional


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
                "L": Parameter(
                    torch.eye(
                        int(bias) + self.in_channels_eff * self.k_size, device=device
                    )
                ),
                "R": Parameter(torch.eye(out_channels, device=device)),
                "A": Parameter(
                    torch.ones(
                        out_channels,
                        int(bias) + self.in_channels_eff * self.k_size,
                        device=device,
                    )
                ),
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
            self.fishleg_aux["A"].data.mul_(np.sqrt(init_scale))
        else:
            if self._bias:
                w, b = v
                w = torch.reshape(w, (self.out_channels, -1))
                b = torch.reshape(b, (self.out_channels, 1))
                A = torch.cat([w, b], dim=-1)
            else:
                (w,) = v
                A = torch.reshape(w, (self.out_channels, -1))

            self.fishleg_aux["A"].data.copy_(A)

    def Qv(
        self, v: Tuple[Tensor, Optional[Tensor]], full: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Inspired by KFAC's conv2D layer by Grosse and Martens: Kronecker product of sizes (out_channels ⊗  (in_channels_eff * k_size))"""

        L = self.fishleg_aux["L"]
        R = self.fishleg_aux["R"]
        A = self.fishleg_aux["A"]

        if self._bias:
            w, b = v
            sw = w.shape
            sb = b.shape
            w = torch.reshape(w, (self.out_channels, -1))
            b = torch.reshape(b, (self.out_channels, 1))
            u = torch.cat([w, b], dim=-1)
        else:
            (w,) = v
            sw = w.shape
            u = torch.reshape(w, (self.out_channels, -1))

        # at this stage, u is out_channels * (in_channels_eff * k_size (perhaps +1))

        u = torch.linalg.multi_dot((R, (A * u), L.T))
        u = A * torch.linalg.multi_dot((R.T, u, L))

        if self._bias:
            return (torch.reshape(u[:, :-1], sw), torch.reshape(u[:, -1], sb))
        else:
            return (torch.reshape(u, sw),)

    def diagQ(self) -> Tensor:
        """Similar maths as the Linear layer"""

        L = self.fishleg_aux["L"]
        R = self.fishleg_aux["R"]
        A = self.fishleg_aux["A"]

        diagA = torch.reshape(A.T, (-1))
        diag = diagA * torch.kron(
            torch.sum(torch.square(L), dim=0), torch.sum(torch.square(R), dim=0)
        )

        if self._bias:
            w = diag[: -self.out_channels]
            b = diag[-self.out_channels :]
            w = torch.reshape(
                w, (self.in_channels_eff * self.k_size, self.out_channels)
            )
            w = torch.reshape(
                w.T,
                (
                    self.out_channels,
                    self.in_channels_eff,
                    self.kernel_size[0],
                    self.kernel_size[1],
                ),
            )
            b = torch.reshape(b, (self.out_channels))
            return (w, b)
        else:
            w = torch.reshape(
                w, (self.in_channels_eff * self.k_size, self.out_channels)
            )
            w = torch.reshape(
                w.T,
                (
                    self.out_channels,
                    self.in_channels_eff,
                    self.kernel_size[0],
                    self.kernel_size[1],
                ),
            )
            return (w,)
