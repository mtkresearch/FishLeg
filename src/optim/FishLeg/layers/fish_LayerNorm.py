import torch
import torch.nn as nn
from torch.nn import ParameterDict

from .fish_base import FishModule, FishAuxParameter
from typing import Tuple, Optional


class FishLayerNorm(nn.LayerNorm, FishModule):
    def __init__(
        self,
        normalized_shape,
        eps: float = 0.00001,
        elementwise_affine: bool = True,
        init_scale: Optional[float] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine, device, dtype)

        self._layer_name = "LayerNorm"

        if elementwise_affine:
            self.fishleg_aux = ParameterDict(
                {
                    "L_w": FishAuxParameter(
                        torch.ones(normalized_shape, device=device)
                        # * np.sqrt(init_scale) # TODO: CHECK
                    ),
                    "L_b": FishAuxParameter(
                        torch.ones(normalized_shape, device=device)
                        # * np.sqrt(init_scale)
                    ),
                }
            )

        self.order = ["weight", "bias"]

    def Qv(self, v: Tuple, full=False):
        return (
            torch.square(self.fishleg_aux["L_w"]) * v[0],
            torch.square(self.fishleg_aux["L_b"]) * v[1],
        )

    def diagQ(self):
        return (
            torch.square(self.fishleg_aux["L_w"]),
            torch.square(self.fishleg_aux["L_b"]),
        )
