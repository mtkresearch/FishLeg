import torch
import torch.nn as nn
import numpy as np
from torch.nn import ParameterDict

from .fish_base import FishModule, FishAuxParameter
from typing import Tuple


class FishBatchNorm2d(nn.BatchNorm2d, FishModule):
    def __init__(
        self,
        num_features: int,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        init_scale = None,
        device=None,
        dtype=None,
    ) -> None:
        super(FishBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, device, dtype
        )
        self._layer_name = "BatchNorm2d"
        if affine:
            self.fishleg_aux = ParameterDict(
                {
                    "L_w": FishAuxParameter(
                        torch.ones(
                            (num_features,), device=device
                        )  # * np.sqrt(init_scale) # TODO: CHECK
                    ),
                    "L_b": FishAuxParameter(
                        torch.ones(
                            (num_features,), device=device
                        )  # * np.sqrt(init_scale)
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
