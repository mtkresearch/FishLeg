import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch.nn import ParameterDict

from .fish_base import FishModule, FishAuxParameter
from typing import Tuple, Optional


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
        dtype=None,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            device,
            dtype,
        )

        self._layer_name = "Embedding"
        self.fishleg_aux = ParameterDict(
            {
                "L": FishAuxParameter(torch.eye(embedding_dim)),
                "R": FishAuxParameter(torch.ones(num_embeddings)),
                "A": FishAuxParameter(torch.ones(num_embeddings, embedding_dim)),
            }
        )

        self.order = [
            "weight",
        ]
        self.device = device

    # TODO: Change to add_warmup_grad and finalise_warmup
    def warmup(
        self,
        v: Tuple[Tensor,] = None,
        init_scale: float = 1.0,
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
        diag = diag * torch.square(self.fishleg_aux["A"].T).reshape(-1)

        diag = diag.reshape(L.shape[0], R.shape[0]).T
        return (diag,)
