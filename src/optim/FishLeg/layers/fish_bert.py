from typing import Tuple
import torch
import numpy as np
from torch.nn import ParameterDict

from transformers.models.bert.modeling_bert import BertAttention
from .fish_base import FishModule, FishAuxParameter


class FishBertAttention(BertAttention, FishModule):
    def __init__(
        self,
        config,
        position_embedding_type=None,
        device=None,
    ) -> None:
        super(FishBertAttention, self).__init__(config, position_embedding_type)

        self._layer_name = "BertSelfAttention"
        self.all_head_size, self.hidden_size = self.self.key.weight.shape
        self.fishleg_aux = ParameterDict(
            {
                "L": FishAuxParameter(torch.eye(self.hidden_size + 1)),
                "R": FishAuxParameter(torch.eye(self.hidden_size + 1)),
                "U": FishAuxParameter(torch.eye(self.hidden_size)),
                "A": FishAuxParameter(torch.eye(self.hidden_size + 1)),
                "B": FishAuxParameter(torch.eye(self.hidden_size + 1)),
                "C": FishAuxParameter(torch.eye(self.hidden_size + 1)),
                "D": FishAuxParameter(torch.eye(self.hidden_size)),
                "K": FishAuxParameter(
                    torch.cat(
                        [torch.eye(self.hidden_size), torch.zeros(1, self.hidden_size)],
                        dim=0,
                    )
                ),
                "Q": FishAuxParameter(
                    torch.cat(
                        [torch.eye(self.hidden_size), torch.zeros(self.hidden_size, 1)],
                        dim=-1,
                    )
                ),
                "V": FishAuxParameter(
                    torch.eye(self.hidden_size + 1),
                ),
                "O": FishAuxParameter(
                    torch.eye(self.hidden_size),
                ),
                "Sk": FishAuxParameter(torch.ones(self.hidden_size + 1, self.hidden_size)),
                "Sq": FishAuxParameter(torch.ones(self.hidden_size, self.hidden_size + 1)),
                "Sv": FishAuxParameter(torch.ones(self.hidden_size, self.hidden_size + 1)),
                "So": FishAuxParameter(torch.ones(self.hidden_size, self.hidden_size + 1)),
            }
        )

        self.order = [
            "self.key.weight",
            "self.key.bias",
            "self.query.weight",
            "self.query.bias",
            "self.value.weight",
            "self.value.bias",
            "output.dense.weight",
            "output.dense.bias",
        ]
        self.device = device

    def warmup(
        self,
        v: Tuple = None,
        batch_speedup: bool = False,
        init_scale: float = 1.0,
    ) -> None:
        if v is None:
            self.fishleg_aux["Sk"].data.mul_(np.sqrt(init_scale))
            self.fishleg_aux["Sq"].data.mul_(np.sqrt(init_scale))
            self.fishleg_aux["Sv"].data.mul_(np.sqrt(init_scale))
            self.fishleg_aux["So"].data.mul_(np.sqrt(init_scale))
        else:
            self.fishleg_aux["Sk"].data.copy_(
                torch.cat([v[0], v[1][:, None]], dim=-1).T
            )
            self.fishleg_aux["Sq"].data.copy_(torch.cat([v[2], v[3][:, None]], dim=-1))
            self.fishleg_aux["Sv"].data.copy_(torch.cat([v[4], v[5][:, None]], dim=-1))
            self.fishleg_aux["So"].data.copy_(torch.cat([v[6], v[7][:, None]], dim=-1))

    def Qv(self, v: Tuple, full=False) -> Tuple:
        Sk = self.fishleg_aux["Sk"]
        Sq = self.fishleg_aux["Sq"]
        Sv = self.fishleg_aux["Sv"]
        So = self.fishleg_aux["So"]

        Uk = Sk * torch.transpose(torch.cat([v[0], v[1][:, None]], dim=-1), -1, -2)
        Uq = Sq * torch.cat([v[2], v[3][:, None]], dim=-1)
        Uv = Sv * torch.cat([v[4], v[5][:, None]], dim=-1)
        Uo = So * torch.cat([v[6], v[7][:, None]], dim=-1)

        zkq = torch.linalg.multi_dot(
            (self.fishleg_aux["A"].T, Uk, self.fishleg_aux["Q"])
        ) + torch.linalg.multi_dot((self.fishleg_aux["K"], Uq, self.fishleg_aux["B"].T))

        zvo = torch.linalg.multi_dot(
            (self.fishleg_aux["O"], Uv, self.fishleg_aux["C"].T)
        ) + torch.linalg.multi_dot((self.fishleg_aux["D"].T, Uo, self.fishleg_aux["V"]))

        L = self.fishleg_aux["L"]
        U = self.fishleg_aux["U"]
        R = self.fishleg_aux["R"]

        zkq = torch.linalg.multi_dot((R, R.T, zkq, L, L.T))

        zvo = torch.linalg.multi_dot((U, U.T, zvo, L, L.T))

        Vk = torch.transpose(
            Sk
            * torch.linalg.multi_dot(
                (self.fishleg_aux["A"], zkq, self.fishleg_aux["Q"].T)
            ),
            -1,
            -2,
        )

        Vq = Sq * torch.linalg.multi_dot(
            (self.fishleg_aux["K"].T, zkq, self.fishleg_aux["B"])
        )

        Vv = Sv * torch.linalg.multi_dot(
            (self.fishleg_aux["O"].T, zvo, self.fishleg_aux["C"])
        )

        Vo = So * torch.linalg.multi_dot(
            (self.fishleg_aux["D"], zvo, self.fishleg_aux["V"].T)
        )

        return (
            Vk[:, :-1],
            Vk[:, -1],
            Vq[:, :-1],
            Vq[:, -1],
            Vv[:, :-1],
            Vv[:, -1],
            Vo[:, :-1],
            Vo[:, -1],
        )

    def diagQ(self) -> Tuple:
        L = self.fishleg_aux["L"]
        U = self.fishleg_aux["U"]
        R = self.fishleg_aux["R"]

        # K: (in+1, out)
        # Sk: (in+1, out)
        # diagk: column-wise, in+1, in+1, ..., in+1
        # Sk -> (out, in+1), -> -1 -> in+1, in+1, ..., in+1

        # Q: (out, in+1)
        # Sq: (out, in+1)
        # diagq: out, out, ..., bias
        # Sq -> (in+1, out), -> -1 -> out, out, ..., bias

        diagk = torch.kron(
            torch.sum(self.fishleg_aux["Q"] @ L, dim=-1),
            torch.sum(self.fishleg_aux["A"] @ R, dim=-1),
        ) * torch.square(self.fishleg_aux["Sk"].T).reshape(-1)

        diagq = torch.kron(
            torch.sum(self.fishleg_aux["B"].T @ L, dim=-1),
            torch.sum(self.fishleg_aux["K"].T @ R, dim=-1),
        ) * torch.square(self.fishleg_aux["Sq"].T).reshape(-1)

        diagv = torch.kron(
            torch.sum(self.fishleg_aux["C"].T @ L, dim=-1),
            torch.sum(self.fishleg_aux["O"].T @ U, dim=-1),
        ) * torch.square(self.fishleg_aux["Sv"].T).reshape(-1)
        diago = torch.kron(
            torch.sum(self.fishleg_aux["V"] @ L, dim=-1),
            torch.sum(self.fishleg_aux["D"] @ U, dim=-1),
        ) * torch.square(self.fishleg_aux["So"].T).reshape(-1)

        K = diagk.reshape(self.all_head_size, self.hidden_size + 1)
        Q = diagq.reshape(self.hidden_size + 1, self.all_head_size).T
        V = diagv.reshape(self.hidden_size + 1, self.all_head_size).T
        O = diago.reshape(self.all_head_size + 1, self.hidden_size).T

        return (
            K[:, :-1],
            K[:, -1],
            Q[:, :-1],
            Q[:, -1],
            V[:, :-1],
            V[:, -1],
            O[:, :-1],
            O[:, -1],
        )
