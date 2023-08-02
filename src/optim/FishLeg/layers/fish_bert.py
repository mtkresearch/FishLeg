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
        init_scale: int = 1.0,
        device=None,
    ) -> None:
        super(FishBertAttention, self).__init__(config, position_embedding_type)

        self._layer_name = "BertSelfAttention"
        self.all_head_size, self.hidden_size = self.self.key.weight.shape
        self.fishleg_aux = ParameterDict(
            {
                "L": FishAuxParameter(torch.eye(self.hidden_size + 1)),
                "R": FishAuxParameter(torch.eye(self.hidden_size + 1)),
                #"U": FishAuxParameter(torch.eye(self.hidden_size)),
                "A": FishAuxParameter(torch.eye(self.hidden_size + 1)),
                "B": FishAuxParameter(torch.eye(self.hidden_size + 1)),
                #"C": FishAuxParameter(torch.eye(self.hidden_size + 1)),
                #"D": FishAuxParameter(torch.eye(self.hidden_size)),
                "K": FishAuxParameter(
                    torch.cat(
                        [torch.eye(self.hidden_size), torch.zeros(self.hidden_size,1)],
                        dim=1,
                    )
                ),
                "Q": FishAuxParameter(
                    torch.cat(
                        [torch.eye(self.hidden_size), torch.zeros(1,self.hidden_size)],
                        dim=0,
                    )
                ),
                #"V": FishAuxParameter(
                #    torch.eye(self.hidden_size + 1),
                #),
                #"O": FishAuxParameter(
                #    torch.eye(self.hidden_size),
                #),
                "Lv": FishAuxParameter(torch.eye(self.hidden_size + 1)),
                "Lo": FishAuxParameter(torch.eye(self.hidden_size + 1)),
                "Rv": FishAuxParameter(torch.eye(self.hidden_size)),
                "Ro": FishAuxParameter(torch.eye(self.hidden_size)),
                
                "Sk": FishAuxParameter(torch.ones(self.hidden_size, self.hidden_size+1).mul_(np.sqrt(init_scale))),
                "Sq": FishAuxParameter(torch.ones(self.hidden_size+1, self.hidden_size).mul_(np.sqrt(init_scale))),
                "Sv": FishAuxParameter(torch.ones(self.hidden_size, self.hidden_size+1).mul_(np.sqrt(init_scale))),
                "So": FishAuxParameter(torch.ones(self.hidden_size, self.hidden_size+1).mul_(np.sqrt(init_scale))),
            }
        )

        self.order = [
            "self.query.weight",
            "self.query.bias",
            "self.key.weight",
            "self.key.bias",
            "self.value.weight",
            "self.value.bias",
            "output.dense.weight",
            "output.dense.bias",
        ]
        self.device = device


    def Qv(self, v: Tuple, full=False) -> Tuple:
        Sk = self.fishleg_aux["Sk"]
        Sq = self.fishleg_aux["Sq"]
        Sv = self.fishleg_aux["Sv"]
        So = self.fishleg_aux["So"]

        Uk = Sk * torch.cat([v[2], v[3][:, None]], dim=-1)
        Uq = Sq * torch.transpose(torch.cat([v[0], v[1][:, None]], dim=-1), -1, -2)
        Uv = Sv * torch.cat([v[4], v[5][:, None]], dim=-1)
        Uo = So * torch.cat([v[6], v[7][:, None]], dim=-1)

        zkq = torch.linalg.multi_dot(
            (self.fishleg_aux["Q"], Uk, self.fishleg_aux["A"].T)
        ) + torch.linalg.multi_dot((self.fishleg_aux["B"].T, Uq, self.fishleg_aux["K"]))

        #zvo = torch.linalg.multi_dot(
        #    (self.fishleg_aux["C"].T, Uv, self.fishleg_aux["O"])
        #) + torch.linalg.multi_dot((self.fishleg_aux["V"], Uo, self.fishleg_aux["D"].T))

        L = self.fishleg_aux["L"]
        #U = self.fishleg_aux["U"]
        R = self.fishleg_aux["R"]

        zkq = torch.linalg.multi_dot((L, L.T, zkq, R, R.T))

        #zvo = torch.linalg.multi_dot((L, L.T, zvo, U, U.T))

        Vk = Sk * torch.linalg.multi_dot(
                (self.fishleg_aux["Q"].T, zkq, self.fishleg_aux["A"])
            )

        Vq = torch.transpose(Sq * torch.linalg.multi_dot(
            (self.fishleg_aux["B"], zkq, self.fishleg_aux["K"].T)
        ), -1,-2)

        #Vv = torch.transpose(Sv * torch.linalg.multi_dot(
        #    (self.fishleg_aux["C"], zvo, self.fishleg_aux["O"].T)
        #), -1,-2)

        #Vo = torch.transpose(So * torch.linalg.multi_dot(
        #    (self.fishleg_aux["V"].T, zvo, self.fishleg_aux["D"])
        #), -1,-2)
        Rv = self.fishleg_aux["Rv"]
        Lv = self.fishleg_aux["Lv"]
        Ro = self.fishleg_aux["Ro"]
        Lo = self.fishleg_aux["Lo"]

        Vv = Sv * torch.linalg.multi_dot((Rv, Rv.T, Uv, Lv, Lv.T))
        Vo = So * torch.linalg.multi_dot((Ro, Ro.T, Uo, Lo, Lo.T))

        return (
            Vq[:, :-1],
            Vq[:, -1],
            Vk[:, :-1],
            Vk[:, -1],
            Vv[:, :-1],
            Vv[:, -1],
            Vo[:, :-1],
            Vo[:, -1],
        )

    def diagQ(self) -> Tuple:
        L = self.fishleg_aux["L"]
        U = self.fishleg_aux["U"]
        R = self.fishleg_aux["R"]

        # K: (out, in+1)
        # Sk: (out, in+1)
        # diagk: column-wise, out, out, ..., bias
        # Sk -> (out, in+1), -> -1 -> in+1, in+1, ..., in+1

        # Q: (out, in+1)
        # Sq: (out, in+1)
        # diagq: out, out, ..., bias
        # Sq -> (in+1, out), -> -1 -> out, out, ..., bias

        diagk = torch.kron(
            torch.sum(torch.square(self.fishleg_aux["A"].T @ R), dim=-1),
            torch.sum(torch.square(self.fishleg_aux["Q"].T @ L), dim=-1),
        ) * torch.square(self.fishleg_aux["Sk"].T).reshape(-1)

        diagq = torch.kron(
            torch.sum(torch.square(self.fishleg_aux["K"] @ R), dim=-1),
            torch.sum(torch.square(self.fishleg_aux["B"] @ L), dim=-1),
        ) * torch.square(self.fishleg_aux["Sq"].T).reshape(-1)

        #diagv = torch.kron(
        #    torch.sum(torch.square(self.fishleg_aux["O"] @ U), dim=-1),
        #    torch.sum(torch.square(self.fishleg_aux["C"] @ L), dim=-1),
        #) * torch.square(self.fishleg_aux["Sv"].T).reshape(-1)
        #diago = torch.kron(
        #    torch.sum(torch.square(self.fishleg_aux["D"].T @ U), dim=-1),
        #    torch.sum(torch.square(self.fishleg_aux["V"].T @ L), dim=-1),
        #) * torch.square(self.fishleg_aux["So"].T).reshape(-1)
        
        diagv = torch.kron(
            torch.sum(torch.square(self.fishleg_aux['Lv']), dim=-1),
            torch.sum(torch.square(self.fishleg_aux['Rv']), dim=-1)
        ) * torch.square(self.fishleg_aux["Sv"].T).reshape(-1)
        diago = torch.kron(
            torch.sum(torch.square(self.fishleg_aux['Lo']), dim=-1),
            torch.sum(torch.square(self.fishleg_aux['Ro']), dim=-1)
        ) * torch.square(self.fishleg_aux["So"].T).reshape(-1)
        
        K = diagk.reshape(self.hidden_size+1, self.hidden_size).T
        Q = diagq.reshape(self.hidden_size, self.hidden_size+1)
        V = diagv.reshape(self.hidden_size+1, self.hidden_size).T
        O = diago.reshape(self.hidden_size+1, self.hidden_size).T

        return (
            Q[:, :-1],
            Q[:, -1],
            K[:, :-1],
            K[:, -1],
            V[:, :-1],
            V[:, -1],
            O[:, :-1],
            O[:, -1],
        )
