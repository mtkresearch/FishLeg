from typing import Tuple
import torch
from torch.nn import ParameterDict, Parameter

from transformers.models.bert.modeling_bert import BertAttention
from ..fishleg_layers import FishModule

class FishBertAttention(BertAttention, FishModule):
    def __init__(
        self,
        config,
        position_embedding_type=None, ##?
        init_scale = 1.0,
        device = None,
        dtype = None,
    ) -> None:
        super(FishBertAttention, self).__init__(
            config, position_embedding_type
        )

        self._layer_name = "BertSelfAttention"
        self.all_head_size, self.hidden_size = self.self.key.weight.shape

        eff_scale = init_scale ** (1.0 / 8)
        self.fishleg_aux = ParameterDict(
            {
                "L": Parameter(torch.eye(self.hidden_size + 1) * eff_scale),
                "R": Parameter(torch.eye(self.hidden_size + 1) * eff_scale),
                "U": Parameter(torch.eye(self.hidden_size) * eff_scale),
                "A": Parameter(torch.eye(self.hidden_size + 1) * eff_scale),
                "B": Parameter(torch.eye(self.hidden_size + 1) * eff_scale),
                "C": Parameter(torch.eye(self.hidden_size + 1) * eff_scale),
                "D": Parameter(torch.eye(self.hidden_size) * eff_scale),

                "K": Parameter(
                        torch.cat([
                            torch.eye(self.hidden_size) * eff_scale,
                            torch.zeros(1, self.hidden_size)
                        ], dim=0)
                    ),
                "Q": Parameter(
                        torch.cat([
                            torch.eye(self.hidden_size) * eff_scale,
                            torch.zeros(self.hidden_size, 1)
                        ], dim=-1)
                    ),
                "V": Parameter(
                        torch.eye(self.hidden_size + 1) * eff_scale,
                    ),
                "O": Parameter(
                        torch.eye(self.hidden_size) * eff_scale,
                    )
            }
        )

        self.order = [
                "self.key.weight", "self.key.bias",\
                "self.query.weight", "self.query.bias",\
                "self.value.weight", "self.value.bias",\
                "output.dense.weight", "output.dense.bias"
            ]
        self.device = device

    
    def Qv(self, v: Tuple, full=False) -> Tuple:
        Uk = torch.transpose(
                torch.cat([v[0], v[1][:, None]], dim=-1),
                -1,-2
            )
        Uq = torch.cat([v[2], v[3][:, None]], dim=-1)
        Uv = torch.cat([v[4], v[5][:, None]], dim=-1)
        Uo = torch.cat([v[6], v[7][:, None]], dim=-1)

        zkq = torch.linalg.multi_dot((
                self.fishleg_aux["A"].T,
                Uk,
                self.fishleg_aux["Q"]
        )) + torch.linalg.multi_dot((
                self.fishleg_aux["K"],
                Uq,
                self.fishleg_aux["B"].T
        )) 

        zvo = torch.linalg.multi_dot((
                self.fishleg_aux["O"],
                Uv,
                self.fishleg_aux["C"].T
        )) + torch.linalg.multi_dot((
                self.fishleg_aux["D"].T,
                Uo,
                self.fishleg_aux["V"]
        )) 
   
        L = self.fishleg_aux["L"]
        U = self.fishleg_aux["U"]
        R = L@self.fishleg_aux["R"]

        zkq = torch.linalg.multi_dot((
                R,R.T,zkq,L,L.T
        ))

        zvo = torch.linalg.multi_dot((
                U,U.T,zvo,L,L.T
        ))

        Vk = torch.transpose(
            torch.linalg.multi_dot((
                self.fishleg_aux["A"],
                zkq,
                self.fishleg_aux["Q"].T
            )),
            -1,-2
        )

        Vq = torch.linalg.multi_dot((
                self.fishleg_aux["K"].T,
                zkq,
                self.fishleg_aux["B"]
        ))

        Vv = torch.linalg.multi_dot((
                self.fishleg_aux["O"].T,
                zvo,
                self.fishleg_aux["C"]
        ))

        Vo = torch.linalg.multi_dot((
                self.fishleg_aux["D"],
                zvo,
                self.fishleg_aux["V"].T
        ))

        return (
            Vk[:, :-1], Vk[:, -1],\
            Vq[:, :-1], Vq[:, -1],\
            Vv[:, :-1], Vv[:, -1],
            Vo[:, :-1], Vo[:, -1]
        )

    def diagQ(self) -> Tuple:
        L = self.fishleg_aux["L"]
        U = self.fishleg_aux["U"]
        R = L@self.fishleg_aux["R"]

        diagk = torch.kron(torch.sum(self.fishleg_aux["Q"]@L, dim=-1), 
                        torch.sum(self.fishleg_aux["A"]@R, dim=-1))
        diagq = torch.kron(torch.sum(self.fishleg_aux["B"].T@L, dim=-1), 
                        torch.sum(self.fishleg_aux["K"].T@R, dim=-1))

        diagv = torch.kron(torch.sum(self.fishleg_aux["C"].T@L, dim=-1), 
                        torch.sum(self.fishleg_aux["O"].T@U, dim=-1))
        diago = torch.kron(torch.sum(self.fishleg_aux["V"]@L, dim=-1), 
                        torch.sum(self.fishleg_aux["D"]@U, dim=-1))
        

        K = diagk.reshape(self.all_head_size, self.hidden_size + 1)
        Q = diagq.reshape(self.hidden_size + 1, self.all_head_size).T
        V = diagv.reshape(self.hidden_size + 1, self.all_head_size).T
        O = diago.reshape(self.all_head_size + 1, self.hidden_size).T

        return (
            K[:, :-1], K[:, -1],\
            Q[:, :-1], Q[:, -1],\
            V[:, :-1], V[:, -1],\
            O[:, :-1], O[:, -1]
        )

