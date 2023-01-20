import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import ParameterDict, Parameter
from abc import abstractmethod

from typing import Any, List, Dict, Tuple


class FishModule(nn.Module):
    """Base class for all neural network modules in FishLeg to 

    #. Initialize auxiliary parameters, :math:`\lambda` and its forms, :math:`Q(\lambda)`.
    #. Specify quick calculation of :math:`Q(\lambda)v` products.
    
    :param torch.nn.ParameterDict fishleg_aux: auxiliary parameters 
                with their initialization, including an additional parameter, scale, 
                :math:`\eta`. Make sure that 

                .. math::
                        - \eta Q(\lambda) grad = - \eta_{adam} grad

                is hold in the beginning of the optimization
    :param List order: specify a name order of original parameter

    """


    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(FishModule, self).__init__(*args, **kwargs)
        self.__setattr__('fishleg_aux', ParameterDict())
        self.__setattr__('order', List)

    @property
    def name(self) -> str:
        return self._layer_name

    def cuda(self, device) -> None:
        super.cuda(device)
        for p in self.fishleg_aux.values:
            p.to(device)

    @abstractmethod
    def Qv(aux: Dict, v: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        """ :math:`Q(\lambda)` is a positive definite matrix which will effectively 
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
        Returns:
            Tuple[Tensor, ...]: The calculated :math:`Q(\lambda)v` products, 
                    in same order with `self.order`.

        """
        raise NotImplementedError(f"Module is missing the required \"Qv\" function")


class FishLinear(nn.Linear, FishModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device = None,
        dtype = None,
    ) -> None:
        super(FishLinear, self).__init__(
            in_features, out_features, bias, device=device, dtype=dtype
        )
        self._layer_name = "Linear"
        self.fishleg_aux = ParameterDict(
            {
                "scale": Parameter(torch.ones(size=(1,))),
                "L": Parameter(torch.eye(in_features + 1)),
                "R": Parameter(torch.eye(out_features)),
            }
        )
        self.order = ["weight", "bias"]

    @staticmethod
    def Qv(aux: Dict, v: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        '''For fully-connected layers, the default structure of :math:`Q` as a
        block-diaglonal matrix is,

        .. math::
                    Q_l = (R_lR_l^T \otimes L_lL_l^T)

        where :math:`l` denotes the l-th layer. The matrix :math:`R_l` has size 
        :math:`(N_{l-1} + 1) \\times (N_{l-1} + 1)` while the matrix :math:`L_l` has 
        size :math:`N_l \\times N_l`. The auxiliarary parameters :math:`\lambda` 
        are represented by the matrices :math:`L_l, R_l`.
        
        '''
        L, R = aux["fishleg_aux.L"], aux["fishleg_aux.R"]
        u = torch.cat([v[0], v[1][:, None]], dim=-1)
        z = aux["fishleg_aux.scale"] * torch.linalg.multi_dot((R, R.T, u, L, L.T))
        return (z[:, :-1], z[:, -1])


FISH_LAYERS = {
    "linear": FishLinear
}  # Perhaps this would be better constructed inside the __init__.py file?
