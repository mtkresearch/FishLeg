import torch.nn as nn
from torch import Tensor
from torch.nn import ParameterDict, Parameter

from typing import Any, List, Dict, Tuple

from abc import abstractmethod

class FishAuxParameter(Parameter):
    pass

class FishModule(nn.Module):
    """
    Base class for all neural network modules in FishLeg to

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

    def aux_parameters(self, recurse: bool = True):
        for param in self.parameters(recurse=recurse):
            if isinstance(param, FishAuxParameter):
                yield param
    
    def named_aux_parameters(self, prefix: str = '',
            recurse: bool = True,
            remove_duplicate: bool = True):
        for name, param in self.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate):
            if isinstance(param, FishAuxParameter):
                yield name, param
                                            
    def not_aux_parameters(self, recurse: bool = True):
        for param in self.parameters(recurse=recurse):
            if not isinstance(param, FishAuxParameter):
                yield param
    
    def named_not_aux_parameters(self, prefix: str = '',
            recurse: bool = True,
            remove_duplicate: bool = True):
        for name, param in self.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate):
            if not isinstance(param, FishAuxParameter):
                yield name, param

    @abstractmethod
    def Qv(self, v: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        """:math:`Q(\lambda)` is a positive definite matrix which will effectively
        estimate the inverse damped Fisher Information Matrix. Appropriate choices
        for :math:`Q` should take into account the architecture of the model/module.
        It is usually parameterized as a positive definite Kronecker-factored
        block-diagonal matrix, with block sizes reflecting the layer structure of
        the neural networks.

        Args:
            v: (Tuple[Tensor, ...], required): Values of the original parameters,
                    in an order that align with `self.order`, to multiply with
                    :math:`Q(\lambda)`.
        Returns:
            Tuple[Tensor, ...]: The calculated :math:`Q(\lambda)v` products,
                    in same order with `self.order`.

        """
        raise NotImplementedError(f'Module is missing the required "Qv" function')
