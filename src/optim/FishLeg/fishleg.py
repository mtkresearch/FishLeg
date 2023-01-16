from typing import Optional, Tuple
import torch
import torch.nn as nn
import copy
from torch.optim import Optimizer, Adam

from .fishleg_layers import FISH_LAYERS


class FishLeg(Optimizer):
    """Implement FishLeg algorithm.

    :param torch.nn.Module model: a pytorch neural network module, 
                can be nested in a tree structure
    :param float lr: learning rate,
                for the parameters of the input model using FishLeg (default: 1e-2)
    :param float eps: a small scalar, to evaluate the auxiliary loss 
                in the direction of gradient of model parameters (default: 1e-4)
    :param int aux_K: number of sample to evaluate the entropy (default: 5)

    :param int update_aux_every: number of iteration after which an auxiliary 
                update is executed, if negative, then run -update_aux_every auxiliary 
                updates in each outer iteration. (default: -3)
    :param float aux_lr: learning rate for the auxiliary parameters, 
                using Adam (default: 1e-3)
    :param Tuple[float, float] aux_betas: coefficients used for computing
                running averages of gradient and its square for auxiliary parameters
                (default: (0.9, 0.999))
    :param float aux_eps: term added to the denominator to improve
                numerical stability for auxiliary parameters (default: 1e-8)
    """
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-2,
        eps: float = 1e-4,
        aux_K: int = 5,
        update_aux_every: int = -3,
        aux_lr: float = 1e-3,
        aux_betas: Tuple[float, float] = (0.9, 0.999),
        aux_eps: float = 1e-8
    ) -> None:
        self.model = model
        self.plus_model = copy.deepcopy(self.model)
        self.minus_model = copy.deepcopy(self.model)
        self.model = self.init_model_aux(model)

        # partition by modules
        self.aux_param = [
            param
            for name, param in self.model.named_parameters()
            if "fishleg_aux" in name
        ]

        param_groups = []
        for module_name, module in self.model.named_modules():
            if hasattr(module, "fishleg_aux"):
                params = {
                    name: param
                    for name, param in self.model._modules[
                        module_name
                    ].named_parameters()
                    if "fishleg_aux" not in name
                }
                g = {
                    "params": [params[name] for name in module.order],
                    "aux_params": {
                        name: param
                        for name, param in module.named_parameters()
                        if "fishleg_aux" in name
                    },
                    "Qv": module.Qv,
                    "order": module.order,
                    "name": module_name,
                }
                param_groups.append(g)
        # TODO: add param_group for modules without aux
        defaults = dict(lr=lr)

        super(FishLeg, self).__init__(param_groups, defaults)
        self.aux_opt = Adam(self.aux_param, lr=aux_lr, betas=aux_betas, eps=aux_eps)
        self.eps = eps
        self.aux_K = aux_K
        self.update_aux_every = update_aux_every 
        self.aux_lr = aux_lr
        self.aux_betas = aux_betas
        self.aux_eps = aux_eps
        self.step_t = 0

    def init_model_aux(self, model: nn.Module) -> nn.Module:
        """Given a model to optimize, parameters can be devided to
        
        #. those fixed as pre-trained.
        #. those required to optimize using FishLeg.

        Replace modules in the second group with FishLeg modules.

        Args:
            model (:class:`torch.nn.Module`, required): 
                A model containing modules to replace with FishLeg modules 
                containing extra functionality related to FishLeg algorithm.
        Returns:
            :class:`torch.nn.Module`, the replaced model.
        """
        for name, module in model.named_modules():
            try:
                replace = FISH_LAYERS[type(module).__name__.lower()](
                    module.in_features, 
                    module.out_features, 
                    module.bias is not None
                )
                replace = update_dict(replace, module)
                model._modules[name] = replace
            except KeyError:
                pass

        # TODO: The above may not be a very "correct" way to do this, so please feel free to change, for example, we may want to check the name is in the fish_layer keys before attempting what is in the try statement.
        # TODO: Error checking to check that model includes some auxiliary arguments.

        return model
    
    def update_aux(self) -> None:
        """Performs a single auxliarary parameter update
        using Adam. By minimizing the following objective:

        .. math::
            nll(model, \\theta + \epsilon Q(\lambda)g) + nll(model, \\theta - \epsilon Q(\lambda)g) - 2\epsilon^2g^T Q(\lambda)g 

        where :math:`\\theta` is the parameters of model, :math:`\lambda` is the
        auxliarary parameters.
        """
        self.aux_opt.zero_grad()
        with torch.no_grad():
            data = self.model.sample(self.aux_K)

        aux_loss = 0.0
        for group in self.param_groups:
            name = group["name"]

            grad = [p.grad.data for p in group["params"]]
            qg = group["Qv"](group["aux_params"], grad)

            for g, d_p, para_name in zip(grad, qg, group["order"]):
                param_plus = self.plus_model._modules[name]._parameters[para_name]
                param_plus = param_plus.detach()
                param_minus = self.minus_model._modules[name]._parameters[para_name]
                param_minus = param_minus.detach()

                param_plus.add_(d_p, alpha=self.eps)
                param_minus.add_(d_p, alpha=-self.eps)
                aux_loss -= 2 * torch.sum(g * d_p)

        h_plus = self.plus_model.nll(data)
        h_minus = self.minus_model.nll(data)
        aux_loss += (h_plus + h_minus) / (self.eps**2)

        aux_loss.backward()
        self.aux_opt.step()

        for group in self.param_groups:
            for p, para_name in zip(group["params"], group["order"]):
                self.plus_model._modules[name]._parameters[para_name].data = p.data
                self.minus_model._modules[name]._parameters[para_name].data = p.data

    def step(self) -> None:
        """Performes a single optimization step of FishLeg.
        """
        self.step_t += 1

        if self.update_aux_every > 0:
            if self.step_t % self.update_aux_every == 0:
                self.update_aux()
        elif self.update_aux_every < 0:
            for _ in range(-self.update_aux_every):
                self.update_aux()

        for group in self.param_groups:
            lr = group["lr"]
            order = group["order"]
            name = group["name"]

            if "aux_params" in group.keys():
                grad = grad = [p.grad.data for p in group["params"]]
                qg = group["Qv"](group["aux_params"], grad)

                for p, d_p, para_name in zip(group["params"], qg, order):
                    p.data.add_(d_p, alpha=-lr)
                    self.plus_model._modules[name]._parameters[para_name].data = p.data
                    self.minus_model._modules[name]._parameters[para_name].data = p.data


def update_dict(replace: nn.Module, module: nn.Module) -> nn.Module:
        replace_dict = replace.state_dict()
        pretrained_dict = {
            k: v for k, v in module.state_dict().items() if k in replace_dict
        }
        replace_dict.update(pretrained_dict)
        replace.load_state_dict(replace_dict)
        return replace