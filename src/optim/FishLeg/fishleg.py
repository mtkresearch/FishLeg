import torch
import torch.nn as nn
import copy
from torch.optim import Optimizer, Adam

from .fishleg_layers import FISH_LAYERS


class FishLeg(Optimizer):
    def __init__(
        self,
        model,
        lr=1e-2,
        eps=1e-4,
        aux_K=5,
        update_aux_every=-3,
        aux_scale_init=1,
        aux_lr=1e-3,
        aux_betas=(0.9, 0.999),
        aux_eps=1e-8,
    ):
        self.model = model
        self.plus_model = copy.deepcopy(self.model)
        self.minus_model = copy.deepcopy(self.model)
        self.model = self.__init_model_aux(model)

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
        self.update_aux_every = update_aux_every  # if negative, then run -update_aux_every aux updates in each outer iteration
        self.aux_scale_init = aux_scale_init
        self.aux_lr = aux_lr
        self.aux_betas = aux_betas
        self.aux_eps = aux_eps
        self.step_t = 0

    def update_dict(self, replace, module):
        replace_dict = replace.state_dict()
        pretrained_dict = {
            k: v for k, v in module.state_dict().items() if k in replace_dict
        }
        replace_dict.update(pretrained_dict)
        replace.load_state_dict(replace_dict)
        return replace

    def __init_model_aux(self, model):
        for name, module in model.named_modules():
            try:
                replace = FISH_LAYERS[type(module).__name__.lower()](
                    module.in_features, module.out_features, module.bias is not None
                )
                replace = self.update_dict(replace, module)
                model._modules[name] = replace
            except KeyError:
                pass

        # TODO: The above may not be a very "correct" way to do this, so please feel free to change, for example, we may want to check the name is in the fish_layer keys before attempting what is in the try statement.
        # TODO: Error checking to check that model includes some auxiliary arguments.

        return model

    def update_aux(self):
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

    def step(self):
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
