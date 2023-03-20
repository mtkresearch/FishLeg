from typing import Tuple, Callable, Any, Union, List
from collections.abc import Mapping
import torch
import torch.nn as nn
import copy
import numpy as np
from torch.nn import init
from torch.optim import Optimizer, Adam

import sys

try:
    from torch.optim.optimizer import _use_grad_for_differentiable
except ImportError:
    from .utils import _use_grad_for_differentiable

from .utils import recursive_setattr, recursive_getattr, update_dict
from transformers import get_scheduler

from .fishleg_layers import FishLinear, FishConv2d
from .fishleg_likelihood import FishLikelihood

__all__ = [
    "FishLeg",
]


class FishLeg(Optimizer):
    r"""Implement FishLeg algorithm.

    As described in ...

    :param torch.nn.Module model: a pytorch neural network module,
                can be nested in a tree structure
    :param Callable[[nn.Module, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]] draw:
                Sampling function that takes a model :math:`f` and input data :math:`\mathbf X`,
                and returns :math:`(\mathbf X, \mathbf y)`,
                where :math:`\mathbf y` is sampled from
                the conditional distribution :math:`p(\mathbf y|f(\mathbf X))`
    :param Callable[[nn.Module, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor] nll:
                A function that takes a model and data, and evaluate the negative
                log-likelihood.
    :param torch.utiles.data.DataLoader aux_dataloader:
                A function that takes a batch size as input and output dataset
                with corresponding size.
    :param FishLikelihood likelihood : a FishLeg likelihood, with Qv method if
                any parameters are learnable.
    :param float lr: Learning rate,
                for the parameters of the input model using FishLeg (default: 1e-2)

    :param int update_aux_every: Number of iteration after which an auxiliary
                update is executed, if negative, then run -update_aux_every auxiliary
                updates in each outer iteration. (default: 10)
    :param float aux_lr: learning rate for the auxiliary parameters,
                using Adam (default: 1e-3)
    :param Tuple[float, float] aux_betas: Coefficients used for computing
                running averages of gradient and its square for auxiliary parameters
                (default: (0.9, 0.999))
    :param float aux_eps: Term added to the denominator to improve
                numerical stability for auxiliary parameters (default: 1e-8)
    :param int pre_aux_training: Number of auxiliary updates to make before
                any update of the original parameter. This process intends to approximate
                the correct Fisher Information matrix during initialization,
                which is espectially important for fine-tuning of models with pretraining
    :param bool differentiable: Whether the fused implementation (CUDA only) is used
    :param string initialization: Initialization of weights (default: uniform)
    :param float sgd_lr: Help specify initial scale of the inverse Fisher Information matrix
                approximation, :math:`\eta`. Make sure that

                .. math::
                        - \eta_{init} Q(\lambda) grad = - \eta_{sgd} grad

                is hold in the beginning of the optimization.
                And here :math:`\eta_{init}=\eta_{sgd}/\eta_{fl}`.
    :param str device: The device where calculations will be performed using PyTorch Tensors.

    Example:
        >>> aux_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=100)
        >>> train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=100)
        >>>
        >>> likelihood = FixedGaussianLikelihood(sigma=1.0)
        >>>
        >>> def nll(model, data_x, data_y):
        >>>     pred_y = model.forward(data_x)
        >>>     return likelihood.nll(data_y, pred_y)
        >>>
        >>> def draw(model, data_x):
        >>>     pred_y = model.forward(data_x)
        >>>     return likelihood.draw(pred_y)
        >>>
        >>> model = nn.Sequential(
        >>>     nn.Linear(2, 5),
        >>>     nn.ReLU(),
        >>>     nn.Linear(5, 1),
        >>> )
        >>>
        >>> opt = FishLeg(
        >>>     model,
        >>>     draw,
        >>>     nll,
        >>>     aux_loader
        >>> )
        >>>
        >>> for data_x, data_y in dataloader:
        >>>     opt.zero_grad()
        >>>     pred_y = model(data_x)
        >>>     loss = nn.MSELoss()(data_y, pred_y)
        >>>     loss.backward()
        >>>     opt.step()
        >>>     if iteration % 10 == 0:
        >>>         print(loss.detach())

    """

    def __init__(
        self,
        model: nn.Module,
        draw: Callable[[nn.Module, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        nll: Callable[[nn.Module, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
        aux_dataloader: torch.utils.data.DataLoader,
        likelihood: FishLikelihood,
        fish_lr: float = 5e-2,
        weight_decay: float = 1e-5,
        beta: float = 0.9,
        update_aux_every: int = 10,
        aux_lr: float = 1e-4,
        aux_betas: Tuple[float, float] = (0.9, 0.999),
        aux_eps: float = 1e-8,
        damping: float = 1e-5,
        pre_aux_training: int = 10,
        differentiable: bool = False,
        sgd_lr: float = 1e-2,
        initialization: str = "uniform",
        device: str = "cpu",
        num_steps = None,
        para_name: str = ''
    ) -> None:
        self.model = model
        self.sgd_lr = sgd_lr
        self.fish_lr = fish_lr
        self.device = device
        self.para_name = para_name
        self.initialization = initialization

        self.model = self.init_model_aux(model).to(device)
        self.likelihood = likelihood

        self.draw = draw
        self.nll = nll
        self.aux_dataloader = aux_dataloader
        
        param_groups = []
        for module_name, module in self.model.named_modules():
            if hasattr(module, "fishleg_aux"):
                model_module = recursive_getattr(self.model, module_name)
                params = {
                    name: param
                    for name, param in model_module.named_parameters()
                    if "fishleg_aux" not in name
                }
                g = {
                    "params": [params[name] for name in module.order],
                    "gradbar": [
                        torch.zeros_like(params[name]) for name in module.order
                    ],
                    "theta0": [params[name].data.clone() for name in module.order],
                    "grad": [torch.zeros_like(params[name]) for name in module.order],
                    "Qg": module.Qg,
                    "order": module.order,
                    "name": module_name,
                    "module": module,
                }
                param_groups.append(g)

                # Register hooks on trainable modules
                module.register_forward_pre_hook(self._save_input)
                module.register_full_backward_hook(self._save_grad_output)

        likelihood_params = self.likelihood.get_parameters()
        if len(likelihood_params) > 0:
            self.likelihood.init_aux(init_scale=np.sqrt(self.sgd_lr / self.fish_lr))
            g = {
                "params": likelihood_params,
                "gradbar": [torch.zeros_like(p) for p in likelihood_params],
                "grad": [torch.zeros_like(p) for p in likelihood_params],
                "Qv": self.likelihood.Qv,
                "order": self.likelihood.order,
                "name": "likelihood",
            }
            param_groups.append(g)

        defaults = dict(lr=aux_lr, fish_lr=fish_lr, differentiable=differentiable)
        super(FishLeg, self).__init__(param_groups, defaults)

        self.aux_param = [
            param
            for name, param in self.model.named_parameters()
            if "fishleg_aux" in name
        ]

        if len(likelihood_params) > 0:
            self.aux_param.extend(self.likelihood.get_aux_parameters())

        self.aux_opt = Adam(
            self.aux_param,
            lr=aux_lr,
            betas=aux_betas,
            eps=aux_eps,
            weight_decay=weight_decay,
        )

        if num_steps is not None:
            self.aux_scheduler = get_scheduler(
                name='linear', optimizer=self.aux_opt,
                num_warmup_steps=0,
                num_training_steps=num_steps
            )
        else: 
            self.aux_scheduler = None
        
        self.update_aux_every = update_aux_every
        self.aux_lr = aux_lr
        self.aux_betas = aux_betas
        self.aux_eps = aux_eps
        self.damping = damping
        self.weight_decay = weight_decay
        self.beta = beta
        self.pre_aux_training = pre_aux_training
        self.step_t = 0
        self.store_g = True

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
                if self.para_name in name:
                    if isinstance(module, nn.Linear):
                        if any([p.requires_grad for p in module.parameters()]):
                            replace = FishLinear(
                                module.in_features,
                                module.out_features,
                                module.bias is not None,
                                init_scale=np.sqrt(self.sgd_lr / self.fish_lr),
                                device=self.device,
                            )
                            replace = update_dict(replace, module)
                            if self.initialization == 'normal':
                                init.normal_(replace.weight,0,1/np.sqrt(module.in_features)) 
                            recursive_setattr(model, name, replace)
            except KeyError:
                pass

        # TODO: The above may not be a very "correct" way to do this, so please feel free to change, for example, we may want to check the name is in the fish_layer keys before attempting what is in the try statement.
        # TODO: Error checking to check that model includes some auxiliary arguments.

        return model

    def _prepare_input(
        self, data: Union[torch.Tensor, Any]
    ) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=self.device)
            return data.to(**kwargs)
        return data

    def update_aux(self) -> None:
        """Performs a single auxliarary parameter update
        using Adam. By minimizing the following objective:

        .. math::
            nll(model, \\theta + \epsilon Q(\lambda)g) + nll(model, \\theta - \epsilon Q(\lambda)g) - 2\epsilon^2g^T Q(\lambda)g

        where :math:`\\theta` is the parameters of model, :math:`\lambda` is the
        auxliarary parameters.
        """

        data = next(iter(self.aux_dataloader))
        data = self._prepare_input(data)

        self.aux_opt.zero_grad()
        with torch.no_grad():
            self.store_g = False
            samples = self.draw(self.model, data)
            self.store_g = True

        g2 = 0.0
        for group in self.param_groups:
            name = group["name"]
            for i, (p, para_name) in enumerate(zip(group["params"], group["order"])):
                grad = p.grad.data
                g2 = g2 + torch.sum(grad * grad)
                group["grad"][i].copy_(grad)

        g_norm = torch.sqrt(g2)

        self.zero_grad()
        # How to better implement this?
        # The hook is not updated here, locally, only the gradient to the parameters g.grad is updated
        self.store_g = False
        self.nll(self.model, samples).backward()
        self.store_g = True

        reg_term = 0.0
        quad_term = 0.0
        linear_term = 0.0

        for group in self.param_groups:
            name = group["name"]
            
            grad_norm = [grad/g_norm for grad in group['grad']]
            qg = group["Qg"]()

            for p, g, d_p in zip(
                group['params'], grad_norm, qg
            ):

            for p, g, d_p in zip(group["params"], group["grad"], qg):
                grad = p.grad.data
                quad_term = quad_term + torch.sum(grad * d_p)
                linear_term = linear_term + torch.sum(g * d_p)
                reg_term = reg_term + self.damping * torch.sum(d_p * d_p)

        quad_term = quad_term**2

        aux_loss = 0.5 * (reg_term + quad_term) - linear_term
        aux_loss.backward()
        self.aux_loss = aux_loss.item()
        self.aux_opt.step()
        if self.aux_scheduler is not None:
            self.aux_scheduler.step()
        return aux_loss, linear_term, quad_term, reg_term, g2
    
    def step(self) -> None:
        """Performes a single optimization step of FishLeg."""

        if self.step_t == 0:
            self.step_t += 1
            print("== pretraining==")
            aux_losses = []
            aux = 0
            for pre in range(self.pre_aux_training):
                self.zero_grad()
                data = next(iter(self.aux_dataloader))
                data = self._prepare_input(data)
                self.nll(self.model, data).backward()
                aux_loss, linear_term, quad_term, reg_term, g2 = self.update_aux()
                aux += aux_loss

                if pre % 10 == 0 and pre != 0:
                    print('aux_loss: {:.4f}, \t linear: {:.4f}, quad: {:.4f}, reg: {:.4f} g2: {:.4}'.format(
                            aux/10, linear_term, quad_term, reg_term, g2
                         ))
                    aux = 0
                aux_losses.append(aux_loss.detach().cpu().numpy())
            return aux_losses

        if self.update_aux_every > 0:
            if self.step_t % self.update_aux_every == 0:
                aux_loss, linear_term, quad_term, reg_term, g2 = self.update_aux()
        elif self.update_aux_every < 0:
            for _ in range(-self.update_aux_every):
                self.update_aux()

        self.step_t += 1

        for group in self.param_groups:
            name = group["name"]
            with torch.no_grad():
                nat_grad = group["Qg"]()

                for p, d_p, gbar, p0 in zip(
                    group["params"], nat_grad, group["gradbar"], group["theta0"]
                ):
                    gbar.copy_(self.beta * gbar + (1.0 - self.beta) * d_p)
                    delta = gbar.add(p, alpha=self.weight_decay/self.fish_lr)
                    p.add_(delta, alpha=-self.fish_lr) 
                    
    @torch.no_grad()
    def _save_input(
        self,
        module: torch.nn.Module,
        input_: list[torch.Tensor],
    ) -> None:
        if not module.training:
            return
        if self.store_g:
            module.save_layer_input(input_)

    @torch.no_grad()
    def _save_grad_output(
        self,
        module: torch.nn.Module,
        grad_input: Union[tuple[torch.Tensor, ...], torch.Tensor],
        grad_output: Union[tuple[torch.Tensor, ...], torch.Tensor],
    ) -> None:
        if not module.training:
            return
        if self.store_g:
            module.save_layer_grad_output(grad_output)
