from typing import Tuple, Callable, Any, Union, List
from collections.abc import Mapping
import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
from torch.optim import Optimizer, Adam
import sys
import regex as re
import csv
import os
from transformers.models.bert.modeling_bert import BertAttention

from .utils import (
    recursive_setattr,
    recursive_getattr,
    update_dict,
    get_named_layers_by_regex,
    NamedLayer,
)
from transformers import get_scheduler

from .layers import *
from .fishleg_layers import *
from .fishleg_likelihood import FishLikelihood

__all__ = [
    "FishLeg",
]


class FishLeg(Optimizer):
    r"""Implement FishLeg algorithm.

    As described in https://openreview.net/forum?id=c9lAOPvQHS.

    :param torch.nn.Module model: a pytorch neural network module,
                can be nested in a tree structure
    :param torch.utiles.data.DataLoader aux_dataloader:
                A function that takes a batch size as input and output dataset
                with corresponding size.
    :param FishLikelihood likelihood : a FishLeg likelihood, with Qv method if
                any parameters are learnable.
    :param float fish_lr: Learning rate,
                for the parameters of the input model using FishLeg (default: 1e-2)
    :param float damping: Static damping applied to Fisher matrix, :math:`\gamma`,
                for stability when FIM becomes near-singular. (default: 5e-1)
    :param float weight_decay: L2 penalty on weights (default: 1e-5)
    :param float beta: coefficient for running averages of gradient (default: 0.9)
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
    :param bool normalize_aux: Whether to use normalize_aux on gradients when calculating
                the auxiliary loss, this is important to learn about curvature even when
                gradients are small (default: False)
    :param List module_names: A List of module names wished to be optimized/pruned by FishLeg.
                (default: [], meaning all modules optimized/pruned by FishLeg)
    :param string initialization: Initialization of weights (default: uniform)
    :param int warmup_steps: If warmup_steps is zero, the default SGD warmup will be used, where Q is
                initialized as a scaled identity matrix. If warmup is positive, the diagonal
                of Q will be initialized as :math:`\frac{1}{g^2 + \gamma}`; and in this case,
                warmup_data and warmup_loss should be provided for sampling of gradients.
    :param float fish_scale: Help specify initial scale of the inverse Fisher Information matrix
                approximation. If using SGD warmup we suggest, :math:`\eta=\gamma^{-1}`. If
                warmup is positive, scale should be 1. (default: 1)
    :param str device: The device where calculations will be performed using PyTorch Tensors.

    Example:
        >>> aux_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=100)
        >>> train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=100)
        >>>
        >>> likelihood = FixedGaussianLikelihood(sigma=1.0)
        >>>
        >>>
        >>>
        >>> model = nn.Sequential(
        >>>     nn.Linear(2, 5),
        >>>     nn.ReLU(),
        >>>     nn.Linear(5, 1),
        >>> )
        >>>
        >>> opt = FishLeg(
        >>>     model,
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
        aux_dataloader: torch.utils.data.DataLoader,
        likelihood: FishLikelihood = None,  # Is this optional??
        lr: float = 5e-2,
        damping: float = 5e-1,
        weight_decay: float = 1e-5,
        beta: float = 0.9,
        update_aux_every: int = 10,
        aux_lr: float = 1e-4,
        aux_betas: Tuple[float, float] = (0.9, 0.999),
        aux_eps: float = 1e-8,
        normalize_aux: bool = False,
        module_names: str = "__ALL__",  # This should be __ALL__ or something similar
        initialization: str = "uniform",
        fish_scale: float = 1.0,
        grad_clip: bool = False,
        warmup_steps: int = 0,
        device: str = "cpu",
        verbose: bool = False,
    ) -> None:
        self.model = model

        self.aux_dataloader = aux_dataloader
        self.likelihood = likelihood

        self.lr = lr
        self.beta = beta

        self.weight_decay = weight_decay

        self.aux_lr = aux_lr
        self.fish_scale = fish_scale
        self.damping = damping
        self.update_aux_every = update_aux_every

        self.warmup_steps = warmup_steps

        self.grad_clip = grad_clip
        self.initialization = initialization
        self.normalize_aux = normalize_aux

        self.device = device
        self.verbose = verbose

        self.model, param_groups = self.init_model_aux(
            model, module_names=module_names, config=config
        )
        defaults = dict(aux_lr=aux_lr, lr=lr)
        super(FishLeg, self).__init__(param_groups, defaults)

        aux_param = [
            param for name, param in model.named_parameters() if "fishleg_aux" in name
        ]
        if self.likelihood is not None:
            if len(self.likelihood.get_parameters()) > 0:
                aux_param.extend(self.likelihood.get_aux_parameters())

        self.aux_opt = Adam(
            aux_param,
            lr=aux_lr,
            betas=aux_betas,
            eps=aux_eps,
            weight_decay=0,  # weight_decay need to be fixed to zero
        )

        if self.warmup_steps > 0:
            self.warmup_aux(fish_scale=fish_scale)

        self.step_t = 0
        self.store_g = True

    # I think we should demand a Fisher model be constructed outside the optimizer before being passed in.
    def init_model_aux(
        self,
        model: nn.Module,
        module_names: str,
    ) -> Union[nn.Module, List]:
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
        if any(["weight" in m for m in module_names]):
            raise TypeError(
                f"Parameters to be optimized in FishLeg are considered together in one module, and cannot be optimized individually"
            )

        # Add auxiliary parameters
        if module_names == "__ALL__":
            named_layers = [
                NamedLayer(name, layer) for name, layer in model.named_modules()
            ]

        else:
            named_layers = get_named_layers_by_regex(model, module_names)

        replaced_layers = []
        for named_layer in named_layers:
            module = named_layer.layer
            module_name = named_layer.layer_name

            for layer in replaced_layers:
                if re.match(layer.layer_name, module_name):
                    inner_name = module_name[len(layer.layer_name) + 1 :]
                    if any(
                        re.match(inner_name, param_name)
                        for param_name in layer.layer.order
                    ):
                        continue

            if isinstance(module, nn.Linear):
                replace = FishLinear(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    device=self.device,
                )

                # if self.initialization == "normal":
                #     init.normal_(replace.weight, 0, 1 / np.sqrt(module.in_features))
                # elif self.initialization == "zero":  # fill with zeros for adapters
                #     module.weight.data.zero_()
                #     module.bias.data.zero_()
            
            elif isinstance(module, nn.Embedding):
                replace = FishEmbedding(
                    num_embeddings=module.num_embeddings,
                    embedding_dim=module.embedding_dim,
                    padding_idx=module.padding_idx,
                    max_norm=module.max_norm,
                    norm_type=module.norm_type,
                    scale_grad_by_freq=module.scale_grad_by_freq,
                    sparse=module.sparse,
                    device=self.device,
                )
            
            elif isinstance(module, nn.Conv2d):
                replace = FishConv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=(module.bias is not None),
                    padding_mode=module.padding_mode,
                    device=self.device
                    # TODO: deal with dtype and device?
                )
            
            elif isinstance(module, BertAttention):
                config = model.config
                replace = FishBertAttention(config, device=self.device)
            
            elif isinstance(module, nn.BatchNorm2d):
                replace = FishBatchNorm2d(
                    num_features=module.num_features,
                    eps=module.eps,
                    momentum=module.momentum,
                    affine=module.affine,
                    track_running_stats=module.track_running_stats,
                    init_scale=self.fish_scale,
                    device=self.device,
                )
            
            elif isinstance(module, nn.LayerNorm):
                replace = FishLayerNorm(
                    normalized_shape=module.normalized_shape,
                    eps=module.eps,
                    elementwise_affine=module.elementwise_affine,
                    init_scale=self.fish_scale,
                    device=self.device,
                )
            else:
                continue
                
            replace = update_dict(replace, module)
            recursive_setattr(model, module_name, replace)
            replaced_layers.append(NamedLayer(module_name, replace))
            if self.verbose:
                print("Replaced with FishLeg Layer: \t ", module_name)

        # Define each modules
        param_groups = []
        for named_layer in replaced_layers:
            module = named_layer.layer
            params = {
                name: param
                for name, param in module.named_parameters()
                if "fishleg_aux" not in name
            }

            g = {
                "params": [params[name] for name in module.order],
                "gradbar": [torch.zeros_like(params[name]) for name in module.order],
                "grad": [torch.zeros_like(params[name]) for name in module.order],
                "u": [torch.zeros_like(params[name]) for name in module.order],
                "Qv": module.Qv,
                "order": module.order,
                "name": named_layer.layer_name,
                "module": module,
            }
            param_groups.append(g)

        if self.likelihood is not None:
            likelihood_params = self.likelihood.get_parameters()
            if len(likelihood_params) > 0:
                self.likelihood.init_aux(init_scale=self.fish_scale)
                g = {
                    "params": likelihood_params,
                    "gradbar": [torch.zeros_like(p) for p in likelihood_params],
                    "theta0": [p.clone() for p in likelihood_params],
                    "grad": [torch.zeros_like(p) for p in likelihood_params],
                    "Qv": self.likelihood.Qv,
                    "order": self.likelihood.order,
                    "name": "likelihood",
                }
                param_groups.append(g)

        # TODO: The above may not be a very "correct" way to do this, so please feel free to change, for example, we may want to check the name is in the fish_layer keys before attempting what is in the try statement.
        # TODO: Error checking to check that model includes some auxiliary arguments.

        return model, param_groups

    def warmup_aux(
        self,
        dataloader: torch.utils.data.DataLoader,
        loss: Callable[
            [nn.Module, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor
        ] = None,
    ) -> None:
        """
        Warm up auxilirary parameters with approxiamte Adam
        """
        for _ in range(0, self.warmup_steps, dataloader.batch_size):
            data = self._prepare_input(next(iter(dataloader)))
            loss(self.model, data).backward()
            self._store_u(transform=lambda x: x * x)

        for group in self.param_groups:
            ds = []  # What is this?
            for i, g2 in enumerate(group["grad"]):
                g2_avg = g2 / int(self.warmup_steps / dataloader.batch_size)
                ds.append(
                    self.fish_scale / torch.sqrt(torch.sqrt(g2_avg + self.damping))
                )

            group["module"].warmup(ds, init_scale=self.fish_scale)

    def parameters(
        self,
    ):
        for group in self.param_groups:
            for param in group["params"]:
                yield param

    def pretrain_fish(
        self,
        dataloader: torch.utils.data.DataLoader,
        loss: Callable[[nn.Module, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
        iterations: int = 10000,
    ) -> List:
        aux, checks = 0, 0

        for pre in range(iterations):
            self.zero_grad()

            batch = next(iter(dataloader))
            batch = self._prepare_input(batch)

            loss(self.model, batch).backward()

            if self.grad_clip:
                nn.utils.clip_grad_norm_(
                    self.parameters(),
                    1,
                )

            self._store_u(new=True)

            self.zero_grad()

            batch = next(iter(dataloader))
            batch = self._prepare_input(batch)

            loss(self.model, batch).backward()

            self._store_u(alpha=-1.0)

            info = self.update_aux()

        return info

    # This should be external
    def _store_u(
        self,
        transform: Callable = lambda x: x,
        alpha: float = 1.0,
        new: bool = False,
    ):
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                grad = transform(p.grad.data)

                if not new:
                    group["grad"][i].add_(grad, alpha=alpha)
                else:
                    group["grad"][i].copy_(grad)
                group["u"][i].copy_(grad)

    # This should be external and done before passing into the optimizer
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
        """
        Performs a single auxliarary parameter update
        using Adam. By minimizing the following objective:

        .. math::
            nll(model, \\theta + \epsilon Q(\lambda)g) + nll(model, \\theta - \epsilon Q(\lambda)g) - 2\epsilon^2g^T Q(\lambda)g

        where :math:`\\theta` is the parameters of model, :math:`\lambda` is the
        auxliarary parameters.
        """
        self.store_g = False
        data = next(iter(self.aux_dataloader))
        data = self._prepare_input(data)

        self.aux_opt.zero_grad()
        with torch.no_grad():
            data_x, data_y = data
            samples_y = self.model(data_x)
            samples = (data_x, self.likelihood.draw(samples_y))

        g2 = 0.0
        for group in self.param_groups:
            for grad in group["grad"]:
                g2 = g2 + torch.sum(grad * grad)
        g_norm = torch.sqrt(g2)

        self.zero_grad()
        self.likelihood.nll(samples_y, samples[1]).backward()
        
        if self.grad_clip:
            nn.utils.clip_grad_norm_(
                self.parameters(),
                1,
            )

        reg_term = 0.0
        quad_term = 0.0
        linear_term = 0.0
        align = 0.0

        for i, group in enumerate(self.param_groups):
            qg = group["Qv"](group["grad"])

            for p, g, d_p in zip(group["params"], group["grad"], qg):
                grad = p.grad.data
            
                quad_term = quad_term + torch.sum(grad * d_p)
                linear_term = linear_term + torch.sum(g * d_p)
                reg_term = reg_term + self.damping * torch.sum(d_p * d_p)
                align = align + torch.sum(grad * g)

        check = quad_term * align + self.damping * linear_term - g2
        quad_term = quad_term**2

        aux_loss = 0.5 * (reg_term + quad_term) - linear_term

        if self.normalize_aux:
            aux_loss = aux_loss / g2
            check = check / g2

        aux_loss.backward()
        self.aux_loss = aux_loss.item()
        self.aux_opt.step()

        if train:
            aux_loss.backward()
            self.aux_loss = aux_loss.item()
            self.aux_opt.step()
            if self.aux_scheduler is not None:
                self.aux_scheduler.step()

        self.store_g = True
        return aux_loss, check, linear_term, quad_term, reg_term, g2

    def step(self, closure=None) -> None:
        """Performes a single optimization step of FishLeg."""
        self.updated = False

        if self.update_aux_every > 0:
            if self.step_t % self.update_aux_every == 0:
                self._store_u(new=True)
            if self.step_t % self.update_aux_every == 1:
                # once for difference of gradients
                self._store_u(alpha=-1.0)
                info = self.update_aux()
                info = [e.detach().cpu().numpy() for e in info]

                if self.verbose == True:
                    if self.step_t % 200 == 1:
                        print(
                            "iter:{:d}, lr:{:.2f} \tauxloss:{:.2f} \tcheck:{:.2f} \tlinear:{:.2f} \tquad:{:.2f} \treg:{:.2f} \tg2:{:.2f}".format(
                                self.step_t, self.fish_lr, *info
                            )
                        )
                self.updated = True
            if self.step_t % self.update_aux_every == 2:
                # once for gradient
                self._store_u(new=True)
                self.update_aux()
                self.updated = True
        elif self.update_aux_every < 0:
            self._store_u(new=True)
            for _ in range(-self.update_aux_every):
                self.update_aux()
            self.updated = True

        self.step_t += 1

        for group in self.param_groups:
            name = group["name"]
            with torch.no_grad():
                nat_grad = group["Qv"](
                    group["u"]
                    if self.updated
                    else [p.grad.data for p in group["params"]]
                )
                for p, d_p, gbar in zip(group["params"], nat_grad, group["gradbar"]):
                    gbar.copy_(self.beta * gbar + (1.0 - self.beta) * d_p)
                    delta = gbar.add(p, alpha=self.weight_decay)
                    p.add_(delta, alpha=-self.fish_lr)

    @torch.no_grad()
    def _save_input(
        self,
        module: torch.nn.Module,
        input_: List[torch.Tensor],
    ) -> None:
        if not module.training:
            return
        if self.store_g:
            module.save_layer_input(input_)

    @torch.no_grad()
    def _save_grad_output(
        self,
        module: torch.nn.Module,
        grad_input: Union[Tuple[torch.Tensor, ...], torch.Tensor],
        grad_output: Union[Tuple[torch.Tensor, ...], torch.Tensor],
    ) -> None:
        if not module.training:
            return
        if self.store_g:
            module.save_layer_grad_output(grad_output)
