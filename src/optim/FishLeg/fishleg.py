from typing import Tuple, Callable, Any, Union, List
from collections.abc import Mapping
import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
from torch.optim import Optimizer, Adam
import sys
import regex as re
from functools import partial
from transformers.models.bert.modeling_bert import BertAttention

from .utils import recursive_setattr, recursive_getattr, update_dict
from transformers import get_scheduler

from layers import *
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
    :param bool batch_speedup: Whether to use speed-up Qv product (default: False)
    :param bool full: Whether to use full inner and outer diagonal rescalling
                for block Kronecker approximation of Q. (default: True)
    :param bool normalization: Whether to use normalization on gradients when calculating
                the auxiliary loss, this is important to learn about curvature even when 
                gradients are small (default: False)
    :param bool fine_tune: Whether to use Fisher as preconditioner of pretrained tasks,
                and fine-tune on a downstream task. If True, Q will be fixed and
                continual learning will be performed (default: False)
    :param List module_names: A List of module names wished to be optimized/pruned by FishLeg. 
                (default: [], meaning all modules optimized/pruned by FishLeg)
    :param string initialization: Initialization of weights (default: uniform)
    :param int warmup: If warmup is zero, the default SGD warmup will be used, where Q is
                initialized as a scaled identity matrix. If warmup is positive, the diagonal
                of Q will be initialized as :math:`\frac{1}{g^2 + \gamma}`; and in this case,
                warmup_data and warmup_loss should be provided for sampling of gradients.
    :param float scale: Help specify initial scale of the inverse Fisher Information matrix
                approximation. If using SGD warmup we suggest, :math:`\eta=\gamma^{-1}`. If 
                warmup is positive, scale should be 1. (default: 1)
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
        draw: Callable[
                [nn.Module, torch.Tensor], 
                Tuple[torch.Tensor, torch.Tensor]
              ],
        nll: Callable[
                [nn.Module, 
                Tuple[torch.Tensor, torch.Tensor]
                ], torch.Tensor],
        aux_dataloader: torch.utils.data.DataLoader,
        likelihood: FishLikelihood,
        fish_lr: float = 5e-2,
        damping: float = 5e-1,
        weight_decay: float = 1e-5,
        beta: float = 0.9,
        update_aux_every: int = 10,
        aux_lr: float = 1e-4,
        aux_betas: Tuple[float, float] = (0.9, 0.999),
        aux_eps: float = 1e-8,
        num_steps = None,
        batch_speedup: bool = False,
        full: bool = True,
        normalization: bool = False,
        fine_tune: bool = False,
        module_names: List[str] = [],
        initialization: str = "uniform",
        scale: float = 1.,
        warmup: int = 0,
        warmup_data: torch.utils.data.DataLoader = None,
        warmup_loss: Callable = None
        device: str = "cpu",
        config = None
    ) -> None:

        self.model = model
        self.fish_lr = fish_lr
        self.device = device
        self.batch_speedup = batch_speedup
        self.full = full
        self.initialization = initialization
        self.normalization = normalization
        self.fine_tune = fine_tune
        self.likelihood = likelihood
        self.warmup = warmup
        self.scale = scale
        self.aux_lr = aux_lr
        self.damping = damping

        self.draw = draw
        self.nll = nll
        self.aux_dataloader = aux_dataloader

        self.model, param_groups = self.init_model_aux(
                                        model, module_names,config
                                        )
        self.model.to(device)
        defaults = dict(lr=aux_lr, fish_lr=fish_lr)
        super(FishLeg, self).__init__(param_groups, defaults)

        if self.warmup > 0:
            self.warmup_aux(warmup_data, warmup_loss, scale=scale)
        else:
            self.warmup_aux(scale=scale)

        aux_param = [
            param for name, param in model.named_parameters() if "fishleg_aux" in name
        ]
        if len(self.likelihood.get_parameters()) > 0:
            aux_param.extend(self.likelihood.get_aux_parameters())
        self.aux_opt = Adam(
            aux_param,
            lr=aux_lr,
            betas=aux_betas,
            eps=aux_eps,
            weight_decay=0,
            # weight_decay need to be fixed to zero
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
        self.weight_decay = weight_decay
        self.beta = beta
        self.step_t = 0
        self.store_g = True

    def init_model_aux(
        self,
        model: nn.Module,
        module_names: List[str],
        config = None
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
        if any(['weight' in m for m in module_names]):
            raise TypeError(f'Parameters to be optimized in FishLeg are considered together in one module, and cannot be optimized individually')
        # Add auxiliary parameters

        if len(module_names) == 0:
            fl_names = [name for name, _ in model.named_modules()]
        else:
            fl_names = []
            for name, _ in model.named_modules():
                if any([
                    re.match(fish_name + '$', name)
                    for fish_name in module_names
                ]):
                    fl_names.append(name)

        for module_name in fl_names:
            try: 
                module = recursive_getattr(model, module_name)
                if any([~p.requires_grad for p in module.parameters()]):
                    raise TypeError(f'There exists untrainable parameter in the module named {module_name}.')
                
                if isinstance(module, nn.Linear):
                    replace = FishLinear(
                                module.in_features,
                                module.out_features,
                                module.bias is not None,
                                device=self.device,
                            )
                    replace = update_dict(replace, module)
                    if self.initialization == "normal":
                        init.normal_(
                                    replace.weight, 0, 1 / np.sqrt(module.in_features)
                                )
                    elif self.initialization == "zero": # fill with zeros for adapters
                        module.weight.data.zero_()
                        module.bias.data.zero_()
                    recursive_setattr(model, module_name, replace)
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
                    replace = update_dict(replace, module)
                    recursive_setattr(model, module_name, replace)
                elif isinstance(module, BertAttention):
                    replace = FishBertAttention(config, device=self.device)
                    replace = update_dict(replace, module)
                    recursive_setattr(model, module_name, replace)
                else:
                    raise Warning(f'The FishLayer for module named {module_name} has not been implemented and hence skipped.')
            except:
                raise TypeError(f'The given model has no module named {module_name}.')
        
        # Define each modules
        param_groups = []
        for module_name in module_names:
            module = recursive_getattr(model, module_name)
            params = {
                    name: param
                    for name, param in module.named_parameters()
                    if "fishleg_aux" not in name
                }
            g = {
                    "params": [params[name] for name in module.order],
                    "gradbar": [
                        torch.zeros_like(params[name]) for name in module.order
                    ],
                    "grad": [torch.zeros_like(params[name]) for name in module.order],
                    "Qv": module.Qg if self.batch_speedup \
                            else partial(module.Qv, full=self.full),
                    "order": module.order,
                    "name": module_name,
                    "module": module,
                }
            if self.fine_tune:
                g["theta0"] = [params[name].clone() for name in module.order]
            param_groups.append(g)

            # Register hooks on trainable modules
            if self.batch_speedup:
                if isinstance(module, FishLinear):
                    module.register_forward_pre_hook(self._save_input)
                    module.register_full_backward_hook(self._save_grad_output)
                else:
                    raise NotImplementedError(f'Batch Speedup has not been implemented for module {module_name}')
            

        likelihood_params = self.likelihood.get_parameters()
        if len(likelihood_params) > 0:
            self.likelihood.init_aux(init_scale=self.scale)
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
        dataloader: torch.utils.data.DataLoader = None,
        loss: Callable[
            [nn.Module, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor
        ] = None,
        scale: float = 1.,
    ) -> None:
        """Warm up auxilirary parameters,
        if warmup is larger zero, follow approxiamte Adam,
        if warmup is zero, follow SGD
        """
        # Warm up following adam:
        if self.warmup > 0:
            for _ in range(0, self.warmup, dataloader.batch_size):

                data = self._prepare_input(next(iter(dataloader)))
                loss(self.model, data).backward()
                self._store_u(transform=lambda x: x * x)

        for group in self.param_groups:

            if self.warmup > 0:
                ds = []
                for i, g2 in enumerate(group["grad"]):
                    g2_avg = g2 / int(self.warmup/dataloader.batch_size)
                    ds.append(scale / torch.sqrt(g2_avg + self.damping))
                
                group["module"].warmup(
                    ds, batch_speedup=self.batch_speedup, init_scale=scale
                )
            else:
                group["module"].warmup(
                    batch_speedup=self.batch_speedup, init_scale=scale
                )

    def pretrain_fish(
        self,
        dataloader: torch.utils.data.DataLoader,
        loss: Callable[[
                nn.Module, 
                Tuple[torch.Tensor, 
                      torch.Tensor]
                ], torch.Tensor],
        iterations: int = 10000,
        difference: bool = False,
        verbose: bool = False,
        testloader: torch.utils.data.DataLoader = None,
        batch_size: int = 500,
        fisher: bool = True
    ) -> List:

        aux_losses = []
        aux, checks = 0, 0
        for pre in range(iterations):
            self.zero_grad()
            batch = next(iter(dataloader))
            batch = self._prepare_input(batch)
            loss(self.model, batch).backward()
            self._store_u(new=True)
            
            if difference:
                self.zero_grad()
                batch = next(iter(dataloader))
                batch = self._prepare_input(batch)
                loss(self.model, batch).backward()
                self._store_u(alpha=-1.0)

            info = self.update_aux(fisher=fisher)
            aux_loss = info[0].detach().cpu().numpy()
            check = info[1].detach().cpu().numpy()
            linear_term = info[2].detach().cpu().numpy()
            aux_losses.append(aux_loss + 0.5 * linear_term)

            if verbose:
                aux += aux_loss
                checks += check
                if pre % batch_size == 0:
                    info = [e.detach().cpu().numpy() for e in info]
                    if testloader is not None:
                        test_checks = 0
                        for _ in range(100):
                            self.zero_grad()
                            test_batch = next(iter(testloader))
                            test_batch = self._prepare_input(test_batch)
                            loss(self.model, test_batch).backward()
                            self._store_u(new=True)
                        
                            if difference:
                                self.zero_grad()
                                test_batch = next(iter(testloader))
                                test_batch = self._prepare_input(test_batch)
                                loss(self.model, test_batch).backward()
                                self._store_u(alpha=-1.0)

                            test_info = self.update_aux(fisher=fisher,train=False)
                            test_checks += test_info[1].detach().cpu().numpy()
                            
                        print("iter:{:d}, \t train:{:.2f} \t test:{:.2f} \t auxloss:{:.2f} check:{:.2f} \tlinear:{:.2f} \tquad:{:.2f} \treg:{:.2f} \tg2:{:.2f}".format(
                                pre, checks / batch_size, test_checks / 100, *info))
                    else:
                        print("iter:{:d}, \t train:{:.2f} \t auxloss:{:.2f} \check:{:.2f} \tlinear:{:.2f} \tquad:{:.2f} \treg:{:.2f} \tg2:{:.2f}".format(
                                pre, checks / batch_size, *info))
                    aux = 0
                    checks = 0

        return aux_losses

    def _store_u(
            self, transform: Callable = lambda x: x, alpha: float = 1.0, new: bool = False, add_noise: bool = False
    ):
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if add_noise:
                    scale = torch.sqrt(p.grad.data.var())
                    group["grad"][i].copy_(torch.randn_like(p.data)*scale)
                else:
                    grad = transform(p.grad.data)
                    if not new:
                        group["grad"][i].add_(grad, alpha=alpha)
                    else:
                        group["grad"][i].copy_(grad)

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

    def update_aux(self, train=True, fisher=True) -> None:
        """Performs a single auxliarary parameter update
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
            if fisher:
                samples = self.draw(self.model, data)
            else:
                samples = data
        if True:
            g2 = 0.0
            for group in self.param_groups:
                for grad in group["grad"]:
                    g2 = g2 + torch.sum(grad * grad)
            g_norm = torch.sqrt(g2)

        self.zero_grad()
        self.nll(self.model, samples).backward()

        reg_term = 0.0
        quad_term = 0.0
        linear_term = 0.0
        align = 0.0

        for i,group in enumerate(self.param_groups):
            qg = group["Qv"]() if self.batch_speedup else group["Qv"](group["grad"])

            for p, g, d_p in zip(
                group["params"], group["grad"], qg, group["order"]
            ):
                grad = p.grad.data
                quad_term = quad_term + torch.sum(grad * d_p)
                linear_term = linear_term + torch.sum(g * d_p)
                reg_term = reg_term + self.damping * torch.sum(d_p * d_p)
                align = align + torch.sum(grad * g)

        check = quad_term * align + self.damping * linear_term - g2
        quad_term = quad_term**2

        aux_loss = 0.5 * (reg_term + quad_term) - linear_term
        
        #if self.normalization:
        #    aux_loss = aux_loss / g2
        #    check = check / g2
        
        if train:
            aux_loss.backward()
            self.aux_loss = aux_loss.item()
            self.aux_opt.step()
            if self.aux_scheduler is not None:
                self.aux_scheduler.step()
        
        if self.normalization:
            aux_loss = aux_loss / g2
            check = check / g2

        if train:                                                                                                                                                                  
              aux_loss.backward()
              self.aux_loss = aux_loss.item()
              self.aux_opt.step()
              if self.aux_scheduler is not None:
                  self.aux_scheduler.step() 
        
        self.store_g = True
        return aux_loss, check, linear_term, quad_term, reg_term, g2

    def step(self) -> None:
        """Performes a single optimization step of FishLeg."""
        self.updated = False
        if self.update_aux_every > 0:
            if self.step_t % self.update_aux_every == 0:
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
                nat_grad = (
                    group["Qv"]()
                    if self.batch_speedup
                    else group["Qv"](
                        group["grad"]
                        if self.updated
                        else [p.grad.data for p in group["params"]]
                    )
                )

                for p, d_p, gbar, p0 in zip(
                    group["params"], nat_grad, group["gradbar"], group["theta0"]
                ):
                    gbar.copy_(self.beta * gbar + (1.0 - self.beta) * d_p)
                    if self.fine_tune:
                        delta = gbar.add(p - p0, alpha=self.weight_decay)
                    else:
                        delta = gbar.add(p, alpha=self.weight_decay / self.fish_lr)
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
