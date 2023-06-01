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
import csv
import os
import copy
from transformers.models.bert.modeling_bert import BertAttention

from .utils import recursive_setattr, recursive_getattr, update_dict, get_named_layers_by_regex, NamedLayer
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
        draw: Callable[[nn.Module, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        nll: Callable[[nn.Module, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
        aux_dataloader: torch.utils.data.DataLoader,
        likelihood: FishLikelihood = None,
        lr: float = 5e-2,
        eps: float = 1e-4,
        damping: float = 5e-1,
        weight_decay: float = 1e-5,
        beta: float = 0.9,
        update_aux_every: int = 10,
        aux_lr: float = 1e-4,
        aux_betas: Tuple[float, float] = (0.9, 0.999),
        aux_eps: float = 1e-8,
        num_steps=None,
        batch_speedup: bool = False,
        full: bool = True,
        normalization: bool = False,
        fine_tune: bool = False,
        module_names: List[str] = [],
        skip_names: List[str] = [],
        initialization: str = "uniform",
        scale: float = 1.0,
        sample_gm: int = 10,
        clip: bool = False,
        warmup: int = 0,
        warmup_data: torch.utils.data.DataLoader = None,
        warmup_loss: Callable = None,
        device: str = "cpu",
        config = None,
        verbose = False,
        output_dir = None
    ) -> None:
        self.model = model
        self.lr = lr
        self.device = device
        self.batch_speedup = batch_speedup
        self.full = full
        self.initialization = initialization
        self.normalization = normalization
        self.fine_tune = fine_tune
        self.likelihood = likelihood
        self.warmup = warmup
        self.scale = scale
        self.eps = eps
        self.sample_gm = sample_gm
        self.clip = clip
        self.aux_lr = aux_lr
        self.fish_lr = lr
        self.damping = damping
        self.verbose = verbose
        self.output_dir = output_dir

        self.draw = draw
        self.nll = nll
        self.aux_dataloader = aux_dataloader
        
        self.plus, self.minus = copy.deepcopy(model), copy.deepcopy(model)
        
        self.model, param_groups = self.init_model_aux(
                            model, 
                            module_names=module_names,
                            skip_names=skip_names,
                            config=config
                        )
        defaults = dict(aux_lr=aux_lr, lr=lr)
        super(FishLeg, self).__init__(param_groups, defaults)

        if self.warmup > 0:
            self.warmup_aux(warmup_data, warmup_loss, scale=scale)
        else:
            self.warmup_aux(scale=scale)

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
            weight_decay=0,
            # weight_decay need to be fixed to zero
        )

        if num_steps is not None:
            self.aux_scheduler = get_scheduler(
                name="linear",
                optimizer=self.aux_opt,
                num_warmup_steps=0,
                num_training_steps=num_steps,
            )
        else:
            self.aux_scheduler = None

        self.scheduler = None

        self.update_aux_every = update_aux_every
        self.weight_decay = weight_decay
        self.beta = beta
        self.step_t = 0
        self.store_g = True
        
        self.aux_loss, self.check, self.check2 = [],[],[]
        aux_file = os.path.join(self.output_dir, "aux_evolve.csv")
        if os.path.exists(aux_file):
            os.remove(aux_file)

    def init_model_aux(
        self,
        model: nn.Module,
        module_names: List[str],
        skip_names: List[str],
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
        if len(module_names) == 0 or module_names[0] == 're:*':
            named_layers = [NamedLayer(name, layer) for name, layer in model.named_modules()]
        
        else:
            named_layers = get_named_layers_by_regex(model, module_names)
        
        replaced_layers = [] 
        for named_layer in named_layers:
            if True: 
                module = named_layer.layer
                module_name = named_layer.layer_name
                
                skip = False
                for layer in replaced_layers:
                    if re.match(layer.layer_name, module_name):
                        inner_name = module_name[len(layer.layer_name)+1:]
                        if any(
                                re.match(inner_name, param_name)
                                for param_name in layer.layer.order
                            ):
                            skip = True
                if skip or any([
                        name in module_name for name in skip_names
                    ]): continue
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
                    replace = update_dict(replace, module)
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
                elif isinstance(module, BertAttention):
                    if config is None:
                        config = model.config
                    replace = FishBertAttention(
                            config, 
                            device=self.device
                    )
                    replace = update_dict(replace, module)
                elif isinstance(module, nn.BatchNorm2d):
                    replace = FishBatchNorm2d(
                            num_features=module.num_features,
                            eps=module.eps,
                            momentum=module.momentum,
                            affine=module.affine,
                            track_running_stats=module.track_running_stats,
                            init_scale=self.scale,
                            device=self.device
                    )
                    replace = update_dict(replace, module)
                elif isinstance(module, nn.LayerNorm):
                    replace = FishLayerNorm(
                        normalized_shape=module.normalized_shape,
                        eps=module.eps,
                        elementwise_affine=module.elementwise_affine,
                        init_scale=self.scale,
                        device=self.device
                    )
                    replace = update_dict(replace, module)
                else:
                    continue
                    #raise Warning(f'The FishLayer for module named {module_name} has not been implemented and hence skipped.')
                recursive_setattr(model, module_name, replace)
                replaced_layers.append(NamedLayer(module_name, replace))
                if self.verbose:
                    print("Replaced with FishLeg Layer: \t ", module_name)
            #except:
            #    raise TypeError(f'The given model has no module named {module_name}.')
        
        # Define each modules
        model.to(self.device)
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
                    "gradbar": [
                        torch.zeros_like(params[name]) for name in module.order
                    ],
                    "u": [torch.zeros_like(params[name]) for name in module.order],
                    "grad": [torch.zeros_like(params[name]) for name in module.order],
                    "ggu": [torch.zeros_like(params[name]) for name in module.order],
                    "Qv": module.Qg
                    if self.batch_speedup
                    else partial(module.Qv, full=self.full),
                    "order": module.order,
                    "name": named_layer.layer_name,
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
            
        if self.likelihood is not None:
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
        scale: float = 1.0,
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
                    g2_avg = g2 / int(self.warmup / dataloader.batch_size)
                    ds.append(1. / torch.sqrt(g2_avg + self.damping))

                group["module"].warmup(
                    ds, batch_speedup=self.batch_speedup, init_scale=scale
                )
            else:
                group["module"].warmup(
                    batch_speedup=self.batch_speedup, init_scale=scale
                )
    
    #def parameters(self,):
    #    for group in self.param_groups:
    #        for param in group["params"]:
    #            yield param
    
    def update_fisher(
        self,
        num_samples: int = 256
    ):
        aux, check, check2 = 0, 0, 0
        for t in range(num_samples):
            data = next(iter(self.aux_dataloader))
            data = self._prepare_input(data)
            self.zero_grad()
            self.nll(self.model, data).backward()
            self._store_u(new=True)

            data = next(iter(self.aux_dataloader))
            data = self._prepare_input(data)
            self.zero_grad()
            self.nll(self.model, data).backward()
            self._store_u(alpha=-1.0)

            info = self.update_aux()
            info = [e.detach().cpu().numpy() for e in info]
            
            aux += info[0]
            check += info[1]
            check2 += info[2]
            if t % 32 == 0:
                if t > 0: 
                    aux /= 32
                    check /= 32
                    check2 /= 32
                print(
                        "iter:{:d}, \taux:{:.2f} check:{:.2f} check2:{:.2f} \tauxloss:{:.2f} \tcheck:{:.2f} \tcheck2:{:.2f} \tlinear:{:.2f} \tquad:{:.2f} \treg:{:.2f} \tg2:{:.2f}".format(
                                    t, aux, check, check2, *info
                                ))
                if t > 0: aux = 0
        
    def pretrain_fish(
        self,
        dataloader: torch.utils.data.DataLoader,
        loss: Callable[[nn.Module, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
        iterations: int = 10000,
        initial: int = 1000,
        difference: bool = False,
        verbose: bool = False,
        testloader: torch.utils.data.DataLoader = None,
        batch_size: int = 500,
        fisher: bool = True,
        noise: bool = False,
    ) -> List:

        aux, checks, checks2 = 0, 0, 0
        
        aux_file = os.path.join(self.output_dir, "pretrain.csv")
        if os.path.exists(aux_file):
            os.remove(aux_file)

        with open(aux_file, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            names = ["iter", "lr", "auxloss", "check", "check2", "linear", "quad", "reg", "g2"]
            for group in self.param_groups:
                names.append(group["name"])
            writer.writerow(names)

            
        for pre in range(iterations):
            self.zero_grad()
            batch = next(iter(dataloader))
            batch = self._prepare_input(batch)
            loss(self.model, batch).backward()
            self._store_u(new=True, add_noise=noise)

            if difference:
                self.zero_grad()
                batch = next(iter(dataloader))
                batch = self._prepare_input(batch)
                loss(self.model, batch).backward()
                self._store_u(alpha=-1.0)

            info = self.update_aux(fisher=fisher)
            aux_loss = info[0].detach().cpu().numpy()
            check = info[1].detach().cpu().numpy()
            check2 = info[2].detach().cpu().numpy()
            checks_layers = info[-1]
            infos = [e.detach().cpu().numpy() for e in info[:-1]]    
            
            with open(aux_file, "a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                row = [pre,  self.aux_opt.param_groups[0]['lr'], *infos]
                for name in names[len(infos)+2:]:
                    row.append(checks_layers[name])
                writer.writerow(row)

            if pre % (batch_size * 10) == 0:
                save_path = os.path.join(self.output_dir, "fl_model_checkpoint_" + str(pre) + ".pth")
                with open(save_path, mode="wb") as file_:
                    torch.save(
                                {
                                    "model_state_dict": self.model.state_dict(),
                                }, file_
                            )
            if verbose:
                aux += aux_loss
                checks += check
                checks2 += check2
                if pre % batch_size == 0 and pre > 0:
                    checks = checks / batch_size
                    checks2 = checks2 / batch_size
                    aux = aux / batch_size

                    print(
                        "iter:{:d},auxlr:{:.5f} \tBATCH auxloss:{:.2f} \tcheck:{:.2f} \tcheck2:{:.2f} \tauxloss:{:.2f} \tcheck:{:.2f} \tcheck2:{:.2f} \tlinear:{:.2f} \tquad:{:.2f} \treg:{:.2f} \tg2:{:.2f}".format(
                                    pre, self.aux_opt.param_groups[0]['lr'], aux, checks, checks2, *info
                                )
                            )
                            
                    aux, checks, checks2 = 0, 0, 0

        return

    def _store_u(
        self,
        transform: Callable = lambda x: x,
        alpha: float = 1.0,
        new: bool = False,
        add_noise: bool = False,
    ):
        '''
        if add_noise:
            g2 = 0.0
            for group in self.param_groups:
                for p in group["params"]:
                    mask = ~(p.data == 0)
                    g2 += torch.sum(torch.square(p.grad.data))
        '''
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if add_noise:
                    mask = ~(p.data == 0)
                    scale = torch.sqrt(p.grad.data[mask].var()) # * (1./(torch.abs(p.data) + 1.))
                    group["grad"][i].copy_(torch.randn_like(p.data) * scale * mask)
                else:
                    grad = transform(p.grad.data)
                    
                    if not new:
                        group["grad"][i].add_(grad, alpha=alpha)
                    else:
                        group["grad"][i].copy_(grad)
                    group["u"][i].copy_(grad)
    
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

    def update_aux(self, train=True, fisher=True, data=None) -> None:
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
         
        u2 = 0.0
        for group in self.param_groups:
            for grad in group["grad"]:
                u2 = u2 + torch.sum(grad * grad)
        u_norm = torch.sqrt(u2)
        
        self.zero_grad()
        loss = self.nll(self.model, samples)
        #loss.backward()

        reg_term = 0.0
        quad = 0.0
        linear_term = 0.0
        align = 0.0

        for i, group in enumerate(self.param_groups):
            if self.normalization:
                u_normalized = [u/u_norm for u in group["grad"]]
            else:
                u_normalized = group["grad"]
            qu = group["Qv"]() if self.batch_speedup else group["Qv"](u_normalized)
            name = group["name"]
            group['v'] = qu
            for p, para_name, u, dqu in zip(
                group["params"], group["order"], u_normalized, qu
            ):
                #grad = p.grad.data
                #quad = quad + torch.sum(grad * dqu)
                mask = ~(u == 0)
                linear_term = linear_term + torch.sum(u * dqu)
                reg_term = reg_term + self.damping * torch.sum(dqu * dqu * mask)
                #align = align + torch.sum(grad * u)

                plus_param = recursive_getattr(self.plus, '.'.join([name, *para_name.split('.')[:-1]]))
                minus_param = recursive_getattr(self.minus, '.'.join([name, *para_name.split('.')[:-1]]))

                plus_param._parameters[para_name.split('.')[-1]] = (p.data + dqu * self.eps) * mask
                minus_param._parameters[para_name.split('.')[-1]]= (p.data - dqu * self.eps) * mask
        
        h_plus = self.nll(self.plus, samples)
        h_minus = self.nll(self.minus, samples)
        
        #check = align * quad + self.damping * linear_term - 1
         
        quad_term = (h_plus + h_minus) / (self.eps**2)
        aux_loss = quad_term + reg_term - 2 * linear_term
        
        '''
        if self.normalization:
            aux_loss = aux_loss / u2
            check = check / u2
        '''

        if train:
            aux_loss.backward()
            self.aux_opt.step()
            if self.aux_scheduler is not None:
                self.aux_scheduler.step()
        
        checks_layer = {}
        check, check3, norms1, norms2 = 0,0,0,0
        batch_size = samples["input_ids"].shape[0]
        for k in range(batch_size):
            self.zero_grad()
            sample = {name: v[[k]] for name,v in samples.items()}
            self.nll(self.model, sample).backward()
            with torch.no_grad():
                gu, uqg = 0, 0
                for group in self.param_groups:
                    for p, u, dqu in zip(
                        group["params"], group["grad"], group["v"]   
                    ):
                        if self.normalization: u = u / u_norm
                        grad = p.grad.data
                        gu += torch.sum(grad * u)
                        uqg += torch.sum(grad * dqu)

                check += uqg * gu
                for i, group in enumerate(self.param_groups):
                    qg = group["Qv"]([p.grad.data for p in group["params"]])
                
                    for ggu, dqg in zip(
                        group['ggu'], qg
                    ):
                        if k==0:
                            ggu.copy_(dqg * gu)
                        else:
                            ggu.add_(dqg * gu)

        with torch.no_grad():
            check = check / batch_size + self.damping * linear_term - 1
        
            for group in self.param_groups:
                sim, norm1, norm2 = 0,0,0
                for u, dqu, qggu in zip(
                    group["grad"], group["v"], group["ggu"]
                ):
                    if self.normalization: u = u / u_norm
                    qfu = qggu / batch_size + self.damping * dqu
                    norm1 += torch.sum(qfu * qfu)
                    norm2 += torch.sum(u * u)
                    sim += torch.sum(qfu * u)
                checks_layer[group["name"]] = (sim/torch.sqrt(norm1*norm2)).detach().cpu().numpy()
                check3 += sim
                norms1 += norm1
                norms2 += norm2
        
            check3 = check3 / torch.sqrt(norms1 * norms2)
            #check2 = (h_plus + h_minus - 2*loss)/(self.eps**2) + reg_term - linear_term
        
        self.store_g = True
        return aux_loss, check, check3, linear_term, quad_term, reg_term, u2, checks_layer

    def step(self, closure=None, data=None) -> None:
        """Performes a single optimization step of FishLeg."""
        self.updated = False

        if self.update_aux_every > 0:
            if self.step_t % self.update_aux_every == 0:
                #self._store_u(new=True)
                pass
            if self.step_t % self.update_aux_every == 1:
                # once for difference of gradients
                self._store_u(add_noise=True)
                info = self.update_aux()
                info = [e.detach().cpu().numpy() for e in info]
                        
                self.aux_loss.append(info[0])
                self.check.append(info[1])
                self.check2.append(info[2])

                if self.verbose==True:

                    if self.step_t % int(self.update_aux_every*10) == 1:
                        batch_aux = sum(self.aux_loss[-10:])/min(len(self.aux_loss), 10)
                        batch_check = sum(self.check[-10:])/min(len(self.check), 10)
                        batch_check2 = sum(self.check2[-10:])/min(len(self.check2), 10)
                        print(
                                "iter:{:d},lr:{:.6f},auxlr:{:.6f}\t BATCH aux:{:.2f}, check:{:.2f} check2:{:.2f} \tauxloss:{:.2f} check:{:.2f} check2:{:.2f} \tlinear:{:.2f} \tquad:{:.2f} \treg:{:.2f} \tg2:{:.2f}".format(
                                    self.step_t,  self.param_groups[0]['lr'], self.aux_opt.param_groups[0]["lr"], batch_aux, batch_check, batch_check2, *info
                                )
                            )
                        with open(
                            os.path.join(self.output_dir, "aux_evolve.csv"), "a", newline=""
                        ) as csv_file:
                            writer = csv.writer(csv_file)
                            writer.writerow([
                                    self.step_t, self.aux_opt.param_groups[0]["lr"], batch_aux, batch_check, batch_check2, *info
                                ])
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
                        group["u"]
                        if self.updated
                        else [p.grad.data for p in group["params"]]
                    )
                )
                if self.fine_tune:

                    for p, d_p, gbar, p0 in zip(
                        group["params"], nat_grad, group["gradbar"], group["theta0"]
                    ):
                        gbar.copy_(self.beta * gbar + (1.0 - self.beta) * d_p)
                        delta = gbar.add(p - p0, alpha=self.weight_decay)
                        p.add_(delta, alpha=-self.fish_lr)
                else:
                    for p, d_p, gbar in zip(
                        group["params"], nat_grad, group["gradbar"]
                    ):
                        mask = ~(p.data == 0)
                        gbar.copy_((self.beta * gbar + (1.0 - self.beta) * d_p) * mask)
                        delta = gbar.add(p * mask, alpha=self.weight_decay)
                        p.add_(delta, alpha=-group['lr'])
        if self.scheduler is not None:
            self.scheduler.step()

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
