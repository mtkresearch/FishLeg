from typing import Optional, Tuple, Callable
import torch
import torch.nn as nn
from torch.nn import Parameter
import copy
from torch.optim import Optimizer, Adam
from torch.optim.optimizer import _use_grad_for_differentiable
from .utils import recursive_setattr, recursive_getattr

from .fishleg_layers import FishLinear

def update_dict(replace: nn.Module, module: nn.Module) -> nn.Module:
    replace_dict = replace.state_dict()
    pretrained_dict = {
        k: v for k, v in module.state_dict().items() if k in replace_dict
    }
    replace_dict.update(pretrained_dict)
    replace.load_state_dict(replace_dict)
    return replace

class FishLeg(Optimizer):
    r"""Implement FishLeg algorithm.

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
    :param Callable[[int], Tuple[torch.Tensor, torch.Tensor]] dataloader:
                A function that takes a batch size as input and output dataset 
                with corresponding size.
    :param float lr: learning rate,
                for the parameters of the input model using FishLeg (default: 1e-2)
    :param float eps: a small scalar, to evaluate the auxiliary loss
                in the direction of gradient of model parameters (default: 1e-4)

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

    Example:
        >>> auxloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=100)
        >>> trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=100)
        >>>
        >>> likelihood = FixedGaussianLikelihood(sigma_fixed=1.0)
        >>>
        >>> def nll(model, data):
        >>>     data_x, data_y = data
        >>>     pred_y = model.forward(data_x)
        >>>     return likelihood.nll(data_y, pred_y)
        >>>
        >>> def draw(model, data_x):
        >>>     pred_y = model.forward(data_x)
        >>>     return (data_x, likelihood.draw(pred_y))
        >>>
        >>> def dataloader():
        >>>     data_x, _ = next(iter(auxloader))
        >>>     return data_x
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
        >>>     dataloader
        >>> )
        >>>
        >>> for iteration in range(100):
        >>>     data_x, data_y = next(iter(trainloader))
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
        dataloader: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
        lr: float = 1e-2,
        eps: float = 1e-4,
        weight_decay: float = 1e-5,
        beta: float = 0.9,
        update_aux_every: int = -3,
        aux_lr: float = 1e-3,
        aux_betas: Tuple[float, float] = (0.9, 0.999),
        aux_eps: float = 1e-8,
        damping: float = 1e-5,
        pre_aux_training: int = 10,
        differentiable: bool = False,
        sgd_lr : float = 1e-2

    ) -> None:
        self.model = model
        self.plus_model = copy.deepcopy(self.model)
        self.sgd_lr = sgd_lr
        self.lr = lr
        
        self.model = self.init_model_aux(model)

        self.draw = draw
        self.nll = nll
        self.dataloader = dataloader

        # partition by modules
        self.aux_param = [
            param
            for name, param in self.model.named_parameters()
            if "fishleg_aux" in name
        ]

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
                    "gradbar": [torch.zeros_like(params[name]) for name in module.order],
                    "Qv": module.Qv,
                    "order": module.order,
                    "name": module_name,
                }
                param_groups.append(g)
        # TODO: add param_group for modules without aux
        defaults = dict(lr=lr, differentiable=differentiable)

        super(FishLeg, self).__init__(param_groups, defaults)
        self.aux_opt = Adam(self.aux_param, lr=aux_lr, betas=aux_betas, eps=aux_eps, weight_decay=weight_decay)
        self.eps = eps
        self.update_aux_every = update_aux_every
        self.aux_lr = aux_lr
        self.aux_betas = aux_betas
        self.aux_eps = aux_eps
        self.damping = damping
        self.weight_decay = weight_decay
        self.beta = beta
        self.pre_aux_training = pre_aux_training
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
                if isinstance(module, nn.Linear):
                    replace = FishLinear(
                        module.in_features, module.out_features, module.bias is not None,
                        init_scale=self.sgd_lr/self.lr
                    )
                    replace = update_dict(replace, module)
                    recursive_setattr(model, name, replace)
            except KeyError:
                pass

        # TODO: The above may not be a very "correct" way to do this, so please feel free to change, for example, we may want to check the name is in the fish_layer keys before attempting what is in the try statement.
        # TODO: Error checking to check that model includes some auxiliary arguments.

        return model

    def update_aux(self, D, grads_group, grad_norm) -> None:
        """Performs a single auxliarary parameter update
        using Adam. By minimizing the following objective:

        .. math::
            nll(model, \\theta + \epsilon Q(\lambda)g) + nll(model, \\theta - \epsilon Q(\lambda)g) - 2\epsilon^2g^T Q(\lambda)g

        where :math:`\\theta` is the parameters of model, :math:`\lambda` is the
        auxliarary parameters.
        """
        data_x, pred_y = D
        with torch.no_grad():
            _, sample_y = self.draw(self.model, data_x)

        self.zero_grad()
        self.nll(sample_y, pred_y).backward(retain_graph=True)

        self.aux_opt.zero_grad()
            
        aux_loss = 0.0
        for group, grad in zip(self.param_groups, grads_group):
            name = group["name"]

            grad = [g/grad_norm for g in grad]
            qg = group["Qv"](grad)

            for p, g, d_p, para_name in zip(group["params"], grad, qg, group["order"]):
                
                self.plus_model._modules[name]._parameters[para_name] = p.data + d_p * self.eps
                
                aux_loss -= torch.sum(g * d_p)
                aux_loss -= torch.sum((p.grad.data / grad_norm) * d_p) / self.eps
                aux_loss += 0.5 * self.damping * d_p.norm(p=2)**2
        

        h_plus = self.nll(sample_y, self.plus_model(data_x))

        aux_loss += (h_plus) / (self.eps**2)
        aux_loss.backward()
        
        self.aux_opt.step()
        

    
    def step(self, D) -> None:
        """Performes a single optimization step of FishLeg."""

        if self.step_t == 0 or (self.update_aux_every > 0 and 
            self.step_t % self.update_aux_every == 0) or self.update_aux_every < 0:
            
            grad_norm = 0.0
            grads_group = []
            for group in self.param_groups:
                grads = []
                for p in group['params']:
                    grad_norm += p.grad.data.norm(p=2)**2
                    grads.append(p.grad.data.clone())
                grads_group.append(grads)
            grad_norm = torch.sqrt(grad_norm)
        else:
            @_use_grad_for_differentiable
            def helper(self):
                for group in self.param_groups:
                    lr = group["lr"]

                    grad = [p.grad.data for p in group["params"]]
                    nat_grad = group["Qv"](grad)

                    for p, d_p, gbar in zip(group["params"], nat_grad, group["gradbar"]):
                        gbar.add_(d_p, alpha=(1.-self.beta)/self.beta).mul_(self.beta)
                        delta = gbar.add(p, alpha=self.weight_decay)
                        p.add_(delta, alpha=-lr)
            helper(self)
            return 

        if self.step_t == 0:
            for _ in range(self.pre_aux_training):
                self.update_aux(D, grads_group, grad_norm)
        
        if self.update_aux_every > 0:
            if self.step_t % self.update_aux_every == 0:
                self.update_aux(D, grads_group, grad_norm)
        elif self.update_aux_every < 0:
            for _ in range(-self.update_aux_every):
                self.update_aux(D, grads_group, grad_norm)

        self.step_t += 1

        @_use_grad_for_differentiable
        def helper(self):
            for group, grad in zip(self.param_groups, grads_group):
                lr = group["lr"]
                nat_grad = group["Qv"](grad)

                for p, d_p, gbar in zip(group["params"], nat_grad, group["gradbar"]):
                    gbar.add_(d_p, alpha=(1.-self.beta)/self.beta).mul_(self.beta)
                    delta = gbar.add(p, alpha=self.weight_decay)
                    p.add_(delta, alpha=-lr)
        helper(self)
        
        
            
        



