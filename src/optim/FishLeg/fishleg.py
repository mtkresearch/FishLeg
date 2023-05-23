from typing import Tuple, Callable, Any, Union, List
from collections.abc import Mapping
import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam

from .layers import *
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
        beta: float = 0.9,
        weight_decay: float = 1e-5,
        aux_lr: float = 1e-4,
        aux_betas: Tuple[float, float] = (0.9, 0.999),
        aux_eps: float = 1e-8,
        fish_scale: float = 1.0,
        damping: float = 5e-1,
        update_aux_every: int = 10,
        warmup_steps: int = 0,
        normalize_aux: bool = False,
        initialization: str = "uniform",
        grad_clip: bool = False,
        device: str = "cpu",
        verbose: bool = False,
    ) -> None:
        self.model = model

        self.aux_dataloader = aux_dataloader
        self.likelihood = likelihood

        self.update_aux_every = update_aux_every

        self.warmup_steps = warmup_steps

        self.grad_clip = grad_clip
        self.initialization = initialization
        self.normalize_aux = normalize_aux

        self.device = device
        self.verbose = verbose

        # TODO: Add more in here?
        defaults = dict(
            lr=lr,
            beta=beta,
            weight_decay=weight_decay,
            aux_lr=aux_lr,
            fish_scale=fish_scale,
            damping=damping,
        )
        params = [
            param
            for name, param in model.named_parameters()
            if "fishleg_aux" not in name
        ]
        super(FishLeg, self).__init__(params, defaults)

        aux_params = [
            param for name, param in model.named_parameters() if "fishleg_aux" in name
        ]

        if len(self.likelihood.get_parameters()) > 0:
            aux_params.extend(self.likelihood.get_aux_parameters())

        self.aux_opt = Adam(
            aux_params,
            lr=aux_lr,
            betas=aux_betas,
            eps=aux_eps,
            weight_decay=0,  # weight_decay need to be fixed to zero
        )

        if self.warmup_steps > 0:
            self.warmup_aux()

        self.step_t = 0
        self.store_g = True

    def warmup_aux(self) -> None:
        """
        Warm up auxilirary parameters with approxiamte Adam
        """
        # TODO: Add checking for this function!
        # TODO: The below loop can be refactored.
        for n, (data_x, data_y) in enumerate(self.aux_dataloader):
            data_x, data_y = data_x.to(self.device), data_y.to(self.device)
            output = self.model(data_x)
            self.likelihood(output, data_y).backward()
            if n == self.warmup_steps:
                break

        group = self.param_groups[0]
        fish_scale = group["fish_scale"]
        damping = group["damping"]

        for module in self.model.modules():
            warm_facs = []
            if not isinstance(module, nn.Sequential) and hasattr(module, "warmup"):
                for name, param in module.named_parameters():
                    if "fishleg_aux" not in name:
                        g_avg = param.grad / self.warmup_steps
                        warm_facs.append(fish_scale / (torch.square(g_avg) + damping))
                module.warmup(warm_facs, init_scale=fish_scale)
                print(module.fishleg_aux["A"])
            else:
                continue

    # TODO: ALL of this.
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

    # TODO: ALL of this.
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

        group = self.param_groups[0]
        damping = group["damping"]

        self.aux_opt.zero_grad()

        data_x, data_y = data
        pred_y = self.model(data_x)
        samples_y = self.likelihood.draw(pred_y)

        g2 = 0.0
        for group in self.param_groups:
            for grad in group["grad"]:
                g2 = g2 + torch.sum(grad * grad)
        g_norm = torch.sqrt(g2)

        self.zero_grad()
        self.likelihood.nll(pred_y, samples_y).backward()

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
                reg_term = reg_term + damping * torch.sum(d_p * d_p)
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

    # TODO: What do we need here?
    def _init_group(
        self,
        group,
        # params_with_grad,
        # grads,
        exp_avgs,
        # exp_avg_sqs,
        # max_exp_avg_sqs,
        state_steps,
    ):
        for p in group["params"]:
            if p.grad is not None:
                # params_with_grad.append(p)
                # if p.grad.is_sparse:
                #     raise RuntimeError(
                #         "Adam does not support sparse gradients, please consider SparseAdam instead"
                #     )
                # grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = (
                        torch.zeros((1,), dtype=torch.float, device=p.device)
                        # if group["capturable"] or group["fused"]
                        # else torch.tensor(0.0)
                    )
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    # state["exp_avg_sq"] = torch.zeros_like(
                    #     p, memory_format=torch.preserve_format
                    # )
                    # if group["amsgrad"]:
                    #     # Maintains max of all exp. moving avg. of sq. grad. values
                    #     state["max_exp_avg_sq"] = torch.zeros_like(
                    #         p, memory_format=torch.preserve_format
                    #     )

                exp_avgs.append(state["exp_avg"])
                # exp_avg_sqs.append(state["exp_avg_sq"])

                # if group["amsgrad"]:
                #     max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                # if group["differentiable"] and state["step"].requires_grad:
                #     raise RuntimeError(
                #         "`requires_grad` is not supported for `step` in differentiable mode"
                #     )
                state_steps.append(state["step"])

    # TODO: Add in the aux update.
    def step(self) -> None:
        """
        Performes a single optimization step of FishLeg.
        """
        self.updated = False

        # if self.step_t % self.update_aux_every == 0:
        #     self._store_u(new=True)

        # if self.step_t % self.update_aux_every == 1:
        #     # once for difference of gradients
        #     self._store_u(alpha=-1.0)
        #     info = self.update_aux()

        #     self.updated = True

        # if self.step_t % self.update_aux_every == 2:
        #     # once for gradient
        #     self._store_u(new=True)
        #     self.update_aux()
        #     self.updated = True

        self.step_t += 1

        group = self.param_groups[0]

        # TODO: get rid of anything we don't need
        # params_with_grad = []
        # grads = []
        exp_avgs = []
        # exp_avg_sqs = []
        # max_exp_avg_sqs = []
        state_steps = []
        beta = group["beta"]
        weight_decay = group["weight_decay"]
        step_size = group["lr"]  # TODO: Add scheduling in here.

        self._init_group(
            group,
            # params_with_grad,
            # grads,
            exp_avgs,
            # exp_avg_sqs,
            # max_exp_avg_sqs,
            state_steps,
        )

        # TODO: Could do with a better way of indexing the exp_avgs
        with torch.no_grad():
            p_idx = 0
            for module in self.model.modules():
                if not isinstance(module, nn.Sequential) and hasattr(module, "Qv"):
                    layer_grads = []
                    for name, param in module.named_parameters():
                        if "fishleg_aux" not in name:
                            layer_grads.append(param.grad.data)

                    nat_grads = module.Qv(layer_grads)

                    for n, (name, param) in enumerate(module.named_parameters()):
                        if "fishleg_aux" not in name:
                            nat_grad = nat_grads[n]
                            exp_avg = exp_avgs[p_idx]
                            p_idx += 1

                            exp_avg.mul_(beta).add_(nat_grad, alpha=1 - beta)

                            if weight_decay != 0:
                                exp_avg = exp_avg.add(param, alpha=weight_decay)

                            param.add_(exp_avg, alpha=-step_size)
