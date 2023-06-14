from typing import Tuple, Callable, Any, Union, List
from collections.abc import Mapping
import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam
import sys

from torch.utils.tensorboard import SummaryWriter

from .layers import *
from .likelihoods import FishLikelihoodBase

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
    :param int warmup_steps: If warmup_steps is zero, the default SGD warmup will be used, where Q is
                initialized as a scaled identity matrix. If warmup is positive, the diagonal
                of Q will be initialized as :math:`\frac{1}{g^2 + \gamma}`; and in this case,
                warmup_data and warmup_loss should be provided for sampling of gradients.
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
        likelihood: FishLikelihoodBase,
        lr: float = 5e-2,
        beta: float = 0.9,
        weight_decay: float = 1e-5,
        aux_lr: float = 1e-4,
        aux_betas: Tuple[float, float] = (0.9, 0.999),
        aux_eps: float = 1e-8,
        damping: float = 5e-1,
        update_aux_every: int = 10,
        warmup_steps: int = 0,
        device: str = "cpu",
        writer: SummaryWriter or bool = False,
    ) -> None:
        self.model = model

        self.aux_dataloader = aux_dataloader
        self.likelihood = likelihood

        self.device = device
        self.writer = writer

        defaults = dict(
            lr=lr,
            beta=beta,
            weight_decay=weight_decay,
            aux_lr=aux_lr,
            damping=damping,
            update_aux_every=update_aux_every,
            warmup_steps=warmup_steps,
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

        if warmup_steps > 0:
            self.warmup_aux(num_steps=warmup_steps)

    def __setstate__(self, state):
        super().__setstate__(state)
        # TODO: Add things here for the state_dict!
        # for group in self.param_groups:
        # group.setdefault('amsgrad', False)
        # group.setdefault('maximize', False)
        # group.setdefault('foreach', None)
        # group.setdefault('capturable', False)
        # group.setdefault('differentiable', False)
        # group.setdefault('fused', None)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    def warmup_aux(self, num_steps: int) -> None:
        """
        Warm up auxilirary parameters with approxiamte Adam
        """
        # TODO: Add checking for this function!
        for n, (data_x, data_y) in enumerate(self.aux_dataloader):
            data_x, data_y = data_x.to(self.device), data_y.to(self.device)
            pred_y = self.model(data_x)
            loss = self.likelihood.nll(pred_y, data_y)

            group = self.param_groups[0]
            grads = torch.autograd.grad(
                outputs=loss,
                inputs=group["params"],
                # create_graph=True,  # TODO: Check this!
                allow_unused=True,
            )

            for module in self.model.modules():
                if not isinstance(module, nn.Sequential) and hasattr(module, "warmup"):
                    warm_g2 = []
                    for (name, param), grad in zip(module.named_parameters(), grads):
                        if "fishleg_aux" not in name:
                            g2 = grad**2
                            warm_g2.append(g2)
                    module.add_warmup_grad(warm_g2)
                else:
                    continue

            if n == num_steps:
                break

        group = self.param_groups[0]
        damping = group["damping"]

        # Moves through all diagonal Fisher initialisations and divides by the warmup
        # factor
        for module in self.model.modules():
            if not isinstance(module, nn.Sequential) and hasattr(
                module, "finalise_warmup"
            ):
                module.finalise_warmup(damping, num_steps)
            else:
                continue

    # TODO: ALL of this.
    def fish_pretrain_step(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: Callable[
            [nn.Module, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor
        ],  # TODO: Maybe call this likelihood?
    ) -> List:
        data_x, data_y = next(iter(dataloader))
        data_x, data_y = data_x.to(self.device), data_y.to(self.device)

        self.zero_grad()

        pred_y = self.model(data_x)

        criterion(pred_y, data_y).backward()

        for param in self.param_groups["params"]:
            param.grad.data.mul_(-1)

        # TODO: Get current gradients of model and muliply by -1
        # TODO: Do second backward step to add second set of gradients to the model
        # TODO: Change this to using torch.autograd.grad?

        data_x, data_y = next(iter(dataloader))
        data_x, data_y = data_x.to(self.device), data_y.to(self.device)

        pred_y = self.model(data_x)

        criterion(pred_y, data_y).backward()

        aux_loss = self.update_aux()

        return aux_loss

    def update_aux(self) -> None:
        """
        Performs a single auxliarary parameter update
        using Adam. By minimizing the following objective:

        .. math::
            nll(model, \\theta + \epsilon Q(\lambda)g) + nll(model, \\theta - \epsilon Q(\lambda)g) - 2\epsilon^2g^T Q(\lambda)g

        where :math:`\\theta` is the parameters of model, :math:`\lambda` is the
        auxliarary parameters.
        """
        self.aux_opt.zero_grad()

        data_x, data_y = next(iter(self.aux_dataloader))
        data_x, data_y = data_x.to(self.device), data_y.to(self.device)

        group = self.param_groups[0]
        damping = group["damping"]

        pred_y = self.model(data_x)

        with torch.no_grad():
            samples_y = self.likelihood.draw(pred_y)

        loss = self.likelihood.nll(pred_y, samples_y)

        # TODO: I think using autograd.grad avoids accumulating gradients like backward does?
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=group["params"],
            # create_graph=True,  # TODO: Check this!
            allow_unused=True,
        )

        u2 = 0.0
        for group in self.param_groups:
            for params in group["params"]:
                u2 = u2 + torch.sum(params.grad * params.grad)
        u_norm = torch.sqrt(u2)

        reg_term = 0.0
        quad_term = 0.0
        linear_term = 0.0

        p_idx = 0
        for module in self.model.modules():
            if not isinstance(module, nn.Sequential) and hasattr(module, "Qv"):
                layer_grads = []
                for name, param in module.named_parameters():
                    if "fishleg_aux" not in name:
                        layer_grads.append(param.grad.data / u_norm)

                nat_grads = module.Qv(layer_grads)

                for n, (name, param) in enumerate(module.named_parameters()):
                    if "fishleg_aux" not in name:
                        nat_grad = nat_grads[n]
                        sample_grad = grads[p_idx]
                        true_grad = layer_grads[n]
                        p_idx += 1

                        quad_term += torch.sum(sample_grad * nat_grad)
                        linear_term += torch.sum(true_grad * nat_grad)
                        reg_term += damping * torch.sum(nat_grad * nat_grad)

        quad_term = quad_term**2

        aux_loss = 0.5 * (reg_term + quad_term) - linear_term

        if self.writer:
            self.writer.add_scalar(
                "AuxLoss/train",
                aux_loss,
                self.state[self.param_groups[0]["params"][-1]]["step"],
            )

        aux_grads = torch.autograd.grad(
            outputs=aux_loss,
            inputs=self.aux_opt.param_groups[0]["params"],
            # create_graph=True,  # TODO: Check this!
            allow_unused=True,
        )
        for n, a_p in enumerate(self.aux_opt.param_groups[0]["params"]):
            a_p.grad = aux_grads[n]

        self.aux_opt.step()
        return aux_loss.item()

    # TODO: What do we need here? - This method is taken from Adam, to discuss!
    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        # exp_avg_sqs,
        # max_exp_avg_sqs,
        state_steps,
    ):
        for p in group["params"]:
            if p.grad is not None:
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = (
                        # torch.zeros((1,), dtype=torch.float, device=p.device)
                        # if group["capturable"] or group["fused"]
                        # else
                        torch.tensor(0.0)
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

    def step(self) -> None:
        """
        Performes a single optimization step of FishLeg.
        """
        # TODO: Adam loops over groups - Not sure what this does?
        group = self.param_groups[0]

        # TODO: get rid of anything we don't need
        params_with_grad = []
        grads = []
        exp_avgs = []
        # exp_avg_sqs = []
        # max_exp_avg_sqs = []
        state_steps = []
        beta = group["beta"]
        weight_decay = group["weight_decay"]
        update_aux_every = group["update_aux_every"]
        step_size = group["lr"]  # TODO: Add scheduling in here.

        self._init_group(
            group,
            params_with_grad,
            grads,
            exp_avgs,
            # exp_avg_sqs,
            # max_exp_avg_sqs,
            state_steps,
        )

        step_t = state_steps[0]

        if step_t % update_aux_every == 0 and step_t != 0:
            self.update_aux()

        # TODO: Could do with a better way of indexing here maybe?
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
                            state_steps[p_idx] += 1
                            p_idx += 1

                            exp_avg.mul_(beta).add_(nat_grad, alpha=1 - beta)

                            # TODO: CHECK
                            if weight_decay != 0:
                                exp_avg = exp_avg.add(param, alpha=weight_decay)

                            param.add_(exp_avg, alpha=-step_size)

                # This is for updating non-FishLeg layers - do we want this/needs checking.
                elif not isinstance(module, nn.Sequential) and not isinstance(
                    module, nn.ParameterDict
                ):
                    for n, (name, param) in enumerate(module.named_parameters()):
                        if "fishleg_aux" not in name:
                            grad = grads[p_idx]
                            exp_avg = exp_avgs[p_idx]
                            state_steps[p_idx] += 1
                            p_idx += 1

                            exp_avg.mul_(beta).add_(grad, alpha=1 - beta)

                            if weight_decay != 0:
                                exp_avg = exp_avg.add(param, alpha=weight_decay)

                            param.add_(exp_avg, alpha=-step_size)
