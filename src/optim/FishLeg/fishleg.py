from typing import Tuple, Callable, Any, Dict, List
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
        method: str = "rank-1",
        method_kwargs: Dict = {},
        precondition_aux: bool = True,
        u_sampling: str = "gradient",
        writer: SummaryWriter or bool = False,
    ) -> None:
        self.model = model

        self.aux_dataloader = aux_dataloader
        self.likelihood = likelihood

        self.writer = writer

        defaults = dict(
            lr=lr,
            beta=beta,
            weight_decay=weight_decay,
            aux_lr=aux_lr,
            damping=damping,
            update_aux_every=update_aux_every,
            method=method,
            method_kwargs=method_kwargs,
            precondition_aux=precondition_aux,
            u_sampling=u_sampling,
        )
        params = [
            param
            for _, param in model.named_parameters()
            if not isinstance(param, FishAuxParameter)
        ]
        super(FishLeg, self).__init__(params, defaults)

        aux_params = [
            param
            for _, param in model.named_parameters()
            if isinstance(param, FishAuxParameter)
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

    def __setstate__(self, state):
        super().__setstate__(state)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

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

        data_x, _ = next(iter(self.aux_dataloader))

        group = self.param_groups[0]
        damping = group["damping"]
        method = group["method"]
        method_kwargs = group["method_kwargs"]
        precondition_aux = group["precondition_aux"]
        u_sampling = group["u_sampling"]

        pred_y = self.model(data_x)
        with torch.no_grad():
            samples_y = self.likelihood.draw(pred_y)

        def sample_u(g: torch.Tensor) -> torch.Tensor:
            if u_sampling == "gradient":
                return g
            elif u_sampling == "gaussian":
                return torch.randn(size=g.shape)
            else:
                raise NotImplementedError(
                    f"{u_sampling} method of sampling u not implemented yet!"
                )

        surrogate_loss = 0

        v_model, u_model = [], []
        v2, u2 = 0, 0
        for module in self.model.modules():
            if isinstance(module, FishModule):
                layer_grads = []
                for param in module.not_aux_parameters():
                    layer_grads.append(sample_u(param.grad.data))

                v_layer = module.Qv(layer_grads)

                for v, u in zip(v_layer, layer_grads):
                    v2 += torch.sum(v.detach() ** 2)
                    u2 += torch.sum(u**2)
                    v_model.append(v)
                    u_model.append(u)

        u_norm = torch.sqrt(u2)
        v_norm = torch.sqrt(v2) * u_norm

        # the different methods differ in how they compute Fv_norm

        if method == "antithetic":
            eps = method_kwargs["eps"] if "eps" in method_kwargs.keys() else 1e-4

            def _augment_params_by(eps: float):
                p_idx = 0
                for module in self.model.modules():
                    if isinstance(module, FishModule):
                        for param in module.not_aux_parameters():
                            param.data += eps * v_model[p_idx].detach() / v_norm
                            p_idx += 1

            # Add to gradients
            _augment_params_by(eps)

            pred_y = self.model(data_x)
            plus_loss = self.likelihood.nll(pred_y, samples_y)

            plus_grad = torch.autograd.grad(
                plus_loss,
                inputs=group["params"],
                allow_unused=True,
            )

            # Minus gradients
            _augment_params_by(-2 * eps)

            pred_y = self.model(data_x)
            minus_loss = self.likelihood.nll(pred_y, samples_y)

            minus_grad = torch.autograd.grad(
                minus_loss,
                inputs=group["params"],
                allow_unused=True,
            )

            # Restore parameters
            _augment_params_by(eps)

            Fv_norm = list(map(lambda a, b: 0.5 * (a - b) / eps, plus_grad, minus_grad))

        elif method == "rank-1":
            loss = self.likelihood.nll(pred_y, samples_y)

            gs = torch.autograd.grad(
                outputs=loss,
                inputs=group["params"],
                allow_unused=True,
            )

            p_idx = 0
            g_dot_w = 0
            for module in self.model.modules():
                if isinstance(module, FishModule):
                    for param in module.not_aux_parameters():
                        nat_grad = v_model[p_idx]
                        sample_grad = gs[p_idx]
                        g_dot_w += torch.sum(nat_grad.detach() * sample_grad)
                        p_idx += 1

            Fv_norm = list(map(lambda a: g_dot_w * a / v_norm, v_model))

        else:
            raise NotImplementedError(
                f"{method} method of approximation not implemented yet!"
            )

        # Note that here v_norm already contains a factor of u_norm!
        v_adj = list(
            map(
                lambda Fv, v, u: (
                    (Fv + float(damping) * v.detach() / v_norm) - u / v_norm
                ),
                Fv_norm,
                v_model,
                u_model,
            )
        )

        if precondition_aux:
            aux_loss = 0
            with torch.no_grad():
                v_adj_new = []
                count = 0
                for module in self.model.modules():
                    if isinstance(module, FishModule):
                        v_adj_group = []
                        for param in module.not_aux_parameters():
                            v_adj_group.append(v_adj[count])
                            aux_loss += torch.dot(
                                v_adj[count].reshape(-1), v_model[count].reshape(-1)
                            )
                            count += 1
                        v_adj_new_group = module.Qv(v_adj_group)
                        for v in v_adj_new_group:
                            v_adj_new.append(v / v_norm)
                v_adj = v_adj_new

        for v_a, v in zip(v_adj, v_model):
            surrogate_loss += torch.dot(v_a.reshape(-1), v.reshape(-1))

        surrogate_loss.backward()

        # With no preconditioning, aux loss and surrogate loss should be identical.
        if not precondition_aux:
            aux_loss = surrogate_loss.item()

        if self.writer:
            self.writer.add_scalar(
                "AuxLoss/train",
                aux_loss,
                self.state[group["params"][-1]]["step"],
            )
            self.writer.add_scalar(
                "SurrogateLoss/train",
                surrogate_loss.detach(),
                self.state[group["params"][-1]]["step"],
            )
            self.writer.add_scalar(
                "lr",
                group["lr"],
                self.state[group["params"][-1]]["step"],
            )

        self.aux_opt.step()
        return surrogate_loss.item()

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
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
                    state["step"] = torch.tensor(0.0)
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avgs.append(state["exp_avg"])
                state_steps.append(state["step"])

    def step(self) -> None:
        """
        Performes a single optimization step of FishLeg.
        """
        group = self.param_groups[0]

        params_with_grad = []
        grads = []
        exp_avgs = []
        state_steps = []
        beta = group["beta"]
        weight_decay = group["weight_decay"]
        update_aux_every = group["update_aux_every"]
        step_size = group["lr"]

        self._init_group(
            group,
            params_with_grad,
            grads,
            exp_avgs,
            state_steps,
        )

        step_t = state_steps[0]

        if step_t % update_aux_every == 0 and step_t != 0:
            self.update_aux()

        # TODO: Could do with a better way of indexing here maybe?
        with torch.no_grad():
            p_idx = 0
            for module in self.model.modules():
                if isinstance(module, FishModule):
                    layer_grads = []
                    for param in module.not_aux_parameters():
                        layer_grads.append(param.grad.data)

                    nat_grads = module.Qv(layer_grads)

                    for n, (_, param) in enumerate(module.named_parameters()):
                        if not isinstance(param, FishAuxParameter):
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
                    for n, (_, param) in enumerate(module.named_parameters()):
                        if not isinstance(param, FishAuxParameter):
                            grad = grads[p_idx]
                            exp_avg = exp_avgs[p_idx]
                            state_steps[p_idx] += 1
                            p_idx += 1

                            exp_avg.mul_(beta).add_(grad, alpha=1 - beta)

                            if weight_decay != 0:
                                exp_avg = exp_avg.add(param, alpha=weight_decay)

                            param.add_(exp_avg, alpha=-step_size)
