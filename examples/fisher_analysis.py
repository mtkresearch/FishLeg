import torch
import torch.nn as nn
import numpy as np
import sys
from datetime import datetime
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

sys.path.append("../src")

import matplotlib.pyplot as plt

from optim.FishLeg import FishLeg, FISH_LIKELIHOODS, FishLinear

np.random.seed(1)
torch.random.manual_seed(1)

N = 100
gamma = 0.001

U, _ = torch.linalg.qr(torch.randn((N, N)))

lambda_i = []
for i in range(1, N + 1):
    lambda_i.append(1 / (i**2))

Lambda = torch.Tensor(lambda_i)

F = (U @ Lambda) @ U.T

teacher_model = nn.Linear(N, 1, bias=False)

targets = 1 / (Lambda + gamma)


def dataloader(batch_size: int = 1):
    while True:
        z = torch.Tensor(np.random.normal(0, 1, size=(batch_size, N)))
        x = torch.matmul(z * torch.sqrt(Lambda), U.T)
        yield x, teacher_model(x)


# for K in [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 100000]:
#     loader = dataloader(batch_size=K)

#     x, y = next(loader)

#     F_app = torch.matmul(x.T, x) / K

#     eig_app = torch.svd(F_app)[1]

writer = SummaryWriter(
    log_dir=f"runs/tests/{datetime.now().strftime('%Y%m%d-%H%M%S')}",
)

loader = dataloader(batch_size=100)

student_model = FishLinear(N, 1, init_scale=1 / gamma, bias=False)

likelihood = FISH_LIKELIHOODS["gaussian"](sigma=1.0)

lr_SGD = 1e-4

lr_fl_inf = 0  # gamma*lr_SGD
lr_fl_zero = 0  # gamma * lr_SGD
warmup_lr_K = 100

opt = FishLeg(
    student_model,
    loader,
    likelihood,
    lr=lr_fl_zero,
    beta=0.7,
    weight_decay=0,  # 1e-5,
    aux_lr=0.0001,
    aux_betas=(0.9, 0.99),
    aux_eps=1e-4,
    damping=gamma,
    update_aux_every=1,
    method="antithetic",
    method_kwargs={
        "eps": 1e-4,
    },
    writer=writer,
)


# rrt = torch.matmul(student_model.fishleg_aux["R"], student_model.fishleg_aux["R"].T)
# llt = torch.matmul(
#     student_model.fishleg_aux["L"][:N, :N], student_model.fishleg_aux["L"][:N, :N].T
# )
# A = student_model.fishleg_aux["A"][:, :N].squeeze()

target_diag = 1 / (Lambda + gamma)

print(target_diag)

rtr = torch.matmul(student_model.fishleg_aux["R"].T, student_model.fishleg_aux["R"])
llt = torch.matmul(
    student_model.fishleg_aux["L"],
    student_model.fishleg_aux["L"].T,
)
A = student_model.fishleg_aux["A"].squeeze()

Q = torch.kron(rtr, torch.diag(A) @ llt @ torch.diag(A))
final_diag = torch.diag(Q).detach().numpy()

fig, ax = plt.subplots(1, 1)
ax.plot(sorted(final_diag), sorted(target_diag), ".")
ax.plot(sorted(target_diag), sorted(target_diag), ls="--", color="k")


k = 0
for epoch in range(1, 1001):
    with tqdm(loader, unit="batch") as tepoch:
        running_loss = 0
        tepoch.set_description(f"Epoch {epoch}")
        for batch in range(100):
            for g in opt.param_groups:
                g["lr"] = min(
                    lr_fl_zero + (lr_fl_inf - lr_fl_zero) * k / warmup_lr_K, lr_fl_inf
                )
            opt.zero_grad()
            x, y = next(loader)
            pred_y = student_model(x)
            loss = likelihood(pred_y, y)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            k += 1
            if batch % 50 == 0:
                # Write out the losses per epoch
                writer.add_scalar(
                    "Loss/train",
                    running_loss / (batch + 1),
                    (epoch * 100) + batch,
                )

                rtr = torch.matmul(
                    student_model.fishleg_aux["R"].T, student_model.fishleg_aux["R"]
                )
                llt = torch.matmul(
                    student_model.fishleg_aux["L"],
                    student_model.fishleg_aux["L"].T,
                )
                A = student_model.fishleg_aux["A"].squeeze()

                Q = torch.kron(rtr, torch.diag(A) @ llt @ torch.diag(A))

                # Q = sum(llt) * torch.diag(A) @ rtr @ torch.diag(A)
                # # F_inv = torch.matmul(llt, torch.diag(A) ** 2)
                # # Q = student_model.Qv((torch.eye(N),))
                # # Q = torch.eye(N)

                eig_app = torch.diag((U.T @ Q) @ U)
                eig_mse = torch.sum((eig_app - targets) ** 2).item()

                # for n, (eigenval, target) in enumerate(zip(eig_app, targets)):
                #     writer.add_scalars(
                #         f"Eigenvalues/{n}",
                #         {"pred": eigenval, "target": target},
                #         (epoch * 100) + batch,
                #     )
                #     if n == 5:
                #         break

                # for n, (eigenval, target) in enumerate(
                #     zip(reversed(eig_app), reversed(targets))
                # ):
                #     writer.add_scalars(
                #         f"Eigenvalues/{100 - n}",
                #         {"pred": eigenval, "target": target},
                #         (epoch * 100) + batch,
                #     )
                #     if n == 5:
                #         break

                tepoch.set_postfix(loss=running_loss / (batch + 1), eig_mse=eig_mse)

    if epoch % 25 == 0:
        rtr = torch.matmul(
            student_model.fishleg_aux["R"].T, student_model.fishleg_aux["R"]
        )
        llt = torch.matmul(
            student_model.fishleg_aux["L"],
            student_model.fishleg_aux["L"].T,
        )
        A = student_model.fishleg_aux["A"].squeeze()

        Q = torch.kron(rtr, torch.diag(A) @ llt @ torch.diag(A))
        final_diag = torch.diag(Q).detach().numpy()

        print(final_diag)

        ax.plot(sorted(final_diag), sorted(target_diag), ".")

fig.savefig("test.png")
