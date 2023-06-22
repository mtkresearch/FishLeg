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

A = torch.randn((N, N))

U, _ = torch.linalg.qr(A)

lambda_i = []
for i in range(1, 101):
    lambda_i.append(1 / (i**2))

Lambda = torch.diag(torch.Tensor(lambda_i))

F = U.T @ Lambda @ U

teacher_model = nn.Linear(N, 1)

targets = torch.svd(1 / (F + gamma))[1]


def dataloader(batch_size: int = 1):
    while True:
        z = torch.Tensor(np.random.normal(0, 1, size=(batch_size, N))).T
        x = torch.matmul(torch.matmul(U, torch.sqrt(Lambda)), z)
        yield x, teacher_model(x.T).T


for K in [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 100000]:
    loader = dataloader(batch_size=K)

    x, y = next(loader)

    F_app = torch.matmul(x, x.T) / K

    eig_app = torch.svd(F_app)[1]


writer = SummaryWriter(
    log_dir=f"runs/tests/{datetime.now().strftime('%Y%m%d-%H%M%S')}",
)

loader = dataloader(batch_size=100)

student_model = FishLinear(N, 1, init_scale=1 / gamma)

# student_model.weight.data = teacher_model.weight.data

likelihood = FISH_LIKELIHOODS["gaussian"](sigma=1.0)

opt = FishLeg(
    student_model,
    loader,
    likelihood,
    lr=0.0001,
    beta=0.7,
    weight_decay=1e-5,
    aux_lr=0.001,
    aux_betas=(0.9, 0.9),
    aux_eps=1e-8,
    warmup_steps=0,
    damping=gamma,
    update_aux_every=10,
    method="antithetic",
    eps=1e-4,
    writer=writer,
)


rrt = torch.matmul(student_model.fishleg_aux["R"], student_model.fishleg_aux["R"].T)
llt = torch.matmul(
    student_model.fishleg_aux["L"][:N, :N], student_model.fishleg_aux["L"][:N, :N].T
)
A = student_model.fishleg_aux["A"][:, :N].squeeze()

target_diag = np.diag(1 / (Lambda + gamma))

L = student_model.fishleg_aux["L"][:N, :N]
R = student_model.fishleg_aux["R"][:N, :N]
A = student_model.fishleg_aux["A"][:, :N]

Q = (
    U.T
    @ torch.diag(A.squeeze(0))
    @ torch.kron(L @ L.T, R.T @ R)
    @ torch.diag(A.squeeze(0))
    @ U
)
final_diag = torch.diag(Q).detach().numpy()

fig, ax = plt.subplots(1, 1)
ax.plot(sorted(final_diag), sorted(target_diag), ".")
ax.plot(sorted(target_diag), sorted(target_diag), ls="--", color="k")

for epoch in range(1, 101):
    with tqdm(loader, unit="batch") as tepoch:
        running_loss = 0
        tepoch.set_description(f"Epoch {epoch}")
        for batch in range(100):
            opt.zero_grad()

            x, y = next(loader)

            pred_y = student_model(x.T)

            loss = likelihood(pred_y, y)

            loss.backward()

            opt.step()

            running_loss += loss.item()

            if batch % 50 == 0:
                # Write out the losses per epoch
                writer.add_scalar(
                    "Loss/train",
                    running_loss / (batch + 1),
                    (epoch * 100) + batch,
                )

                rrt = torch.matmul(
                    student_model.fishleg_aux["R"], student_model.fishleg_aux["R"].T
                )
                llt = torch.matmul(
                    student_model.fishleg_aux["L"][:N, :N],
                    student_model.fishleg_aux["L"][:N, :N].T,
                )
                A = student_model.fishleg_aux["A"][:, :N].squeeze()
                F_inv = torch.matmul(llt, torch.diag(A) ** 2)

                eig_app = torch.svd(F_inv)[1]
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
        target_diag = np.diag(1 / (Lambda + gamma))

        L = student_model.fishleg_aux["L"][:N, :N]
        R = student_model.fishleg_aux["R"][:N, :N]
        A = student_model.fishleg_aux["A"][:, :N]

        Q = (
            U.T
            @ torch.diag(A.squeeze(0))
            @ torch.kron(L @ L.T, R.T @ R)
            @ torch.diag(A.squeeze(0))
            @ U
        )
        final_diag = torch.diag(Q).detach().numpy()

        ax.plot(sorted(final_diag), sorted(target_diag), ".")
        ax.plot(sorted(target_diag), sorted(target_diag), ls="--", color="k")

fig.savefig("test.png")
