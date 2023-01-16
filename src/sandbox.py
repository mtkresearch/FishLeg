import torch
import torch.nn as nn
import numpy as np

from optim.FishLeg import FishLeg, FishModel, GaussianLikelihood


class Model(nn.Sequential, FishModel):
    def __init__(self, data, likelihood, *args):
        super(Model, self).__init__(*args)
        self.data = data
        self.N = data[0].shape[0]
        self.likelihood = likelihood


if __name__ == "__main__":

    x1 = np.linspace(-2.0, 2.0, 128)
    x2 = np.linspace(-2.0, 2.0, 128)
    xx1, xx2 = np.meshgrid(x1, x2)
    x = np.concatenate([xx1.reshape(-1, 1), xx2.reshape(-1, 1)], axis=-1)
    y = x[:, 0] ** 2 - x[:, 0] * 2 + x[:, 1] ** 3 + np.random.normal(size=(128**2,))

    x = torch.from_numpy(x).to(torch.float32)
    y = torch.from_numpy(y[:, None]).to(torch.float32)

    data = (x, y)
    likelihood = GaussianLikelihood(sigma_init=1.0, sigma_fixed=True)

    model = Model(
        (x, y),
        GaussianLikelihood(sigma_init=1.0, sigma_fixed=True),
        nn.Linear(2, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
    )

    opt = FishLeg(
        model,
        lr=1e-2,
        eps=1e-4,
        aux_K=5,
        update_aux_every=-3,
        aux_lr=1e-3,
        aux_betas=(0.9, 0.999),
        aux_eps=1e-8,
    )

    for iteration in range(100):
        opt.zero_grad()
        pred_y = model(x)
        loss = nn.MSELoss()(y, pred_y)
        loss.backward()
        opt.step()
        if iteration % 10 == 0:
            print(loss.detach())
