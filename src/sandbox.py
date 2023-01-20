import torch
import torch.nn as nn
import numpy as np

from optim.FishLeg import FishLeg, FixedGaussianLikelihood


if __name__ == "__main__":

    x1 = np.linspace(-2.0, 2.0, 1000)
    x2 = np.linspace(-2.0, 2.0, 1000)
    xx1, xx2 = np.meshgrid(x1, x2)
    x = np.concatenate([xx1.reshape(-1, 1), xx2.reshape(-1, 1)], axis=-1)
    y = x[:, 0] ** 2 - x[:, 0] * 2 + x[:, 1] ** 3 + np.random.normal(size=(1000**2,))

    x = torch.from_numpy(x).to(torch.float32)
    y = torch.from_numpy(y[:, None]).to(torch.float32)

    train_data = []
    for i in range(len(x)):
        train_data.append([x[i], y[i]])

    auxloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=100)
    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=100)

    likelihood = FixedGaussianLikelihood(sigma_fixed=1.0)

    def nll(model, data):
        data_x, data_y = data
        pred_y = model.forward(data_x)
        return likelihood.nll(data_y, pred_y)

    def draw(model, data_x):
        pred_y = model.forward(data_x)
        return (data_x, likelihood.draw(pred_y))

    def dataloader():
        data_x, _ = next(iter(auxloader))
        return data_x


    model = nn.Sequential(
        nn.Linear(2, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
    )

    opt = FishLeg(
        model,
        draw,
        nll,
        dataloader,
        lr=1e-2,
        eps=1e-4,
        update_aux_every=10,
        aux_lr=1e-3,
        aux_betas=(0.9, 0.999),
        aux_eps=1e-8,
    )

    for iteration in range(100):
        data_x, data_y = next(iter(trainloader))
        opt.zero_grad()
        pred_y = model(data_x)
        loss = nn.MSELoss()(data_y, pred_y)
        loss.backward()
        opt.step()
        if iteration % 10 == 0:
            print(loss.detach())
