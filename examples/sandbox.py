import torch
import torch.nn as nn
import numpy as np
import time
import os
import gzip
import urllib.request
import jax.numpy as jnp
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset
import copy
import argparse

torch.set_default_dtype(torch.float32)

from optim.FishLeg import FishLeg, FISH_LIKELIHOODS





if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--exp", type=str, help="which dataset", default="MNIST")
    args = argparser.parse_args()

    seed = 13
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = None

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    ## Hyperparams
    if args.exp == "FACES":

        batch_size = 100
        epochs = 5
        eta_adam = 3e-5
        eta_fl = 0.05
        eta_sgd = 0.01
        aux_eta = 5e-4
        weight_decay = 1e-5
        beta = 0.9
        damping = 1.0

        dataset = read_data_sets("FACES", "../data/", if_autoencoder=True)

    if args.exp == "MNIST":
        batch_size = 100
        epochs = 10
        
        eta_adam = 1e-4

        fish_lr = 0.02
        beta = 0.9
        weight_decay = 1e-5
        update_aux_every = 10
        aux_lr = 2e-3
        aux_eps = 1e-8
        damping = 0.3
        pre_aux_training = 10
        scale = 1
        initialization = "normal"
        normalization = True
        batch_speedup = False
        fine_tune = False
        warmup = 0

        dataset = read_data_sets("MNIST", "../data/", if_autoencoder=True)

    ## Dataset
    train_dataset = dataset.train
    test_dataset = dataset.test
    if args.exp == "FACES":
        likelihood = FISH_LIKELIHOODS["fixedgaussian"](sigma=1.0, device=device)

        def mse(model, data):
            data_x, data_y = data
            pred_y = model.forward(data_x)
            return torch.mean(torch.square(pred_y - data_y))

    if args.exp == "MNIST":
        likelihood = FISH_LIKELIHOODS["bernoulli"](device=device)

        def mse(model, data):
            data_x, data_y = data
            pred_y = model.forward(data_x)
            pred_y = torch.sigmoid(pred_y)
            return torch.mean(torch.square(pred_y - data_y))

    def nll(model, data):
        data_x, data_y = data
        pred_y = model.forward(data_x)
        return likelihood.nll(data_y, pred_y)

    def draw(model, data):
        data_x, data_y = data
        pred_y = model.forward(data_x)
        return (data_x, likelihood.draw(pred_y))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    aux_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False
    )

    test_loader_adam = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False
    )

    if args.exp == "FACES":
        model = nn.Sequential(
            nn.Linear(625, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 30),
            nn.Linear(30, 500),
            nn.ReLU(),
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 625),
        ).to(device)

    if args.exp == "MNIST":
        model = nn.Sequential(
            nn.Linear(784, 1000, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(1000, 500, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(500, 250, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(250, 30, dtype=torch.float32),
            nn.Linear(30, 250, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(250, 500, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(500, 1000, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(1000, 784, dtype=torch.float32),
        ).to(device)

    model_adam = copy.deepcopy(model)

    # print("lr fl={}, lr sgd={}, lr aux={}".format(eta_fl, eta_sgd, aux_eta))

    opt = FishLeg(
        model,
        draw,
        nll,
        aux_loader,
        likelihood,
        fish_lr=fish_lr,
        beta=beta,
        weight_decay=weight_decay,
        update_aux_every=update_aux_every,
        aux_lr=aux_lr,
        aux_betas=(0.9, 0.999),
        aux_eps=aux_eps,
        damping=damping,
        pre_aux_training=pre_aux_training,
        initialization=initialization,
        device=device,
        batch_speedup=batch_speedup,
        scale=scale,
    )

    print(opt.__dict__["fish_lr"])
    print(opt.__dict__["beta"])
    print(opt.__dict__["aux_lr"])
    print(opt.__dict__["damping"])
    print(opt.__dict__["scale"])

    FL_time = []
    LOSS = []
    AUX_LOSS = []
    TEST_LOSS = []
    st = time.time()
    iteration = 0
    for e in range(1, epochs + 1):
        print("######## EPOCH", e)
        for n, (batch_data, batch_labels) in enumerate(train_loader, start=1):
            iteration += 1
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            opt.zero_grad()
            loss = nll(opt.model, (batch_data, batch_labels))
            loss.backward()
            opt.step()

            if n % 50 == 0:
                FL_time.append(time.time() - st)
                LOSS.append(loss.detach().cpu().numpy())
                AUX_LOSS.append(opt.aux_loss)

                test_batch_data, test_batch_labels = next(iter(test_loader))
                test_batch_data, test_batch_labels = test_batch_data.to(
                    device
                ), test_batch_labels.to(device)
                test_loss = mse(opt.model, (test_batch_data, test_batch_labels))

                TEST_LOSS.append(test_loss.detach().cpu().numpy())

                print(n, LOSS[-1], AUX_LOSS[-1], TEST_LOSS[-1])

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(FL_time, LOSS, label="Fishleg")  # color=colors_group[i])
    axs[1].plot(
        FL_time, TEST_LOSS, label="Fishleg"
    )  # linestyle='--', color=colors_group[i])

    opt = optim.Adam(
        model_adam.parameters(),
        lr=eta_adam,
        betas=(0.9, 0.9),
        weight_decay=weight_decay,
        eps=1e-4,
    )
    iteration = 0
    FL_time = []
    LOSS = []
    TEST_LOSS = []
    st = time.time()
    for e in range(1, epochs + 1):
        print("######## EPOCH", e)
        for n, (batch_data, batch_labels) in enumerate(train_loader):
            iteration += 1
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            opt.zero_grad()
            loss = nll(model_adam, (batch_data, batch_labels))
            loss.backward()
            opt.step()

            if n % 50 == 0:
                FL_time.append(time.time() - st)
                LOSS.append(loss.detach().cpu().numpy())
                test_batch_data, test_batch_labels = next(iter(test_loader_adam))
                test_batch_data, test_batch_labels = test_batch_data.to(
                    device
                ), test_batch_labels.to(device)
                test_loss = mse(model_adam, (test_batch_data, test_batch_labels))
                TEST_LOSS.append(test_loss.detach().cpu().numpy())

                print(n, LOSS[-1], TEST_LOSS[-1])

    axs[0].plot(FL_time, LOSS, label="Adam")
    axs[1].plot(FL_time, TEST_LOSS, label="Adam")

    axs[0].legend()
    axs[1].legend()

    axs[0].set_title("Training Loss")
    axs[1].set_title("Test MSE")
    fig.savefig("result/result.png", dpi=300)
