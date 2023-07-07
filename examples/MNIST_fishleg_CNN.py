import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import time
import os
import sys
import matplotlib.pyplot as plt
import torch.optim as optim
from datetime import datetime
from utils import class_accuracy
from torch.utils.tensorboard import SummaryWriter

from data_utils import read_data_sets

torch.set_default_dtype(torch.float32)

sys.path.append("../src")

from optim.FishLeg import FishLeg, FISH_LIKELIHOODS, initialise_FishModel

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

seed = 13
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

dataset = read_data_sets("MNIST", "../data/", if_autoencoder=False, reshape=False)

## Dataset
train_dataset = dataset.train
test_dataset = dataset.test

batch_size = 500

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

aux_loader = torch.utils.data.DataLoader(
    train_dataset, shuffle=True, batch_size=batch_size
)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

model = nn.Sequential(
    nn.Conv2d(
        in_channels=1,
        out_channels=16,
        kernel_size=5,
        stride=1,
        padding=2,
    ),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    # nn.Conv2d(
    #     in_channels=16,
    #     out_channels=32,
    #     kernel_size=5,
    #     stride=1,
    #     padding=2,
    # ),
    # nn.ReLU(),
    # nn.MaxPool2d(kernel_size=2),
    nn.Flatten(),
    nn.Linear(16 * 14 * 14, 10),
)

lr = 0.02
beta = 0.9
weight_decay = 1e-5

aux_lr = 1e-4
aux_eps = 1e-8
scale = 2
damping = 0.5
update_aux_every = 3

initialization = "normal"
normalization = True

model = initialise_FishModel(model, module_names="__ALL__", fish_scale=scale)

model = model.to(device)

likelihood = FISH_LIKELIHOODS["softmax"](device=device)

writer = SummaryWriter(
    log_dir=f"runs/MNIST_fishleg_CNN/lr={lr}_lambda={weight_decay}/{datetime.now().strftime('%Y%m%d-%H%M%S')}",
)

opt = FishLeg(
    model,
    aux_loader,
    likelihood,
    lr=lr,
    beta=beta,
    weight_decay=weight_decay,
    aux_lr=aux_lr,
    aux_betas=(0.9, 0.999),
    aux_eps=aux_eps,
    damping=damping,
    update_aux_every=update_aux_every,
    writer=writer,
    method="antithetic",
    method_kwargs={},
    precondition_aux=True,
)

epochs = 100

st = time.time()
eval_time = 0

for epoch in range(1, epochs + 1):
    with tqdm(train_loader, unit="batch") as tepoch:
        running_loss = 0
        running_acc = 0
        for n, (batch_data, batch_labels) in enumerate(tepoch, start=1):
            tepoch.set_description(f"Epoch {epoch}")

            batch_data, batch_labels = (
                batch_data.to(device),
                batch_labels.to(device),
            )

            opt.zero_grad()
            output = model(batch_data)

            loss = likelihood(output, batch_labels)

            running_loss += loss.item()

            running_acc += class_accuracy(output, batch_labels).item()

            loss.backward()
            opt.step()

            et = time.time()
            if n % 50 == 0:
                model.eval()

                running_test_loss = 0
                running_test_acc = 0

                for m, (test_batch_data, test_batch_labels) in enumerate(
                    test_loader, start=1
                ):
                    test_batch_data, test_batch_labels = test_batch_data.to(
                        device
                    ), test_batch_labels.to(device)

                    test_output = model(test_batch_data)

                    test_loss = likelihood(test_output, test_batch_labels)

                    running_test_loss += test_loss.item()

                    running_test_acc += class_accuracy(
                        test_output, test_batch_labels
                    ).item()

                running_test_loss /= m
                running_test_acc /= m

                tepoch.set_postfix(
                    acc=100 * running_acc / n, test_acc=running_test_acc * 100
                )
                model.train()
                eval_time += time.time() - et

        epoch_time = time.time() - st - eval_time

        tepoch.set_postfix(
            loss=running_loss / n, test_loss=running_test_loss, epoch_time=epoch_time
        )
        # Write out the losses per epoch
        writer.add_scalar("Acc/train", 100 * running_acc / n, epoch)
        writer.add_scalar("Acc/test", 100 * running_test_acc, epoch)

        # Write out the losses per epoch
        writer.add_scalar("Loss/train", running_loss / n, epoch)
        writer.add_scalar("Loss/test", running_test_loss, epoch)

        # Write out the losses per wall clock time
        writer.add_scalar("Loss/train/time", running_loss / n, epoch_time)
        writer.add_scalar("Loss/test/time", running_test_loss, epoch_time)
