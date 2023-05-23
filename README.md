<h1>FishLeg</h1>

<h3>Repository containing a PyTorch implementation of the Fisher-Legendre Second Order Optimization method. </h3>

<p>
    <a href="https://mtkresearch.github.io/FishLeg/">
        <img alt="Documentation" src="https://img.shields.io/badge/documentation-informational?&style=for-the-badge&logo=read-the-docs" height=25>
    </a>
    <a href="https://github.com/mtkresearch/FishLeg/issues">
        <img src="https://img.shields.io/badge/technical%20support-red?style=for-the-badge&logo=github" height=25>
    </a>
    <a href="https://github.com/mtkresearch/FishLeg/issues">
        <img src="https://img.shields.io/badge/release-v1.0-blue?style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/mtkresearch/FishLeg/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/badge/License-apache--2.0-red?style=for-the-badge" height=25>
    </a>
</p>

## Installation
FishLeg is written in Python, and only requires PyTorch > 1.8.<br />
The example scripts have additional requirements defined in [src/optim/requirements.txt](src/optim/requirements.txt)<br />
The FishLeg library does not requires dedicated installation. <br />

## Usage
FishLeg requires minimal code modifications to introduce it in existing training scripts. 
```Python
from optim.FishLeg import FishLeg, FISH_LIKELIHOODS

...
likelihood = FISH_LIKELIHOODS["FixedGaussian".lower()](sigma=1.0, device=device)

    def nll(model, data_x, data_y):
        pred_y = model.forward(data_x)
        return likelihood.nll(data_y, pred_y)

    def draw(model, data_x):
        pred_y = model.forward(data_x)
        return likelihood.draw(pred_y)



...

model = nn.Sequential(...).to(device)
optimizer =  opt = FishLeg(
        model,
        draw,
        nll,
        aux_loader,
        lr=eta_fl,
        eps=eps,
        beta=beta,
        weight_decay=1e-5,
        update_aux_every=10,
        aux_lr=aux_eta,
        aux_betas=(0.9, 0.999),
        aux_eps=1e-8,
        damping=damping,
        pre_aux_training=25,
        sgd_lr=eta_sgd,
        device=device,
    )

...
```
In case of pruning/fine-tuning, where a pretrained model is given, one would wish to pretrain the auxiliary parameters by calling the function `pretrain_fish()`

```python
aux_losses = optimizer.pretrain_fish(
        dataloader,
        nll,
        output_dir,
        iterations = 2000,
        batch_size = 16,
        difference = True,
        fisher = True
    )
```

