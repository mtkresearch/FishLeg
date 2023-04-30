<h1>FishLeg</h1>

<h3>Repository containing the official PyTorch implementation of the Fisher-Legendre Second Order Optimization method. </h3>

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
 <!--    [![arxiv.org](http://img.shields.io/badge/cs.CV-arXivnumber.svg)](https://openreview.net/pdf?id=c9lAOPvQHS)
    <a> -->
    </a>
</p>

## Overview
This library contains the official PyTorch implementation of the FishLeg optimizer as introduced in [Fisher-Legendre (FishLeg) optimization of deep neural networks ](https://openreview.net/pdf?id=c9lAOPvQHS).<br />
FishLeg is a learnt second-order optimization method that uses natural gradients and ideas from Legendre-Fenchel duality to learn a direct and efficiently evaluated model for the product of the inverse Fisher with any vector in an online manner. Thanks to its generality, we expect FishLeg to facilitate handling various neural network architectures. The library's primary goal is to provide researchers and developers with an easy-to-use implementation of the FishLeg optimizer and curvature estimator.
## Installation
FishLeg is written in pure Python, and only requires PyTorch > 1.8.<br />
The example scripts have additional requirements defined in [examples/requirments.py](examples/requirments.py)<br />
The FishLeg library do not requires dedicated installation. <br />

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



See [examples/autoencoder.py](examples/autoencoder.py) for an usage demostration. <br />
See the FishLeg documentation for a detailed list of parameters.
 

## Citation
```
@article{garcia2022FishLeg,
  title={Fisher-Legendre (FishLeg) optimization of deep neural networks},
  author={Garcia, Jezabel R and Freddi, Federica and Fotiadis, Stathi1 and Li, Maolin and Vakili, Sattar, and Bernacchia, Alberto and Hennequin,Guillaume },
  journal={},
  year={2023}
}
```

## Contributing
