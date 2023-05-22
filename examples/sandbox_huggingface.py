import torch
import torch.nn as nn
import numpy as np

from transformers import AutoModel, AutoTokenizer
from optim.FishLeg import FishLeg, FishModel, GaussianLikelihood

print("hello FishLeg")


class Model(nn.Sequential, FishModel):
    def __init__(self, data, likelihood, *args):
        super(Model, self).__init__(*args)
        self.data = data
        self.N = data[0].shape[0]
        self.likelihood = likelihood


if __name__ == "__main__":

    model = AutoModel.from_pretrained("bert-base-uncased")
    tknz = AutoTokenizer.from_pretrained("bert-base-uncased")
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    opt = FishLeg(
        model,
        lr=1e-2,
        eps=1e-4,
        aux_K=5,
        update_aux_every=-3,
        aux_scale_init=1,
        aux_lr=1e-3,
        aux_betas=(0.9, 0.999),
        aux_eps=1e-8,
    )

    inputs = tknz("hello fishleg huggingface", return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    rnd = torch.rand(last_hidden_states.shape)
    opt.zero_grad()
    loss = torch.sum((last_hidden_states - rnd) ** 2)
    print(loss)

    loss.backward()
    opt.step()
