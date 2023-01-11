import torch
import torch.nn as nn
import numpy as np

from transformers import AutoModel, AutoTokenizer
from fishleg import FishLeg
from likelihoods import GaussianLikelihood

print('hello FishLeg')


class Model(nn.Sequential):
    def __init__(self, data, likelihood, *args):
        super(Model, self).__init__(*args)
        self.data = data
        self.N = data[0].shape[0]
        self.likelihood = likelihood

    def nll(self, data):
        data_x, data_y = data
        pred_y = self.forward(data_x)
        return self.likelihood.nll(None, pred_y, data_y)

    def sample(self, K):
        data_x = self.data[0][np.random.randint(0,self.N,K)]
        pred_y = self.forward(data_x)
        return (data_x, self.likelihood.sample(None, pred_y))

if __name__ == '__main__':

    #x1 = np.linspace(-2.,2.,128)
    #x2 = np.linspace(-2.,2.,128)
    #xx1,xx2 = np.meshgrid(x1,x2)
    #x = np.concatenate([xx1.reshape(-1,1),xx2.reshape(-1,1)],axis=-1)
    #y = x[:,0]**2-x[:,0]*2+x[:,1]**3 + np.random.normal(size=(128**2,))

    #x = torch.from_numpy(x).to(torch.float32)
    #y = torch.from_numpy(y[:,None]).to(torch.float32)

    #data = (x,y)
    #likelihood = GaussianLikelihood(sigma_init=1.0, sigma_fixed=True)

    model = AutoModel.from_pretrained("bert-base-uncased")
    tknz = AutoTokenizer.from_pretrained("bert-base-uncased")
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    #model = Model(data, likelihood, 
    #            nn.Linear(2, 5), nn.ReLU(), nn.Linear(5,1), nn.ReLU())

    opt = FishLeg(model, lr=1e-2, eps=1e-4, aux_K=5, 
                    update_aux_every=-3, aux_scale_init=1, 
                    aux_lr=1e-3, aux_betas=(0.9, 0.999), 
                    aux_eps=1e-8)

    inputs = tknz("hello fishleg huggingface", return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    print(last_hidden_states.shape)
    rnd = torch.rand(last_hidden_states.shape)
    print(rnd.shape)
    opt.zero_grad()
    loss = torch.sum((last_hidden_states - rnd)**2)
    print(loss)
    
    loss.backward()
    opt.step()

    
