import torch
import torch.nn as nn
import numpy as np

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

    x1 = np.linspace(-2.,2.,32)
    x2 = np.linspace(-2.,2.,32)
    xx1,xx2 = np.meshgrid(x1,x2)
    x = np.concatenate([xx1.reshape(-1,1),xx2.reshape(-1,1)],axis=-1)
    y = x[:,0]**2-x[:,0]*2+x[:,1]**3 + np.random.normal(size=(1024,))

    x = torch.from_numpy(x).to(torch.float32)
    y = torch.from_numpy(y[:,None]).to(torch.float32)

    data = (x,y)
    likelihood = GaussianLikelihood(sigma_init=1.0, sigma_fixed=True)

    model = Model(data, likelihood, 
                nn.Linear(2, 5), nn.ReLU(), nn.Linear(5,1), nn.ReLU())

    opt = FishLeg(model)

    for i in range(10):
        opt.zero_grad()
        pred_y = model(x)
        loss = nn.MSELoss()(y,pred_y)
        loss.backward()
        opt.step()
        print(loss)