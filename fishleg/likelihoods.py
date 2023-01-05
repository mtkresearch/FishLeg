import torch
import numpy as np
from utils import dense_to_one_hot
#Note, need to check that the recuction of the nll is correct, default reducation is mean
class GaussianLikelihood():
    def __init__(self, sigma_init, sigma_fixed):
        self.sigma_init 	= sigma_init
        self.sigma_fixed 	= sigma_fixed 

    def init_theta(self):
        return self.sigma_init

    def init_lam(self, scale):
        return scale
    
    def nll(self, theta, y_pred,y):
        """
        Negative Log Likelihood Function
        params:
            theta 		: sigma (standard deviation of the conditional density)
            y_pred 	: y predicted by the model
            y 		: y_true
        """
        sigma=theta
        if self.sigma_fixed:
            sigma = self.sigma_init
        return 0.5* (torch.nn.MSELoss()(y, y_pred) / sigma + np.log(sigma**2)/y_pred.shape[0])

    def sample(self,theta,y_pred):
        """
        Sample from model's conditional density
        params:
            theta 		: sigma (standard deviation of the conditional density)
            y_pred 	: y predicted by the model

        """
        sigma=theta
        if self.sigma_fixed:
            sigma=self.sigma_init
        return y_pred + torch.normal(0,sigma,y_pred.shape)

    def ef(self, lam,u):
        return 0.5*((lam*u)**2)

    
class BernoulliLikelihood():
    def __init__(self):
        pass
    
    def init_theta(self):
        return 1.0

    def init_lam(self, scale):
        return scale

    def nll(self,theta, y_pred,y):
        return torch.nn.BCEWithLogitsLoss(y, y_pred)

    def sample(self,theta,y_pred):
        #Check this sampler
        pred_dist = torch.normal(y_pred,1.0)
        return 1.0 *torch.bernoulli(pred_dist) 

class SoftMaxLikelihood():
    def __init__(self):
        pass
    
    def init_theta(self):
        return 0.0

    def init_lam(self, scale):
        #return scale (this needs to be a torch tensor to calculate gradients)
        return torch.tensor(scale, requires_grad = True)

    def nll(sef,theta, y_pred,y):
        logits = torch.nn.functional.log_softmax(y_pred, dim=1)
        return -1.0*np.mean(np.sum(logits * y, axis=1)) 
    
    def sample(self,theta,y_pred):
        logits = np.log(y_pred)
        return dense_to_one_hot(torch.distributions.Categorical(logits=logits)) 
    
    def ef(self, lam,u):
        return 0.0*(lam*u)


    

