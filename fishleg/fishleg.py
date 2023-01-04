import torch
import math
import numpy as np
import random as random
import torch.autograd.functional.jvp as jvp
from torch.autograd import grad

class FishLeg():
    def __init__(self, net, netleg, lik, nu_theta, K=1, damping=None):
        self.net = net
        self.netleg = netleg
        self.lik = lik
        self.K = K
        self.nu_theta = nu_theta
        self.damping = damping
                       
    def init_theta(self, key):
        return { 'net': self.net.init_theta(), 
                 'lik': self.lik.init_theta() }
    
    def init_lam(self, scale):
        return { 'net': self.netleg.init_lam(scale),
                 'lik': self.lik.init_lam(scale) }
    

    def ef(self, lam, u):
        return self.netleg.ef(lam['net'], u['net']) + self.lik.ef(lam['lik'], u['lik'])


    def dF_du(self, lam, u):
        return self.netleg.dF_du(lam, u)

    def hessian_v_inner(self, theta, D, z):
        data_x, data_y = D
        _, tangent = jvp(self.ell, (theta, D), (z, (0*data_x, 0*data_y)))
        return tangent
    
    def hessian_v(self, theta, D, z):
        #need to add dumping?
       out= self.hessian_v_inner(theta, D, z)
       out.backward(torch.ones_like(theta))
       return theta.grad 

    def sample_model(self, theta, data_x, key):
        y_pred = self.net.forward(theta['net'], data_x)
        return (data_x, self.lik.sample(theta['lik'], y_pred, key))
    
    def ell(self, theta, D):
        return self.ell_without_reg(theta, D) + self.reg(theta)

    def ell_without_reg(self, theta, D):
        x, y = D
        ##Do i need to move to device here again?
        y_pred = self.net.forward(theta['net'], x)
        return self.lik.nll(theta['lik'], y_pred, y)

    def adam_update(self, lam, delta_lam, update_func, apply_update_func, adam_state):
        aux_params_update, new_adam_state = update_func(delta_lam, adam_state)
        new_lam = apply_update_func(lam, aux_params_update)
        return new_lam, new_adam_state
    
    

    def elisabeth_the_II(self, g_tilde, D, theta, lam):
        z = self.dF_du_direct(lam, g_tilde)
        #Z should have gradients stop by default, if not detach
        h = self.hessian_v(theta, D, z)
        d = h- g_tilde
        zd =  z.dot(d) 
        s = sum(zd)
        return s

    
    def update_aux(self, theta, g, lam, data_x, \
                   adam_state, aux_opt_get_update, aux_opt_apply_updates, \
                   key):
        aux_loss = 0
        g_tilde = self.normalize_gradient(g)
        for k in range(self.K):
            D = self.sample_model(theta, data_x)
            out = self.elisabeth_the_II(g_tilde, D, theta, lam)
            out.backward(torch.ones_like(lam['net']))
            delta_lam = lam['net'].grad            
            aux_loss += 0  # fix todo if you want to get the real aux_loss, currently not implemented
            lam, adam_state = self.adam_update(lam, delta_lam, \
                                               aux_opt_get_update, aux_opt_apply_updates,
                                               adam_state)
        return aux_loss / self.K, lam, adam_state
    
    def step(self, theta, g, lam):
        delta_theta = self.dF_du(lam,g)
        theta = theta- delta_theta
        return theta, delta_theta


