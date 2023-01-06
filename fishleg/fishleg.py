import torch
import torch.nn as nn
import copy
from torch.optim import Optimizer, Adam
from torch.nn import ParameterDict, Parameter


class fishLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None):
        super(fishLinear, self).__init__(in_features, out_features, bias, device=device, dtype=dtype)
        self.fishleg_aux = ParameterDict({
            'L': Parameter(torch.eye(in_features +1)),
            'R': Parameter(torch.eye(out_features)),
        })
        self.order = ['weight', 'bias']
        
    @staticmethod
    def Qv(aux: dict, v: list):
        L, R = aux['fishleg_aux.L'], aux['fishleg_aux.R']
        u = torch.cat([v[0], v[1][:,None]], dim=-1)
        z = torch.linalg.multi_dot((R,R.T, u, L,L.T))
        return [z[:,:-1], z[:,-1]]

    def cuda(self, device):
        super.cuda(device)
        L, R = self.fishleg_aux
        L.to(device)
        R.to(device)

def update_dict(replace, module):
    replace_dict = replace.state_dict()
    pretrained_dict = {k:v for k,v in module.state_dict().items() if k in replace_dict}
    replace_dict.update(pretrained_dict)
    replace.load_state_dict(replace_dict)
    return replace

class FishLeg(Optimizer):
    def __init__(self, model, lr=1e-2, eps=1e-2, aux_K=2, update_aux_every=1, aux_scale_init=1, aux_lr=1e-3, aux_betas=(0.9, 0.999), aux_eps=1e-8):
        self.model = model
        self.aux_model = self.__init_model_aux(model)

        self.aux_param, self.aux_name = [],[]
        for name,param in self.aux_model.named_parameters():
            if "fishleg_aux" in name:
                self.aux_param.append(param)
                self.aux_name.append(name)

        param_groups = []
        for module_name, module in self.aux_model.named_modules():
            if hasattr(module, "fishleg_aux"):
                params = {name:param for name, param in self.model._modules[module_name].named_parameters() 
                            if 'fishleg_aux' not in name}
                g = {
                    'params': [params[name] for name in module.order],
                    'aux_params': {name:param for name, param in module.named_parameters() 
                                    if 'fishleg_aux' in name},
                    'Qv': module.Qv,
                    'order': module.order,
                    'name': module_name
                }
                param_groups.append(g)
        #TODO: add param_group for modules without aux
        defaults = dict(lr=lr, 
                        aux_scale_init=aux_scale_init)

        super(FishLeg, self).__init__(param_groups, defaults)
        self.aux_opt = Adam(self.aux_param, lr=aux_lr, betas=aux_betas, eps=aux_eps)
        self.eps = eps
        self.aux_K = aux_K
        self.update_aux_every = update_aux_every # if negative, then run -update_aux_every aux updates in each outer iteration  
        self.aux_scale_init = aux_scale_init
        self.aux_lr = aux_lr
        self.aux_betas = aux_betas
        self.aux_eps = aux_eps
        self.step_t = 0
        self.z = copy.deepcopy(model) # will host eps*Qg in the aux update
        self.tmp = copy.deepcopy(model) # will host theta ± z in the aux update 
        

    def __init_model_aux(self, model):
        aux_model = copy.deepcopy(model)
        for name, module in aux_model.named_modules():
            if isinstance(module, nn.Linear):
                replace = fishLinear(
                    module.in_features,
                    module.out_features,
                    module.bias is not None
                )
                replace = update_dict(replace, module)
                aux_model._modules[name] = replace
        return aux_model


    def update_aux(self):
        # if z = eps*Qg, then
        # the auxiliary loss is equal to nll(theta+z) + nll(theta-z) - 2*sum(z*g)
        # we're going to compute the gradient w.r.t. z of each of these 3 terms separately
        # they will be summed as we go along

        self.aux_opt.zero_grad()
        
        data = self.model.sample(self.aux_K)

        # first, compute z once and for all

        for group in self.param_groups:
            name = group['name']
            Qv = group['Qv']
            aux_params = group['aux_params']
            order = group['order']
            aux_scale_init = group['aux_scale_init']
            z = self.z._modules[name]
            grad = []
            for p in group['params']:
                grad.append(p.grad.data)
            
            qg = (self.eps * aux_scale_init) * Qv(aux_params, grad)
            for gi, qgi, para_name, thetai in zip(grad, qg, order, group['params']):
                zi = z._parameters[para_name]
                torch._copy_from(input=qgi.data, dst=zi.data)

        # next compute dNLL(theta+z)/dz in tmp.grad, initially set to zero
        for group in self.param_groups:
            name = group['name']
            order = group['order']
            z = self.z._modules[name]
            tmp = self.tmp._modules[name]
            for para_name, thetai in zip(order, group['params']):
                zi = z._parameters[para_name]
                tmpi = tmp._parameters[para_name]
                tmpi.zero_grad()
                torch._copy_from(input=thetai.data, dst=tmpi.data) 
                tmpi.add_(zi) # tmp = theta + z at this point

        h = self.tmp.nll(data)
        h.backward()

        # next add dNLL(theta-z)/dz to tmp.grad
        for group in self.param_groups:
            name = group['name']
            order = group['order']
            z = self.z._modules[name]
            tmp = self.tmp._modules[name]
            for para_name, thetai in zip(order, group['params']):
                zi = z._parameters[para_name]
                tmpi = tmp._parameters[para_name]
                torch._copy_from(input=thetai.data, dst=tmpi.data) 
                tmpi.add_(zi, alpha=-1.) # tmp = theta - z at this point

        h = -self.tmp.nll(data)
        h.backward()

        # add d(-2*dot(g,z))/dz to tmp.grad
        third_term = 0
        for group in self.param_groups:
            name = group['name']
            order = group['order']
            z = self.z._modules[name]
            tmp = self.tmp._modules[name]
            for para_name, gi in zip(order, grad):
                zi = z._parameters[para_name]
                tmpi = tmp._parameters[para_name]
                torch._copy_from(input=zi.data, dst=tmpi.data)
                third_term += -2*sum(tmpi*gi)

        third_term.backward()

        # finally, backprop towards the aux parameters
        for group in self.param_groups:
            name = group['name']
            order = group['order']
            z = self.z._modules[name]
            for para_name in order:
                zi = z._parameters[para_name]
                tmpi = tmp._parameters[para_name]
                tmpi.grad.mul_(1/self.eps**2)
                zi.backward(tmpi.grad)

        # now aux_opt has the right gradient
        self.aux_opt.step()

        
    def step(self):
        self.step_t += 1
        self.plus_model = copy.deepcopy(self.model)
        self.minus_model = copy.deepcopy(self.model)
        
        if self.update_aux_every > 0:
            if self.step_t % self.update_aux_every == 0:
                self.update_aux()
        elif self.update_aux_every < 0:
            for _ in range(-self.update_aux_every):
                self.update_aux()
        for group in self.param_groups:
            lr = group['lr']
            aux_scale_init=group['aux_scale_init']
            Qv = group['Qv']
            aux_params = group['aux_params']

            if 'aux_params' in group.keys():
                grad = []
                for p in group['params']:
                    grad.append(p.grad.data)
                qg = Qv(aux_params, grad)

                for p, d_p in zip(group['params'], qg):
                    p.data.add_(d_p, alpha=-lr*aux_scale_init)