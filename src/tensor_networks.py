# import torch
# import numpy as np

# class MLPLegendreKronDiag():
#     '''
#     Diagonal version of the Kronecker approximation in `F_convex`

#     '''
#    def __init__(self, layer_sizes):
#     self.layer_sizes = layer_sizes

#    def init_lam(self,scale):
#     def initialize_layer(m,n):
#         L=torch.eye(m +1, requires_grad=True)*np.sqrt(scale)
#         R=torch.eye(n, requires_grad=True)*np.sqrt(scale)
#         return L,R

#     return [initialize_layer(m,n) for (m,n) in zip(self.layer_sizes[:-1], self.layer[1:])]

#     def ef(self,lam,u):
#         """
#         F
#         """
#         F=0.0
#         for _lam, _u, in zip(lam,u):
#             L,R = _lam
#             F+=np.sum(np.square(torch.matmul(torch.matmul(L,_u),R.T)))
#         return 0.5*F

#     def dF_du(self,lam,u):
#         """
#         Derivate of F w.r.t u following b7 for diagoanl convex case
#         """
#         w_dict={}
#         for k in lam:
#             w_update =[]
#             if k == "net":
#                 for _lam,_u in zip(lam[k], u[k]):
#                     L,R = _lam
#                     w_update.append(torch.linalg.multi_dot(L.T,L,_u,R.T,R))
#             elif k == 'lik':
#                 w_dic[k]=u[k]*lam[k]**2
#             else:
#                 raise NotImplementedError
#         return w_dict
