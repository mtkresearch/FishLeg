import jax
import jax.numpy as jnp
from functools import partial

@jax.jit
def prod3(a, b, c):
    n = a.shape[0]
    m = c.shape[0]
    if (n*n*m)+(m**3) < (m*m*n)+(n**3):
        return (a@b)@c
    else:
        return a@(b@c)

class MLPLegendreKronDiag():
    ''' 
    Diagonal version of the Kronecker approximation in `F_convex`

    '''
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
    
    def init_lam(self, scale):
        def initialize_layer(m, n):
            L = jnp.sqrt(scale)*jnp.eye(m+1)
            R = jnp.sqrt(scale)*jnp.eye(n)
            return L, R
        return [ (initialize_layer(m, n)) for (m, n) in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]

    @partial(jax.jit, static_argnums=(0,))
    def ef(self, lam, u):
        '''
        F 
        '''
        F = 0.0      
        for _lam, _u in zip(lam, u):
            L, R = _lam
            F += jnp.sum(jnp.square(L @ _u @ R.T))
        return 0.5*F

    @partial(jax.jit, static_argnums=(0,))
    def dF_du_direct(self, lam, u):
        '''
        F but following b7 for diagonal convex case
        '''
        w_dic={}
        for k in lam:
            w_update = []
            if k=="net":
                for _lam, _u in zip(lam[k], u[k]):
                    L, R = _lam
                    w_update.append(prod3(L.T, prod3(L, _u, R.T), R))
                w_dic[k] = w_update
            elif k=='lik': 
                w_dic[k] = u[k]*lam[k]**2
            else:
                raise NotImplementedError
        return w_dic