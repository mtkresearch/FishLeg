import jax
import math
import jax.numpy as jnp
import jax.random as random
from functools import partial

class MLP():
    def __init__(self, activation_funs, layer_sizes, init_scale=1):
        assert(len(activation_funs) == len(layer_sizes)-1)
        self.activation_funs = activation_funs
        self.layer_sizes = layer_sizes
        self.init_scale = init_scale
        
    def init_theta(self, key):
        keys = random.split(key, len(self.layer_sizes))
        # Initialize a single layer with Gaussian weights -  helper function
        def initialize_layer(m, n, key):
            scale = self.init_scale/math.sqrt(m)
            w = scale * random.normal(key, (m, n))
            b = jnp.zeros((1,n))
            return jnp.vstack((b, w))
        return [initialize_layer(m, n, k) \
                for (m, n, k) \
                in zip(self.layer_sizes[:-1], self.layer_sizes[1:], keys)]

    @partial(jax.jit, static_argnums=(0,))
    def forward(self, theta, x):
        accu = x
        for i, p in enumerate(theta):
            wb = p
            activation_fun = self.activation_funs[i]
            b = wb[0,:]
            w = wb[1:,:]
            # note the convention: activity samples are packed as rows of accu
            accu = b + accu @ w
            if activation_fun == "relu":
                accu = jax.nn.relu(accu)
            elif activation_fun == "tanh":
                accu = jnp.tanh(accu)
            elif activation_fun == "linear":
                pass
            else:
                raise TypeError("Activation function " + activation_fun + " not implemented") 
        return accu

    def layer_sizes(self):
        return self.layer_sizes
