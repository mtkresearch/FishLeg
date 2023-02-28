import jax
import math
import jax.numpy as jnp
import jax.random as random
from functools import partial
from jax.flatten_util import ravel_pytree as ravel

class FishLeg():
    def __init__(self, net, netleg, lik, nu_theta, K=1, damping=None):
        self.net = net
        self.netleg = netleg
        self.lik = lik
        self.K = K
        self.nu_theta = nu_theta
        self.damping = damping
                       
    def init_theta(self, key):
        return { 'net': self.net.init_theta(key), \
                 'lik': self.lik.init_theta() }
    
    def init_lam(self, scale):
        return { 'net': self.netleg.init_lam(scale),
                 'lik': self.lik.init_lam(scale) }
    
    @partial(jax.jit, static_argnums=(0,))
    def ef(self, lam, u):
        return self.netleg.ef(lam['net'], u['net']) + self.lik.ef(lam['lik'], u['lik'])

    @partial(jax.jit, static_argnums=(0,))
    def dF_du(self, lam, u):
        return jax.grad(self.ef, argnums=1)(lam, u)

    @partial(jax.jit, static_argnums=(0,))
    def dF_du_direct(self, lam, u):
        return self.netleg.dF_du_direct(lam, u)

    @partial(jax.jit, static_argnums=(0,))
    def hessian_v_inner(self, theta, D, z):
        data_x, data_y = D
        _, tangent = jax.jvp(self.ell, (theta, D), (z, (0*data_x, 0*data_y)))
        return tangent
    
    @partial(jax.jit, static_argnums=(0,))
    def hessian_v(self, theta, D, z):
        if self.damping:
            return jax.tree_map(self.__add_damping, jax.grad(self.hessian_v_inner, argnums=0)(theta, D, z), z)
        else:
            return jax.grad(self.hessian_v_inner, argnums=0)(theta, D, z)
    
    @partial(jax.jit, static_argnums=(0,))
    def __x_minus_nu_y(self, x, y):
        return x-self.nu_theta*y
   
    @partial(jax.jit, static_argnums=(0,))
    def __mul_by_zero(self, x):
        return 0*x
    
    @partial(jax.jit, static_argnums=(0,))
    def __add_damping(self, x, y):
        return x + self.damping*y
    
    @partial(jax.jit, static_argnums=(0,))
    def __diff(self, x, y):
        return x-y

    @partial(jax.jit, static_argnums=(0,))
    def __dot(self, x, y):
        return jnp.sum(x*y)
    
    @partial(jax.jit, static_argnums=(0,))
    def test(self, x, y):
        return jnp.sum(x*y)
    
    @partial(jax.jit, static_argnums=(0,))
    def hessian_v2_inner(self, lam, g_tilde, d):
        lam_tangent = jax.tree_map(self.__mul_by_zero, jax.lax.stop_gradient(lam))
        _, tangent = jax.jvp(self.ef, (lam, g_tilde), (lam_tangent, d))
        return tangent
    
    @partial(jax.jit, static_argnums=(0,))
    def hessian_v2(self, lam, g_tilde, d):
        return jax.grad(self.hessian_v2_inner, argnums=0)(lam, g_tilde, d)
    
    @partial(jax.jit, static_argnums=(0,))
    def sample_model(self, theta, data_x, key):
        y_pred = self.net.forward(theta['net'], data_x)
        return (data_x, self.lik.sample(theta['lik'], y_pred, key))
    
    @partial(jax.jit, static_argnums=(0,))
    def ell(self, theta, D):
        return self.ell_without_reg(theta, D) + self.reg(theta)
    
    @partial(jax.jit, static_argnums=(0,))
    def ell_without_reg(self, theta, D):
        x, y = D
        y_pred = self.net.forward(theta['net'], x)
        return self.lik.nll(theta['lik'], y_pred, y)
    
    @partial(jax.jit, static_argnums=(0,))
    def reg(self, theta):
        theta_flat, _= ravel(theta)
        reg = 1e-5*0.5*jnp.linalg.norm(theta_flat)**2 
        return reg

    @partial(jax.jit, static_argnums=(0,))
    def normalize_gradient(self, g):   
        g_flat, unflatten_g = ravel(g)
        g_norm_flat = g_flat / jnp.linalg.norm(g_flat)
        return unflatten_g(g_norm_flat)

    def adam_update(self, lam, delta_lam, update_func, apply_update_func, adam_state):
        aux_params_update, new_adam_state = update_func(delta_lam, adam_state)
        new_lam = apply_update_func(lam, aux_params_update)
        return new_lam, new_adam_state

    def update_aux(self, theta, g, lam, data_x, \
                   adam_state, aux_opt_get_update, aux_opt_apply_updates, \
                   key):
        aux_loss = 0
        g_tilde = self.normalize_gradient(g)
        for k in range(self.K):
            key, subkey = jax.random.split(key)
            D = self.sample_model(theta, data_x, subkey)
            z = self.dF_du(lam, g_tilde)
            h = self.hessian_v(theta, D, jax.lax.stop_gradient(z))
            d = jax.tree_map(self.__diff, h, g_tilde)
            zd = jax.tree_map(self.__dot, z, d)         
            new_aux_loss = sum(jax.tree_util.tree_leaves(zd))
            aux_loss += new_aux_loss
            delta_lam = self.hessian_v2(lam, g_tilde, jax.lax.stop_gradient(d))
            lam, adam_state = self.adam_update(lam, delta_lam, \
                                               aux_opt_get_update, aux_opt_apply_updates, adam_state)
        return aux_loss / self.K, lam, adam_state
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, theta, g, lam):
        delta_theta = self.dF_du(lam, g)
        theta = jax.tree_map(self.__x_minus_nu_y, theta, delta_theta)
        return theta, delta_theta

    @partial(jax.jit, static_argnums=(0,))
    def elisabeth_the_II(self, g_tilde, D, theta, lam):
        z = self.dF_du_direct(lam, g_tilde)
        h = self.hessian_v(theta, D, jax.lax.stop_gradient(z))
        d = jax.tree_map(self.__diff, h, g_tilde)
        zd = jax.tree_map(self.__dot, z, d) 
        s = sum(jax.tree_util.tree_leaves(zd))
        return s

    def update_aux_direct(self, theta, g, lam, data_x, \
                   adam_state, aux_opt_get_update, aux_opt_apply_updates, \
                   key):
        aux_loss = 0
        g_tilde = self.normalize_gradient(g)
        for k in range(self.K):
            key, subkey = jax.random.split(key)
            D = self.sample_model(theta, data_x, subkey)
            delta_lam = jax.grad(self.elisabeth_the_II, argnums=3)(g_tilde, D, theta, lam)
            aux_loss += 0  # fix todo if you want to get the real aux_loss, currently not implemented
            lam, adam_state = self.adam_update(lam, delta_lam, \
                                               aux_opt_get_update, aux_opt_apply_updates,
                                               adam_state)
        return aux_loss / self.K, lam, adam_state

    @partial(jax.jit, static_argnums=(0,))
    def step_direct(self, theta, g, lam):
        delta_theta = self.dF_du_direct(lam, g)
        theta = jax.tree_map(self.__x_minus_nu_y, theta, delta_theta)
        return theta, delta_theta

