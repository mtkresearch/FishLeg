import jax
import jax.numpy as jnp
import jax.random as random
from functools import partial
from utils import dense_to_one_hot

class GaussianLikelihood():
    def __init__(self, sigma_init, sigma_fixed):
        self.sigma_init = sigma_init
        self.sigma_fixed = sigma_fixed
      
    def init_theta(self):
        return self.sigma_init
    
    def init_lam(self, scale):
        return scale
    
    @partial(jax.jit, static_argnums=(0,))    
    def nll(self, theta, y_pred, y):
        """
        *Inputs*
        y_pred (output layer of the neural net)
        y (some label)
        sigma (standard deviation of the conditional density)

        *Returns: the negative log likelihood
        """
        sigma = theta
        if self.sigma_fixed:
            sigma = self.sigma_init
        return 0.5 * (jnp.sum(jnp.square((y_pred - y) / sigma)) + jnp.log(sigma**2))/ y_pred.shape[0]
    
    @partial(jax.jit, static_argnums=(0,))
    def sample(self, theta, y_pred, key):
        """
        *Inputs*
        y_pred (output layer of the neural net)
        sigma (standard deviation of the conditional density)

        *Returns: a sample from the model's conditional density
        """
        sigma = theta
        if self.sigma_fixed:
            sigma = self.sigma_init
        y = y_pred + sigma * jax.random.normal(key, shape=y_pred.shape)
        return y
    
    @partial(jax.jit, static_argnums=(0,))    
    def ef(self, lam, u):
        return 0.5*((lam*u)**2)
    

class Bernoulli():
    def __init__(self):
        pass
      
    def init_theta(self):
        return 1.0
    
    def init_lam(self, scale):
        return scale
    
    @partial(jax.jit, static_argnums=(0,))    
    def nll(self, theta, y_pred, y):
        """
        *Inputs*
        y_pred (output layer of the neural net)
        y (some label)
        sigma (standard deviation of the conditional density)

        *Returns: the negative log likelihood
        """
        max_val = jnp.clip(-y_pred, 0, None)
        return jnp.sum(y_pred *(1.0- y) + max_val + jnp.log(jnp.exp(-max_val) \
                      + jnp.exp((-y_pred - max_val))))/y_pred.shape[0]
    
    @partial(jax.jit, static_argnums=(0,))
    def sample(self, theta, y_pred, key):
        """
        *Inputs*
        y_pred (output layer of the neural net)
        sigma (standard deviation of the conditional density)

        *Returns: a sample from the model's conditional density
        """
        pred_dist = jax.nn.sigmoid(y_pred)
        return 1.0 *jax.random.bernoulli(key,pred_dist)
        
    
    @partial(jax.jit, static_argnums=(0,))    
    def ef(self, lam, u):
        return 0.0*(lam*u)
       

class SoftMaxLikelihood():
    def __init__(self):       
        pass
    def init_theta(self):
        '''
        zero but needs to be here for current fl class
        '''
        return 0.0
    def init_lam(self, scale):
        '''
        retruns scale but needs to be here for current fl class
        '''
        return scale
    
    @partial(jax.jit, static_argnums=(0,))    
    def nll(self, theta, y_pred, y):
        """
        *Inputs*
        y_pred (output layer of the neural net)
        y (some label)
        sigma, not used but needs to be here becouse of compatibility with fl class

        *Returns: the negative log likelihood
        """
        logits = jax.nn.log_softmax(y_pred, axis=1)

        return -jnp.mean(jnp.sum(logits * y, axis=1)) 

    
    @partial(jax.jit, static_argnums=(0,))
    def sample(self,theta,y_pred, key):
        """
        *Inputs*
        y_pred (output layer of the neural net)
        sigma: zero but needs to be here for current fl class

        *Returns: a sample from the model's conditional density
        """
        logits = jnp.log(y_pred)
        return dense_to_one_hot(jax.random.categorical(key,logits, axis=1))
    
    @partial(jax.jit, static_argnums=(0,))    
    def ef(self, lam, u):
        return 0.0*(lam*u)
