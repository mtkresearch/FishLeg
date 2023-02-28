import jax.numpy as jnp
import jax.random as random

class HilbertBowl():
    def __init__(self, N):
        self.N = N
        self.sigma = jnp.zeros(shape=(N, N))
        for i in range(1, N+1):
            for j in range(1, N+1):
                if (i % j == 0) or (j % i == 0):
                    self.sigma = self.sigma.at[i-1,j-1].set(1.0/(i+j-1))

    def sample(self, key, batch_size):
        return random.normal(key, shape=(batch_size, self.N)) @ self.sigma