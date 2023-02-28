import jax.numpy as jnp
import jax
import math
import os
import json


def block_tree_until_ready(pytree):
  return jax.tree_map(lambda x: x.block_until_ready(), pytree)

def dense_to_one_hot(y, max_value=9,min_value=0):
    """
    converts y into one hot reprsentation.

    Parameters
    ----------
    y : list
        A list containing continous integer values.

    Returns
    -------
    one_hot : numpy.ndarray
        A numpy.ndarray object, which is one-hot representation of y.

    """
    length = len(y)
    one_hot = jnp.zeros((length, (max_value - min_value + 1)))
    return  one_hot.at[list(range(length)), y].set(1)

def validateJSON(jsonData):
    try:
        json.loads(jsonData)
    except ValueError as err:
        return False
    return True

def MSE_accuracy(z, t):
    err = z - t
    return jnp.mean(jnp.square(err))

def bernoulli_accuracy(z, t):
    err = jax.nn.sigmoid(z) - t
    return jnp.mean(jnp.square(err))  

def class_accuracy(z, t):
    #need to be tested 
    y = jnp.argmax(z)
    return jnp.mean(float(y==t))

def ell_for_whole_dataset(fl, theta, D, batch_size): 
    x, y = D
    N1 = jnp.min(jnp.array([batch_size, len(x)]))
    i = 0
    list_loss = []    
    while i + N1 <= int(len(x)):
        list_loss.append(fl.ell(theta, (x[i: i+N1], y[i: i+N1])))
        i += N1
    return sum(list_loss) / len(list_loss)

def mse_for_whole_dataset(fl, theta, D, batch_size, nll_name ): 
    x, y = D
    N1 = jnp.min(jnp.array([batch_size, len(x)]))
    i = 0
    list_acc = []    
    while i + N1 <= int(len(x)):
        if nll_name == 'Gaussian':
            list_acc.append(MSE_accuracy(fl.net.forward(theta['net'], x[i: i+N1]), y[i: i+N1]))
        elif nll_name == 'SoftMax':
             list_acc.append(class_accuracy(fl.net.forward(theta['net'], x[i: i+N1]), y[i: i+N1]))
        elif nll_name == 'Bernoulli':
             list_acc.append(bernoulli_accuracy(fl.net.forward(theta['net'], x[i: i+N1]), y[i: i+N1]))
        else:
            print('This test error is not implemented')
            list_acc.append(None)
            break
                    
        i += N1
    return sum(list_acc) / len(list_acc)
