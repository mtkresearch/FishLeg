# SETUP EXPERIMENT 
import os
import json
import jax
import math
import pickle
import optax
import time
import argparse
from jax.config import config
config.update("jax_enable_x64", True)

from tensor_networks import MLPLegendreKronDiag
from target_networks import MLP
from likelihoods import SoftMaxLikelihood, Bernoulli, GaussianLikelihood
from fishleg import FishLeg
from image_datasets import read_data_sets
from utils import ell_for_whole_dataset, mse_for_whole_dataset, validateJSON
from configs import MLPs_dict, Autoencoders_dict

beta = 0.9

@jax.jit
def gb_update(gb, g):
    return beta*gb + (1-beta)*g

@jax.jit
def gb_debias(gb, beta_t):
    return gb/(1-beta_t)

parser = argparse.ArgumentParser(description="Farm arguments to read. Recomend use json config file",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--config", required=True, help="name of config_file")
parser.add_argument("-o","--output", default='results', help="Modify default output path", )
args = parser.parse_args()
arguments = vars(args)

outputdir = arguments['output']
os.makedirs(outputdir, exist_ok=True)
inputdir = './'
# Opening JSON file
with open(arguments['config']) as json_file:
    config_dict = json.load(json_file)

alg = config_dict['alg']
commit_nr = config_dict['commit_nr']#'961fef0'

# Parameters to change with launch script
sigma_fixed = (config_dict['sigma_fixed']== 'true' or config_dict['sigma_fixed']== 'True')
direct_step = (config_dict['direct_step']== 'true' or config_dict['direct_step']== 'True')
update_every = int(config_dict['update_every']) #1000
nu_theta = float(config_dict['nu_theta']) #0.01 # 4
nu_sgd = float(config_dict['nu_sgd']) #0.01 # 4
nu_lambda = float(config_dict['nu_lambda']) #1e-4 # 6
RUN = int(config_dict['run']) #1 # 3
damping = float(config_dict['damping']) #0
batch_size = int(config_dict['batch_size']) #1000
batch_size_aux = int(config_dict['batch_size_aux']) #1000

TARGET= config_dict['target'] #'FACES'
sigma_init = float( config_dict['sigma_init']) #1.0
epochs = int(config_dict['epochs']) #10

seed = RUN +int(config_dict['seed_init']) #12
task = config_dict['task']

config_save = {'commit_nr': commit_nr ,'algorithm': alg+("_DIRECT" if direct_step else ""), 'config_file': arguments['config'],
            'target': TARGET, 'activation_fun' : Autoencoders_dict[TARGET]['activation_funs'], 
            'layer_sizes':Autoencoders_dict[TARGET]['layer_sizes'],'batch_size': batch_size, 
            'batch_size_aux': batch_size_aux, 'Run': RUN, 'Epochs': epochs, 'beta': beta, 
            'sigma_init': sigma_init, 'nu_lambda': nu_lambda, 'nu_sgd': nu_sgd,'seed':seed,
            'nu_theta': nu_theta, 'damping': damping, 'sigma_fixed': sigma_fixed, 'task': task, 'update_every': update_every}

name_run = task+'_'+alg.replace(" ", "_")+("_DIRECT" if direct_step else "")+"_sigma_fixed_"+str(sigma_fixed)+"_damping_"\
    +str(damping)+"_nu_theta_"+str(nu_theta)+"_nu_sgd_"+str(nu_sgd)+"_nu_lambda_"+str(nu_lambda)+\
    '_autoencoder_'+TARGET+'_RUN_'+str(RUN)+'_BATCH_'+str(batch_size)+"_update_every_"+str(update_every)

pickle_name = os.path.join(outputdir,'JAX_'+name_run+'.pkl')
optimizer_name = os.path.join(outputdir, 'optimizer_'+name_run+'.pkl')
target_name = os.path.join(outputdir, 'target_'+name_run+'.pkl')

import sys
# sys.stdout = open(os.path.join(outputdir,'log_'+name_run+'.txt'), 'w')
print('Config file '+arguments['config'] +' read' )

# Check if configuration is valid and setup
if task == 'Autoencoder':
    NETWORK_dict = Autoencoders_dict
    if TARGET == "MNIST" or TARGET == "CURVES":
        lik = Bernoulli()
        nll_name = 'Bernoulli'
    elif TARGET=='FACES':
        lik = GaussianLikelihood(sigma_init, sigma_fixed = sigma_fixed)
        nll_name = 'Gaussian'
    else:
        raise NotImplementedError
elif task == 'Classification' and TARGET == "MNIST" :
    NETWORK_dict = MLPs_dict
    lik = SoftMaxLikelihood()
    nll_name = 'SoftMax'
else:
    raise NotImplementedError

#Setting up networks and variables
net = MLP(NETWORK_dict [TARGET]['activation_funs'], NETWORK_dict [TARGET]['layer_sizes'])
netleg = MLPLegendreKronDiag(NETWORK_dict [TARGET]['layer_sizes'])
fl = FishLeg(net, netleg, lik, nu_theta=nu_theta, damping=damping)
scale_lam = jax.numpy.sqrt(nu_sgd / nu_theta )
seed = RUN + 12
key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key)
theta = fl.init_theta(subkey)
lam = fl.init_lam(scale_lam)
g_bar = jax.tree_map(lambda x: 0*x, theta)
beta_t = 1   
aux_opt = optax.adam(nu_lambda)
adam_state = aux_opt.init(lam)
aux_opt_get_update = jax.jit(aux_opt.update)
aux_opt_apply_updates = jax.jit(optax.apply_updates)

# Init data logging lists
FL_loss = []
FL_test_loss = []
FL_acc = []
FL_test_acc = []
FL_epoch = []
FL_time = []

# Setting up data
dataset = read_data_sets(TARGET, inputdir, if_autoencoder=True)
input_dist = dataset.train
aux_dist = dataset.train

class_data = dataset.train.images
labels_data = dataset.train.labels
test_class_data = dataset.test.images
test_labels_data = dataset.test.labels

# Evaluate starting metrics
L = ell_for_whole_dataset(fl, theta, (class_data, labels_data), batch_size)
acc = mse_for_whole_dataset(fl, theta, (class_data, labels_data), batch_size, nll_name)
L_test = ell_for_whole_dataset(fl, theta, (test_class_data, test_labels_data), batch_size)
acc_test = mse_for_whole_dataset(fl, theta, (test_class_data, test_labels_data), batch_size, nll_name)
FL_loss.append(L)
FL_acc.append(acc)
FL_test_loss.append(L_test)
FL_test_acc.append(acc_test)
FL_epoch.append(0)
FL_time.append(0)

print("Loss for epoch", 0, "- train", L, "test", L_test, "Acc for epoch", 0, "- train", acc, "test", acc_test)

eval_time = 0
st = time.time()
count_iter = 0

for e in range(1, epochs+1):
    print("######## EPOCH", e, "RUN", RUN, "nu_theta", nu_theta, "nu_sgd", nu_sgd, "nu_lambda", nu_lambda)
    for t, j in enumerate(range(0, len(class_data), batch_size)):     
        beta_t = beta_t * beta

        D = input_dist.sample(batch_size)       
        L, g = jax.value_and_grad(fl.ell, argnums=0)(theta, D)            

        g_bar = jax.tree_map(gb_update, g_bar, g)

        data_x, data_y = aux_dist.sample(batch_size_aux)
        if direct_step:
            if not count_iter % update_every:
                _, lam, adam_state = fl.update_aux_direct(theta, g_bar, lam, data_x, \
                                                adam_state, aux_opt_get_update, aux_opt_apply_updates, subkey)
            theta, delta_theta = fl.step_direct(theta, g_bar, lam)
        else:
            if not count_iter % update_every:
                _, lam, adam_state = fl.update_aux(theta, g_bar, lam, data_x, \
                                                adam_state, aux_opt_get_update, aux_opt_apply_updates, subkey)
            theta, delta_theta = fl.step(theta, g_bar, lam)

        et = time.time()
        if t % 10 == 0:
            print("Run", RUN, "EPOCH", e, " Step", t, "Loss", L)
        if math.isnan(L):
            print("Run", RUN, "EPOCH", e, " Step", t, "Loss", L)
            break
        count_iter += 1
    L = ell_for_whole_dataset(fl, theta, (class_data, labels_data), batch_size)
    L_test = ell_for_whole_dataset(fl, theta, (test_class_data, test_labels_data), batch_size)
    acc = mse_for_whole_dataset(fl, theta, (class_data, labels_data), batch_size, nll_name)
    acc_test = mse_for_whole_dataset(fl, theta, (test_class_data, test_labels_data), batch_size, nll_name)
    FL_loss.append(L)
    FL_test_loss.append(L_test)
    FL_acc.append(acc)
    FL_test_acc.append(acc_test)
    FL_epoch.append(e)
    print("Loss for epoch", e, "- train", L, "test", L_test, "Acc for epoch", e, "- train", acc, "test", acc_test)
    sys.stdout.flush()
    eval_time += time.time() - et
    FL_time.append(time.time() - st - eval_time)
    experiment_list = [FL_loss, FL_acc, FL_test_loss, FL_epoch, FL_test_acc, FL_time, config_save]   
    optimizer_list = [lam, config_save]
    target_list = [theta, config_save]

    with open(optimizer_name, 'wb') as handle:
        pickle.dump(optimizer_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(target_name, 'wb') as handle:
        pickle.dump(target_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(pickle_name, 'wb') as handle:
        pickle.dump(experiment_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if math.isnan(L):
        break




