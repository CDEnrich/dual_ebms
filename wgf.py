import argparse
import numpy as np
import math
import os, sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.special as ss
import matplotlib.pyplot as plt
import time
from statsmodels.distributions.empirical_distribution import ECDF

from mala import get_samples, get_samples_uniform_proposal
from mle import free_energy_sampling, cross_entropy_avgterm
#from util import get_teacher, teacher_samples, to_bool

plt.style.use('ggplot')

TEACHER_BURNIN = 10000
TEACHER_BURN = 4

#theta in S^d with uniform base measure, x in R^d \times \{1\} with multivariate Gaussian base measure

def energy(X, positive_Win, negative_Win, args):
    ones_d = torch.ones(X.shape[0],1).float()
    X = torch.cat((X,ones_d), 1)
    neurons = positive_Win.shape[1] + negative_Win.shape[1]
    #print(X.shape, positive_Win.shape)
    return args.beta * (torch.sum(nn.functional.relu(torch.matmul(X, positive_Win)), dim=1) - torch.sum(nn.functional.relu(torch.matmul(X, negative_Win)), dim=1))/neurons

def energy_sum(X, positive_Win, negative_Win, args):
    ones_d = torch.ones(X.shape[0],1).float()
    #print(X.shape, positive_Win.shape)
    X = torch.cat((X,ones_d), 1)
    #print(X.shape, positive_Win.shape)
    neurons = positive_Win.shape[1] + negative_Win.shape[1]
    return args.beta * (torch.sum(nn.functional.relu(torch.matmul(X, positive_Win))) - torch.sum(nn.functional.relu(torch.matmul(X, negative_Win))))/neurons

def energy_sphere(X, positive_Win, negative_Win, args):
    neurons = positive_Win.shape[1] + negative_Win.shape[1]
    #print(X.shape, positive_Win.shape, negative_Win.shape)
    if args.signed_quadratic:
        X = torch.sign(X)*X**2
    return args.beta * (torch.sum(nn.functional.relu(torch.matmul(X, positive_Win)), dim=1) - torch.sum(nn.functional.relu(torch.matmul(X, negative_Win)), dim=1))/neurons

def energy_sum_sphere(X, positive_Win, negative_Win, args):
    neurons = positive_Win.shape[1] + negative_Win.shape[1]
    if args.signed_quadratic:
        X = torch.sign(X)*X**2
    return args.beta * (torch.sum(nn.functional.relu(torch.matmul(X, positive_Win))) - torch.sum(nn.functional.relu(torch.matmul(X, negative_Win))))/neurons

def get_positive_target_neurons(args, second_dataset=False):
    if not second_dataset:
        fname = os.path.join('target_positive_neurons', f'target_positive_neurons_{args.d}_{args.k}_{args.pre_target_neurons}_{args.seed}.pkl')
    if second_dataset:
        fname = os.path.join('target_positive_neurons', f'target_positive_neurons_2_{args.d}_{args.k}_{args.pre_target_neurons}_{args.seed}.pkl')
    if os.path.exists(fname):
        X = pickle.load(open(fname, 'rb'))
        return X
    else:
        q_k_d = ss.jacobi(args.k, (args.d-3)/2.0, (args.d-3)/2.0)
        legendre_k_d = q_k_d/q_k_d(1)
        if not second_dataset:
            torch.manual_seed(args.seed)
        else:
            torch.manual_seed(10*args.seed)
        for j in range(args.pre_target_neurons//100000):
            start = time.time()
            X0 = torch.randn(100000,args.d)
            X0 = torch.nn.functional.normalize(X0, p=2, dim=1)
            if args.d > 2: 
                acceptance_prob = torch.nn.functional.relu(torch.from_numpy(0.99*legendre_k_d(X0[:,args.d-3]) + 0.495*legendre_k_d(X0[:,args.d-2])))
            else:
                acceptance_prob = torch.nn.functional.relu(torch.from_numpy(0.495*legendre_k_d(X0[:,args.d-2]) + 0.495*legendre_k_d(X0[:,args.d-1])))
            acceptance_vector = torch.bernoulli(acceptance_prob)
            accepted_rows = []
            for i in range(100000):
                if acceptance_vector[i] == 1:
                    accepted_rows.append(i)
            accepted_rows_tensor = torch.tensor(accepted_rows).unsqueeze(1).expand([len(accepted_rows),args.d])
            if j==0:
                X = torch.gather(X0, 0, accepted_rows_tensor)
                print(f'Sample batch {j+1}/{args.pre_target_neurons//100000} done in {time.time()-start}. {X.shape[0]} more samples.')
            elif X.shape[0] < args.target_neurons:
                samples = torch.gather(X0, 0, accepted_rows_tensor)
                X = torch.cat((X,samples),0)
                print(f'Sample batch {j+1}/{args.pre_target_neurons//100000} done in {time.time()-start}. {samples.shape[0]} more samples.')
            else:
                continue
        if not os.path.exists('target_positive_neurons'):
            os.makedirs('target_positive_neurons')
        pickle.dump(X.t(), open(fname, 'wb'))
        print(f'Positive neurons created with shape: {X.shape[0]},{X.shape[1]}')
        return X.t()
    
def get_negative_target_neurons(args, second_dataset=False):
    if not second_dataset:
        fname = os.path.join('target_negative_neurons', f'target_negative_neurons_{args.d}_{args.k}_{args.pre_target_neurons}_{args.seed}.pkl')
    if second_dataset:
        fname = os.path.join('target_negative_neurons', f'target_negative_neurons_2_{args.d}_{args.k}_{args.pre_target_neurons}_{args.seed}.pkl')
    if os.path.exists(fname):
        X = pickle.load(open(fname, 'rb'))
        return X
    else:
        q_k_d = ss.jacobi(args.k, (args.d-3)/2.0, (args.d-3)/2.0)
        legendre_k_d = q_k_d/q_k_d(1)
        if not second_dataset:
            torch.manual_seed(args.seed)
        else:
            torch.manual_seed(10*args.seed)
        for j in range(args.pre_target_neurons//100000):
            start = time.time()
            X0 = torch.randn(100000,args.d)
            X0 = torch.nn.functional.normalize(X0, p=2, dim=1)
            if args.d > 2: 
                acceptance_prob = torch.nn.functional.relu(torch.from_numpy(-0.99*legendre_k_d(X0[:,args.d-3]) - 0.495*legendre_k_d(X0[:,args.d-2])))
            else:
                acceptance_prob = torch.nn.functional.relu(torch.from_numpy(-0.495*legendre_k_d(X0[:,args.d-2]) - 0.495*legendre_k_d(X0[:,args.d-1])))
            acceptance_vector = torch.bernoulli(acceptance_prob)
            accepted_rows = []
            for i in range(100000):
                if acceptance_vector[i] == 1:
                    accepted_rows.append(i)
            accepted_rows_tensor = torch.tensor(accepted_rows).unsqueeze(1).expand([len(accepted_rows),args.d])
            if j==0:
                X = torch.gather(X0, 0, accepted_rows_tensor)
                print(f'Neuron sample batch {j+1}/{args.pre_target_neurons//100000} done in {time.time()-start}. {X.shape[0]} more samples.')
            elif X.shape[0] < args.target_neurons:
                samples = torch.gather(X0, 0, accepted_rows_tensor)
                X = torch.cat((X,samples),0)
                print(f'Neuron sample batch {j+1}/{args.pre_target_neurons//100000} done in {time.time()-start}. {samples.shape[0]} more samples.')
            else:
                continue
        if not os.path.exists('target_negative_neurons'):
            os.makedirs('target_negative_neurons')
        pickle.dump(X.t(), open(fname, 'wb'))
        print(f'Negative neurons created with shape: {X.shape[0]},{X.shape[1]}')
        return X.t()
    
def compute_f2_norm(args):
    q_k_d = ss.jacobi(args.k, (args.d-3)/2.0, (args.d-3)/2.0)
    legendre_k_d = q_k_d/q_k_d(1)
    sum_weights = 0
    sum_squared_weights = 0
    for j in range(args.pre_target_neurons//100000):
        X0 = torch.randn(100000,args.d)
        X0 = torch.nn.functional.normalize(X0, p=2, dim=1)
        if args.d > 2: 
            acceptance_prob = torch.nn.functional.relu(torch.from_numpy(0.99*legendre_k_d(X0[:,args.d-3]) + 0.495*legendre_k_d(X0[:,args.d-2])))
        else:
            acceptance_prob = torch.nn.functional.relu(torch.from_numpy(0.495*legendre_k_d(X0[:,args.d-2]) + 0.495*legendre_k_d(X0[:,args.d-1])))
        sum_weights_j = torch.sum(torch.abs(acceptance_prob))
        sum_squared_weights_j = torch.sum(acceptance_prob**2)
        sum_weights = sum_weights + sum_weights_j
        sum_squared_weights = sum_squared_weights + sum_squared_weights_j
        print(f'F2 norm computation. Batch {j+1}/{args.pre_target_neurons//100000}')
    print(f'Sqrt of sum of squared weights: {np.sqrt(sum_squared_weights)}. Sum of abs of weights: {sum_weights}.')
    f_2_norm = np.sqrt(sum_squared_weights*100000*(args.pre_target_neurons//100000))/sum_weights
    return f_2_norm
            

def teacher_samples(args):
    if not args.signed_quadratic:
        if not args.add_gaussian:
            if not args.SGD:
                if args.sphere:
                    fname = os.path.join('data', f'data_sphere_{args.target_neurons}_{args.beta}_{args.d}_{args.k}_{args.pre_target_neurons}_{args.seed}.pkl')
                else:
                    fname = os.path.join('data', f'data_{args.target_neurons}_{args.beta}_{args.d}_{args.k}_{args.pre_target_neurons}_{args.seed}.pkl')
            else:
                if args.sphere:
                    fname = os.path.join('data_SGD', f'data_sphere_{args.target_neurons}_{args.beta}_{args.d}_{args.k}_{args.pre_target_neurons}_{args.seed}.pkl')
                else:
                    fname = os.path.join('data_SGD', f'data_{args.target_neurons}_{args.beta}_{args.d}_{args.k}_{args.pre_target_neurons}_{args.seed}.pkl')
        else:
            if not args.SGD:
                if args.sphere:
                    fname = os.path.join('data_gaussian', f'data_sphere_{args.target_neurons}_{args.beta}_{args.d}_{args.k}_{args.pre_target_neurons}_{args.seed}.pkl')
                else:
                    fname = os.path.join('data_gaussian', f'data_{args.target_neurons}_{args.beta}_{args.d}_{args.k}_{args.pre_target_neurons}_{args.seed}.pkl')
            else:
                if args.sphere:
                    fname = os.path.join('data_gaussian_SGD', f'data_sphere_{args.target_neurons}_{args.beta}_{args.d}_{args.k}_{args.pre_target_neurons}_{args.seed}.pkl')
                else:
                    fname = os.path.join('data_gaussian_SGD', f'data_{args.target_neurons}_{args.beta}_{args.d}_{args.k}_{args.pre_target_neurons}_{args.seed}.pkl')
    else:
        if not args.add_gaussian:
            if not args.SGD:
                if args.sphere:
                    fname = os.path.join('data', f'data_sphere_signedq_{args.target_neurons}_{args.beta}_{args.d}_{args.k}_{args.pre_target_neurons}_{args.seed}.pkl')
                else:
                    fname = os.path.join('data', f'data_signedq_{args.target_neurons}_{args.beta}_{args.d}_{args.k}_{args.pre_target_neurons}_{args.seed}.pkl')
            else:
                if args.sphere:
                    fname = os.path.join('data_SGD', f'data_sphere_signedq_{args.target_neurons}_{args.beta}_{args.d}_{args.k}_{args.pre_target_neurons}_{args.seed}.pkl')
                else:
                    fname = os.path.join('data_SGD', f'data_signedq_{args.target_neurons}_{args.beta}_{args.d}_{args.k}_{args.pre_target_neurons}_{args.seed}.pkl')
        else:
            if not args.SGD:
                if args.sphere:
                    fname = os.path.join('data_gaussian', f'data_sphere_signedq_{args.target_neurons}_{args.beta}_{args.d}_{args.k}_{args.pre_target_neurons}_{args.seed}.pkl')
                else:
                    fname = os.path.join('data_gaussian', f'data_signedq_{args.target_neurons}_{args.beta}_{args.d}_{args.k}_{args.pre_target_neurons}_{args.seed}.pkl')
            else:
                if args.sphere:
                    fname = os.path.join('data_gaussian_SGD', f'data_sphere_signedq_{args.target_neurons}_{args.beta}_{args.d}_{args.k}_{args.pre_target_neurons}_{args.seed}.pkl')
                else:
                    fname = os.path.join('data_gaussian_SGD', f'data_signedq_{args.target_neurons}_{args.beta}_{args.d}_{args.k}_{args.pre_target_neurons}_{args.seed}.pkl')
        
    
    if not args.recompute_data and os.path.exists(fname):
        positive_neurons = get_positive_target_neurons(args)
        if positive_neurons.shape[1] > 3000:
            positive_neurons = positive_neurons[:,:3000]
        negative_neurons = get_negative_target_neurons(args)
        if negative_neurons.shape[1] > 3000:
            negative_neurons = negative_neurons[:,:3000]
        if not args.SGD:
            Xtr, Xval, Xte = pickle.load(open(fname, 'rb'))
            print(f'Xtr shape: {Xtr.shape[0]},{Xtr.shape[1]}, Xval: {Xval.shape[0]},{Xval.shape[1]}, Xte: {Xte.shape[0]},{Xte.shape[1]}')
            return Xtr, Xval, Xte, positive_neurons, negative_neurons
        else:
            X = pickle.load(open(fname, 'rb'))
            print(f'X shape: {X.shape[0]},{X.shape[1]}')
            return X, positive_neurons, negative_neurons
    else:
        positive_neurons = get_positive_target_neurons(args)
        if positive_neurons.shape[1] > 3000:
            positive_neurons = positive_neurons[:,:3000]
        negative_neurons = get_negative_target_neurons(args)
        if negative_neurons.shape[1] > 3000:
            negative_neurons = negative_neurons[:,:3000]
        '''
        n_gd_samples = 5000
        if args.sphere:
            gd_particles = torch.randn(n_gd_samples,args.d)
            #gd_particles = torch.nn.functional.normalize(gd_particles, p=2, dim=1)
        else:
            gd_particles = torch.randn(n_gd_samples,args.d-1)
        for gd_iteration in range(2000):
            
            if gd_iteration%500 == 0:
                if args.sphere:
                    final_values = energy_sphere(gd_particles, positive_neurons, negative_neurons, args)
                    print(f'Min final_values_gd: {torch.min(final_values)}')
                else:
                    final_values = energy(gd_particles, positive_neurons, negative_neurons, args)
                min_energy = torch.min(final_values)
                print("Minimum energy precomputation, iteration", gd_iteration, min_energy)
                
            gd_particles.requires_grad_()
            if args.sphere:
                fun_value = energy_sum_sphere(gd_particles, positive_neurons, negative_neurons, args)
            else:
                fun_value = energy_sum(gd_particles, positive_neurons, negative_neurons, args)
            gradient = torch.autograd.grad(fun_value, gd_particles)[0]
            #print('Gradient magnitude:', torch.mean(torch.abs(gradient)))
            gd_particles.detach_()
            gd_particles.sub_(0.1*gradient)
            #if args.sphere:
            #    gd_particles = torch.nn.functional.normalize(gd_particles, p=2, dim=1)
            if gd_iteration%500 == 0:
                if args.sphere:
                    final_values = energy_sphere(gd_particles, positive_neurons, negative_neurons, args)
                    print(f'Min final_values_gd: {torch.min(final_values)}')
                else:
                    final_values = energy(gd_particles, positive_neurons, negative_neurons, args)
                min_energy = torch.min(final_values)
                print("Minimum energy precomputation, iteration", gd_iteration, min_energy)
        if args.sphere:
            final_values = energy_sphere(gd_particles, positive_neurons, negative_neurons, args)
        else:
            final_values = energy(gd_particles, positive_neurons, negative_neurons, args)
        min_energy = torch.min(final_values)
        print("Minimum energy:", min_energy)
        '''
        min_energy = 100000
        start = time.time()
        for j in range(args.total_n_samples//100000):
            if args.sphere:
                X0 = torch.randn(100000,args.d)
                if args.signed_quadratic:
                    X0 = torch.sqrt(torch.abs(X0))*torch.sign(X0)
                #print('X0', torch.mean(X0))
                #print(X0[:5,:])
                X_energy = energy_sphere(X0, positive_neurons, negative_neurons, args)
                #print('X_energy')
                #print(X_energy[:5])
                if args.add_gaussian:
                    X_energy = X_energy + 0.5*torch.norm(X0, dim=1)**2
                if torch.min(X_energy) < min_energy:
                    min_energy = torch.min(X_energy)
            else:
                X0 = torch.randn(100000,args.d-1)
                if args.signed_quadratic:
                    X0 = torch.sqrt(X0)*torch.sign(X0)
                X_energy = energy(X0, positive_neurons, negative_neurons, args)
                if args.add_gaussian:
                    X_energy = X_energy + 0.5*torch.norm(X0, dim=1)**2
                if torch.min(X_energy) < min_energy:
                    min_energy = torch.min(X_energy)
        print(f'Minimum energy: {min_energy}. Time elapsed: {time.time() - start}')
        min_energy = torch.min(min_energy*1.15, min_energy*0.8)
        #min_energy = min_energy*1.15
        if args.d < 3:
            min_energy = min_energy-0.05
        
        for j in range(args.total_n_samples//100000):
            start = time.time()
            if args.sphere:
                X0 = torch.randn(100000, args.d)
                #norms_X0 = torch.norm(X0, dim=1)
                if args.signed_quadratic:
                    X0 = torch.sqrt(torch.abs(X0))*torch.sign(X0)
                X_energy = energy_sphere(X0, positive_neurons, negative_neurons, args) #+ 0.5*torch.norm(X0, dim=1)**2
            else:
                X0 = torch.randn(100000,args.d-1)
                #X = torch.nn.functional.normalize(X, p=2, dim=1)
                if args.signed_quadratic:
                    X0 = torch.sqrt(X0)*torch.sign(X0)
                X_energy = energy(X0, positive_neurons, negative_neurons, args) #+ 0.5*torch.norm(X0, dim=1)**2
            if args.add_gaussian:
                X_energy = X_energy + 0.5*torch.norm(X0, dim=1)**2
            print(f'Average energy: {torch.mean(X_energy)}. Minimum energy: {torch.min(X_energy)}')
            
            #min_energy = torch.min(X_energy)
            X_density = torch.exp(-X_energy + min_energy)
        
            acceptance_vector = torch.bernoulli(X_density)
            #print(torch.norm(X_density, p=1), torch.norm(acceptance_vector, p=1))
            accepted_rows = []
            for i in range(100000):
                if acceptance_vector[i] == 1:
                    accepted_rows.append(i)
            if args.sphere:
                accepted_rows_tensor = torch.LongTensor(accepted_rows).unsqueeze(1).expand([len(accepted_rows),args.d])
            else:
                accepted_rows_tensor = torch.tensor(accepted_rows).unsqueeze(1).expand([len(accepted_rows),args.d-1])
            if j==0:
                print('types:', X0.dtype, accepted_rows_tensor.dtype)
                X = torch.gather(X0, 0, accepted_rows_tensor)
                print(f'Sample batch {j+1}/{args.total_n_samples//100000} done in {time.time()-start}. {X.shape[0]} more samples.')
            else:
                samples = torch.gather(X0, 0, accepted_rows_tensor)
                X = torch.cat((X,samples),0)
                print(f'Sample batch {j+1}/{args.total_n_samples//100000} done in {time.time()-start}. {samples.shape[0]} more samples.')
        print(f'X size: {X.shape[0]}')
        
        if not args.SGD:
            Xtr = X[:args.n_samples]
            Xval = X[args.n_samples:(args.n_samples+args.n_samples_val)]
            Xte = X[(args.n_samples+args.n_samples_val):(args.n_samples+args.n_samples_val+args.n_samples_te)]
            if not args.add_gaussian and not os.path.exists('data'):
                os.makedirs('data')
            if args.add_gaussian and not os.path.exists('data_gaussian'):
                os.makedirs('data_gaussian')
            pickle.dump((Xtr, Xval, Xte), open(fname, 'wb'))
            return Xtr, Xval, Xte, positive_neurons, negative_neurons
        else:
            if not args.add_gaussian and not os.path.exists('data_SGD'):
                os.makedirs('data_SGD')
            if args.add_gaussian and not os.path.exists('data_gaussian_SGD'):
                os.makedirs('data_gaussian_SGD')
            pickle.dump(X, open(fname, 'wb'))
            return X, positive_neurons, negative_neurons
        
def target_log_partition(positive_neurons, negative_neurons, args):
    if not args.signed_quadratic:
        if args.add_gaussian:
            if not args.sphere:
                fname = os.path.join('target_log_partitions_gaussian', f'target_log_partition_{args.target_neurons}_{args.beta}_{args.d}_{args.k}_{args.target_log_partition_samples}_{args.seed}.pkl')
            else:
                fname = os.path.join('target_log_partitions_gaussian', f'target_log_partition_sphere_{args.target_neurons}_{args.beta}_{args.d}_{args.k}_{args.target_log_partition_samples}_{args.seed}.pkl')
        else:
            if not args.sphere:
                fname = os.path.join('target_log_partitions', f'target_log_partition_{args.target_neurons}_{args.beta}_{args.d}_{args.k}_{args.target_log_partition_samples}_{args.seed}.pkl')
            else:
                fname = os.path.join('target_log_partitions', f'target_log_partition_sphere_{args.target_neurons}_{args.beta}_{args.d}_{args.k}_{args.target_log_partition_samples}_{args.seed}.pkl')
    else:
        if args.add_gaussian:
            if not args.sphere:
                fname = os.path.join('target_log_partitions_gaussian', f'target_log_partition_signedq_{args.target_neurons}_{args.beta}_{args.d}_{args.k}_{args.target_log_partition_samples}_{args.seed}.pkl')
            else:
                fname = os.path.join('target_log_partitions_gaussian', f'target_log_partition_sphere_signedq_{args.target_neurons}_{args.beta}_{args.d}_{args.k}_{args.target_log_partition_samples}_{args.seed}.pkl')
        else:
            if not args.sphere:
                fname = os.path.join('target_log_partitions', f'target_log_partition_signedq_{args.target_neurons}_{args.beta}_{args.d}_{args.k}_{args.target_log_partition_samples}_{args.seed}.pkl')
            else:
                fname = os.path.join('target_log_partitions', f'target_log_partition_sphere_signedq_{args.target_neurons}_{args.beta}_{args.d}_{args.k}_{args.target_log_partition_samples}_{args.seed}.pkl')
    
    if not args.recompute_data and os.path.exists(fname):
        log_partition, error = pickle.load(open(fname, 'rb'))
        return log_partition, error
    else:
        start = time.time()
        partition = 0
        partition_variance = 0
        if args.sphere:
            for i in range(args.target_log_partition_samples//100000):
                X0 = torch.randn(100000,args.d)
                if not args.add_gaussian:
                    partition = partition + torch.mean(torch.exp(-energy_sphere(X0, positive_neurons, negative_neurons, args)))
                    partition_variance = partition_variance + torch.std(torch.exp(-energy_sphere(X0, positive_neurons, negative_neurons, args)))**2
                else:
                    partition = partition + torch.mean(torch.exp(-energy_sphere(X0, positive_neurons, negative_neurons, args) - 0.5*torch.norm(X0, dim=1)**2))
                    partition_variance = partition_variance + torch.std(torch.exp(-energy_sphere(X0, positive_neurons, negative_neurons, args) - 0.5*torch.norm(X0, dim=1)**2))**2
                print(f'Target log-partition computation, iteration: {i+1}/{args.target_log_partition_samples//100000}')
            partition = partition/(args.target_log_partition_samples//100000)
            partition_variance = partition_variance/(args.target_log_partition_samples//100000)
            log_partition = np.log(partition) 
            log_partition_error = np.sqrt(partition_variance/((args.target_log_partition_samples//100000)*100000))
        else:
            for i in range(args.target_log_partition_samples//100000):
                X0 = torch.randn(100000,args.d-1)
                if not args.add_gaussian:
                    partition = partition + torch.mean(torch.exp(-energy(X0, positive_neurons, negative_neurons, args)))
                    partition_variance = partition_variance + torch.std(torch.exp(-energy(X0, positive_neurons, negative_neurons, args)))**2
                else:
                    partition = partition + torch.mean(torch.exp(-energy(X0, positive_neurons, negative_neurons, args) - 0.5*torch.norm(X0, dim=1)**2))
                    partition_variance = partition_variance + torch.std(torch.exp(-energy(X0, positive_neurons, negative_neurons, args) - 0.5*torch.norm(X0, dim=1)**2))**2
                print(f'Target log-partition computation, iteration: {i+1}/{args.target_log_partition_samples//100000}. Time elapsed: {time.time() - start}.')
            partition = partition/(args.target_log_partition_samples//100000)
            partition_variance = partition_variance/(args.target_log_partition_samples//100000)
            log_partition = np.log(partition) 
            log_partition_error = np.sqrt(partition_variance/((args.target_log_partition_samples//100000)*100000))
        if not args.add_gaussian and not os.path.exists('target_log_partitions'):
            os.makedirs('target_log_partitions')
        if args.add_gaussian and not os.path.exists('target_log_partitions_gaussian'):
            os.makedirs('target_log_partitions_gaussian')
        print(f'Target log-partition computation finished in {time.time() - start}')
        pickle.dump((log_partition, log_partition_error), open(fname, 'wb'))
        return log_partition, log_partition_error
'''    
def exact_train_log_partition(tr_samples,Xtr, args):
    start = time.time()
    partition = 0
    partition_variance = 0
    if args.sphere:
        for i in range(args.target_log_partition_samples//100000):
            X0 = torch.randn(100000,args.d)
            if not args.add_gaussian:
                partition = partition + torch.mean(torch.exp(-energy_sphere(X0, positive_neurons, negative_neurons, args)))
                partition_variance = partition_variance + torch.std(torch.exp(-energy_sphere(X0, positive_neurons, negative_neurons, args)))**2
            else:
                partition = partition + torch.mean(torch.exp(-energy_sphere(X0, positive_neurons, negative_neurons, args) - 0.5*torch.norm(X0, dim=1)**2))
                partition_variance = partition_variance + torch.std(torch.exp(-energy_sphere(X0, positive_neurons, negative_neurons, args) - 0.5*torch.norm(X0, dim=1)**2))**2
            print(f'Target log-partition computation, iteration: {i+1}/{args.target_log_partition_samples//100000}')
        partition = partition/(args.target_log_partition_samples//100000)
        partition_variance = partition_variance/(args.target_log_partition_samples//100000)
        log_partition = np.log(partition) 
        log_partition_error = np.sqrt(partition_variance/((args.target_log_partition_samples//100000)*100000))
    else:
        for i in range(args.target_log_partition_samples//100000):
            X0 = torch.randn(100000,args.d-1)
            partition = partition + torch.mean(torch.exp(-energy(X0, positive_neurons, negative_neurons, args)))
            partition_variance = partition_variance + torch.std(torch.exp(-energy(X0, positive_neurons, negative_neurons, args)))**2
            print(f'Target log-partition computation, iteration: {i+1}/{args.target_log_partition_samples//100000}. Time elapsed: {time.time() - start}.')
        partition = partition/(args.target_log_partition_samples//100000)
        partition_variance = partition_variance/(args.target_log_partition_samples//100000)
        log_partition = np.log(partition) 
        log_partition_error = np.sqrt(partition_variance/((args.target_log_partition_samples//100000)*100000))
    return log_partition, log_partition_error
'''    
def f2_kernel_evaluation(X0, X1, args, fill_diag = True, verbose = False):
    if not args.sphere:
        ones_d0 = torch.ones(X0.shape[0],1).float()
        X0 = torch.cat((X0,ones_d0), 1)
        ones_d1 = torch.ones(X1.shape[0],1).float()
        X1 = torch.cat((X1,ones_d1), 1)
    if args.signed_quadratic:
        X0 = torch.sign(X0)*X0**2
        X1 = torch.sign(X1)*X1**2
    inner_prod = torch.matmul(X0,X1.t())
    normsX0 = torch.norm(X0, dim=1, p=2)
    normsX1 = torch.norm(X1, dim=1, p=2)
    if fill_diag:
        cosines = (0.999999*inner_prod/(normsX0.unsqueeze(1)*normsX1.unsqueeze(0))).fill_diagonal_(fill_value = 1)
    else:
        cosines = 0.999999*inner_prod/(normsX0.unsqueeze(1)*normsX1.unsqueeze(0))
    values = normsX0.unsqueeze(1)*normsX1.unsqueeze(0)*((np.pi-torch.acos(cosines))*inner_prod \
            + torch.sqrt(1-cosines*cosines))/(2*np.pi*(args.d))
    #print(float(torch.max(torch.abs(cosines))))
    return values

def empirical_f2_kernel_evaluation(X0, X1, args):
    if args.signed_quadratic:
        X0 = torch.sign(X0)*X0**2
        X1 = torch.sign(X1)*X1**2
    features = torch.randn(args.d,100000)
    features = torch.nn.functional.normalize(features, p=2, dim=0)
    values = torch.mean(nn.functional.relu(torch.matmul(X0, features)).unsqueeze(1)*nn.functional.relu(torch.matmul(X1, features)).unsqueeze(0), dim=2)
    return values

def compare_kernels(args):
    angles = torch.tensor(np.linspace(0,2*np.pi,1000))
    X0 = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1).float()
    X0 = torch.sign(X0)*torch.sqrt(torch.abs(X0))
    X1 = torch.tensor([1.0,0.0]).unsqueeze(0).float()
    X1 = torch.sign(X1)*torch.sqrt(torch.abs(X1))
    values_theory = f2_kernel_evaluation(X0, X1, args).squeeze(1)
    values_empirical = empirical_f2_kernel_evaluation(X0, X1, args).squeeze(1)
    plt.figure(figsize=(6,6))
    plt.plot(angles, values_theory, label='Theoretical')
    #print(values_theory[:10])
    plt.plot(angles, values_empirical, label='Empirical')
    plt.legend()
    plt.savefig(f'figures/kernel_comparison_{args.d}.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

def gradient_function(X0, X1, X2, target_energy_average, args):
    X0.requires_grad_()
    #print(f'X0 shape: {X0.shape}, X1 shape: {X1.shape}, X2 shape: {X2.shape}.')
    fun_value = 2*torch.sum(torch.mean(f2_kernel_evaluation(X0, X1, args), dim=1) - torch.mean(f2_kernel_evaluation(X0, X2, args), dim=1))
    gradient = torch.autograd.grad(fun_value, X0)[0]
    X0.detach_()
    
    #gradient = torch.zeros_like(X0) ##to be removed
    return gradient

def regression_loss(tr_samples,Xtr):
    return torch.mean(f2_kernel_evaluation(tr_samples,tr_samples,args)) + torch.mean(f2_kernel_evaluation(Xtr,Xtr,args)) - 2*torch.mean(f2_kernel_evaluation(tr_samples,Xtr,args))

def shifted_likelihood_target_model(X, positive_Win, negative_Win, args):
    if args.sphere:
        if not args.add_gaussian:
            return -torch.mean(energy_sphere(X, positive_Win, negative_Win, args))
        else:
            return -torch.mean(energy_sphere(X, positive_Win, negative_Win, args) + 0.5*torch.norm(X, dim=1)**2)
    else:
        if not args.add_gaussian:
            return -torch.mean(energy(X, positive_Win, negative_Win, args))
        else:
            return -torch.mean(energy_sphere(X, positive_Win, negative_Win, args) + 0.5*torch.norm(X, dim=1)**2)

def learned_energy(X,tr_samples,Xtr, args):
    integral = torch.mean(f2_kernel_evaluation(X,tr_samples,args), dim=1) - torch.mean(f2_kernel_evaluation(X,Xtr,args), dim=1)
    if args.EBM_flow:
        return args.beta_train*integral/np.sqrt(regression_loss(tr_samples,Xtr))
    else:
        return 2*args.beta_train*integral*np.sqrt(regression_loss(tr_samples,Xtr))

def teacher_samples_2(args):
    fname = os.path.join('data_2', f'data_sphere_signedq_{args.target_neurons}_{args.beta}_{args.d}_{args.k}_{args.pre_target_neurons}_{args.seed}.pkl')
    
    if not args.recompute_data and os.path.exists(fname):
        Xtr, Xtr_target, Xtr_target_negative = pickle.load(open(fname, 'rb'))
        return Xtr, Xtr_target, Xtr_target_negative
    else:
        Xtr_target_full = torch.randn(2000,args.d)
        if args.signed_quadratic:
            Xtr_target_full = torch.sqrt(torch.abs(Xtr_target_full))*torch.sign(Xtr_target_full)
        acceptance_prob = 0.5*torch.sign(Xtr_target_full[:,0])+0.5
        acceptance_vector = torch.bernoulli(acceptance_prob)
        accepted_rows = []
        accepted_rows_negative = []
        for i in range(2000):
            if acceptance_vector[i] == 1:
                accepted_rows.append(i)
            else:
                accepted_rows_negative.append(i)
        accepted_rows_tensor = torch.tensor(accepted_rows).unsqueeze(1).expand([len(accepted_rows),args.d])
        accepted_rows_negative_tensor = torch.tensor(accepted_rows_negative).unsqueeze(1).expand([len(accepted_rows_negative),args.d])
        Xtr_target = torch.gather(Xtr_target_full, 0, accepted_rows_tensor)
        Xtr_target_negative = torch.gather(Xtr_target_full, 0, accepted_rows_negative_tensor)
        f2_norm = args.beta*np.sqrt(torch.mean(f2_kernel_evaluation(Xtr_target, Xtr_target, args)) + torch.mean(f2_kernel_evaluation(Xtr_target_negative, Xtr_target_negative, args)) -2*torch.mean(f2_kernel_evaluation(Xtr_target, Xtr_target_negative, args)))
        print(f'Xtr_target done. F2 norm: {f2_norm}.')
        
        min_energy = 10000
        for j in range(args.total_n_samples_2//100000):
            start = time.time()
            X0 = torch.randn(100000, args.d)
            if args.signed_quadratic:
                X0 = torch.sqrt(torch.abs(X0))*torch.sign(X0)
            X_energy = args.beta*(torch.mean(f2_kernel_evaluation(X0, Xtr_target, args), dim=1) - torch.mean(f2_kernel_evaluation(X0, Xtr_target_negative, args), dim=1))
            #X_energy = torch.zeros_like(X0[:,0]) ##to be removed
            #X_energy = args.beta*(-torch.mean(f2_kernel_evaluation(X0, Xtr_target_negative, args), dim=1))
            if torch.min(X_energy) < min_energy:
                min_energy = torch.min(X_energy)
        print(f'Minimum energy computed: {min_energy}.')
        
        for j in range(args.total_n_samples_2//100000):
            start = time.time()
            X0 = torch.randn(100000, args.d)
            if args.signed_quadratic:
                X0 = torch.sqrt(torch.abs(X0))*torch.sign(X0)
            X_energy = args.beta*(torch.mean(f2_kernel_evaluation(X0, Xtr_target, args), dim=1) - torch.mean(f2_kernel_evaluation(X0, Xtr_target_negative, args), dim=1))
            #X_energy = args.beta*(-torch.mean(f2_kernel_evaluation(X0, Xtr_target_negative, args), dim=1))
            #X_energy = torch.zeros_like(X0[:,0])  ##to be removed
            print(f'Average energy: {torch.mean(X_energy)}. Minimum energy: {torch.min(X_energy)}. Maximum energy: {torch.max(X_energy)}')
            
            if torch.min(X_energy) < min_energy:
                min_energy = torch.min(X_energy)
            X_density = torch.exp(-X_energy + min_energy)
        
            acceptance_vector = torch.bernoulli(X_density)
            #acceptance_vector = torch.ones_like(X_density) ##to be removed
            #print(torch.norm(X_density, p=1), torch.norm(acceptance_vector, p=1))
            accepted_rows = []
            for i in range(100000):
                if acceptance_vector[i] == 1:
                    accepted_rows.append(i)
            if args.sphere:
                accepted_rows_tensor = torch.LongTensor(accepted_rows).unsqueeze(1).expand([len(accepted_rows),args.d])
            else:
                accepted_rows_tensor = torch.tensor(accepted_rows).unsqueeze(1).expand([len(accepted_rows),args.d-1])
            if j==0:
                print('types:', X0.dtype, accepted_rows_tensor.dtype)
                X = torch.gather(X0, 0, accepted_rows_tensor)
                print(f'Sample batch {j+1}/{args.total_n_samples_2//100000} done in {time.time()-start}. {X.shape[0]} more samples.')
            else:
                samples = torch.gather(X0, 0, accepted_rows_tensor)
                X = torch.cat((X,samples),0)
                print(f'Sample batch {j+1}/{args.total_n_samples_2//100000} done in {time.time()-start}. {samples.shape[0]} more samples.')
        print(f'X size: {X.shape[0]}')
        if not args.add_gaussian and not os.path.exists('data_2'):
            os.makedirs('data_2')
        pickle.dump((X, Xtr_target, Xtr_target_negative), open(fname, 'wb'))
        return X, Xtr_target, Xtr_target_negative
    
    
def shifted_likelihood_trained_model(tr_samples, Xtr, positive_Win, negative_Win, args):
    first_term = -torch.mean(learned_energy(Xtr,tr_samples,Xtr, args))
    if args.log_partition_from_uniform:
        X0 = torch.randn(1000,args.d)
        second_term = -torch.log(torch.mean(torch.exp(-learned_energy(X0,tr_samples,Xtr, args))))
        std_second_term = torch.std(torch.exp(-learned_energy(X0,tr_samples,Xtr, args)))
        error_second_term = std_second_term/torch.mean(torch.exp(-learned_energy(X0,tr_samples,Xtr, args)))
    else:
        if args.sphere:
            if not args.add_gaussian:
                second_term = -torch.log(torch.mean(torch.exp(-learned_energy(Xtr,tr_samples,Xtr, args) + energy_sphere(Xtr, positive_Win, negative_Win, args))))
                std_second_term = torch.std(torch.exp(-learned_energy(Xtr,tr_samples,Xtr, args) + energy_sphere(Xtr, positive_Win, negative_Win, args)))
                error_second_term = std_second_term/(torch.mean(torch.exp(-learned_energy(Xtr,tr_samples,Xtr, args) + energy_sphere(Xtr, positive_Win, negative_Win, args)))*np.sqrt(Xtr.shape[0]))
            else:
                second_term = -torch.log(torch.mean(torch.exp(-learned_energy(Xtr,tr_samples,Xtr, args) + energy_sphere(Xtr, positive_Win, negative_Win, args) + 0.5*torch.norm(Xtr, dim=1)**2)))
                std_second_term = torch.std(torch.exp(-learned_energy(Xtr,tr_samples,Xtr, args) + energy_sphere(Xtr, positive_Win, negative_Win, args) + 0.5*torch.norm(Xtr, dim=1)**2))
                error_second_term = std_second_term/(torch.mean(torch.exp(-learned_energy(Xtr,tr_samples,Xtr, args) + energy_sphere(Xtr, positive_Win, negative_Win, args) + 0.5*torch.norm(Xtr, dim=1)**2))*np.sqrt(Xtr.shape[0]))
        else:
            if not args.add_gaussian:
                second_term = -torch.log(torch.mean(torch.exp(-learned_energy(Xtr,tr_samples,Xtr, args) + energy(Xtr, positive_Win, negative_Win, args))))
                std_second_term = torch.std(torch.exp(-learned_energy(Xtr,tr_samples,Xtr, args) + energy(Xtr, positive_Win, negative_Win, args)))
                error_second_term = std_second_term/(torch.mean(torch.exp(-learned_energy(Xtr,tr_samples,Xtr, args) + energy(Xtr, positive_Win, negative_Win, args)))*np.sqrt(Xtr.shape[0]))
            else:
                second_term = -torch.log(torch.mean(torch.exp(-learned_energy(Xtr,tr_samples,Xtr, args) + energy(Xtr, positive_Win, negative_Win, args) + 0.5*torch.norm(Xtr, dim=1)**2)))
                std_second_term = torch.std(torch.exp(-learned_energy(Xtr,tr_samples,Xtr, args) + energy(Xtr, positive_Win, negative_Win, args) + 0.5*torch.norm(Xtr, dim=1)**2))
                error_second_term = std_second_term/(torch.mean(torch.exp(-learned_energy(Xtr,tr_samples,Xtr, args) + energy(Xtr, positive_Win, negative_Win, args) + 0.5*torch.norm(Xtr, dim=1)**2))*np.sqrt(Xtr.shape[0]))
        #second_term, error_second_term = log_partition_trained_model(tr_samples, Xtr, positive_Win, negative_Win, args)
    print(f'Shifted likelihood first term: {first_term}. Shifted likelihood second term: {second_term}.')
    return first_term + second_term, error_second_term

def log_partition_trained_model(tr_samples, Xtr, positive_Win, negative_Win, args):
    log_partition_target_computation = target_log_partition(positive_neurons, negative_neurons, args)
    if args.sphere:
        if not args.add_gaussian:
            log_partition_difference = torch.log(torch.mean(torch.exp(-learned_energy(Xtr,tr_samples,Xtr, args) + energy_sphere(Xtr, positive_Win, negative_Win, args))))
            log_partition_difference_std = torch.std(torch.exp(-learned_energy(Xtr,tr_samples,Xtr, args) + energy_sphere(Xtr, positive_Win, negative_Win, args)))
            log_partition_difference_error = log_partition_difference_std/(torch.mean(torch.exp(-learned_energy(Xtr,tr_samples,Xtr, args) + energy_sphere(Xtr, positive_Win, negative_Win, args)))*np.sqrt(Xtr.shape[0]))
        else:
            log_partition_difference = torch.log(torch.mean(torch.exp(-learned_energy(Xtr,tr_samples,Xtr, args) + energy_sphere(Xtr, positive_Win, negative_Win, args) + 0.5*torch.norm(Xtr, dim=1)**2)))
            log_partition_difference_std = torch.std(torch.exp(-learned_energy(Xtr,tr_samples,Xtr, args) + energy_sphere(Xtr, positive_Win, negative_Win, args) + 0.5*torch.norm(Xtr, dim=1)**2))
            log_partition_difference_error = log_partition_difference_std/(torch.mean(torch.exp(-learned_energy(Xtr,tr_samples,Xtr, args) + energy_sphere(Xtr, positive_Win, negative_Win, args) + 0.5*torch.norm(Xtr, dim=1)**2))*np.sqrt(Xtr.shape[0]))
    else:
        if not args.add_gaussian:
            log_partition_difference = torch.log(torch.mean(torch.exp(-learned_energy(Xtr,tr_samples,Xtr, args) + energy(Xtr, positive_Win, negative_Win, args))))
            log_partition_difference_std = torch.std(torch.exp(-learned_energy(Xtr,tr_samples,Xtr, args) + energy(Xtr, positive_Win, negative_Win, args)))
            log_partition_difference_error = log_partition_difference_std/(torch.mean(torch.exp(-learned_energy(Xtr,tr_samples,Xtr, args) + energy(Xtr, positive_Win, negative_Win, args)))*np.sqrt(Xtr.shape[0]))
        else:
            log_partition_difference = torch.log(torch.mean(torch.exp(-learned_energy(Xtr,tr_samples,Xtr, args) + energy(Xtr, positive_Win, negative_Win, args) + 0.5*torch.norm(Xtr, dim=1)**2)))
            log_partition_difference_std = torch.std(torch.exp(-learned_energy(Xtr,tr_samples,Xtr, args) + energy(Xtr, positive_Win, negative_Win, args) + 0.5*torch.norm(Xtr, dim=1)**2))
            log_partition_difference_error = log_partition_difference_std/(torch.mean(torch.exp(-learned_energy(Xtr,tr_samples,Xtr, args) + energy(Xtr, positive_Win, negative_Win, args) + 0.5*torch.norm(Xtr, dim=1)**2))*np.sqrt(Xtr.shape[0]))
    return log_partition_difference + log_partition_target_computation[0], log_partition_difference_error + log_partition_target_computation[1]

def regularized_regression(tr_samples, Xtr, positive_Win, negative_Win, args):
    log_partition, error_log_partition = log_partition_trained_model(tr_samples, Xtr, positive_Win, negative_Win, args)
    first_term = -torch.mean(learned_energy(tr_samples,tr_samples,Xtr, args)) - log_partition
    error_first_term = torch.std(learned_energy(tr_samples,tr_samples,Xtr, args))/np.sqrt(tr_samples.shape[0]) + error_log_partition
    if args.EBM_flow:
        second_term = args.beta_train*np.sqrt(regression_loss(tr_samples,Xtr))
    else:
        second_term = args.beta_train*regression_loss(tr_samples,Xtr)
    return first_term + second_term, error_first_term, first_term, second_term
'''
def loss_gradient_weights(Win, wout, X, Xmcmc):
    n, nmcmc = X.shape[0], Xmcmc.shape[0]
    neurons = Win.shape[1]
    gd_data = torch.sum(nn.functional.relu(torch.matmul(X, Win)), dim = 0) / n / neurons
    gd_model = torch.sum(nn.functional.relu(torch.matmul(Xmcmc, Win)), dim = 0) / nmcmc /neurons
    return gd_data - gd_model, gd_model

def explicit_energy(x, Win, wout):
    neurons = Win.shape[1]
    return torch.sum(nn.functional.relu(torch.matmul(x, Win)) * wout, dim=1)/neurons
'''

def plot_trained_energy(tr_samples, Xtr, target_energy, args, iteration_n):
    angles = torch.tensor(np.linspace(0,2*np.pi,1000))
    X_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1).float()
    #print(X_angles.dtype, positive_neurons.dtype)
    trained_energy_angles = learned_energy(X_angles, tr_samples, Xtr, args)
    #print('energy_angles', energy_angles.shape, torch.min(energy_angles), torch.max(energy_angles))
    plt.figure(figsize=(4,3))
    angles_array = np.array(angles)
    #trained_energy_array = np.array(trained_energy_angles - torch.mean(trained_energy_angles))
    trained_energy_array = np.array(trained_energy_angles)
    #target_energy_array = np.array(target_energy - np.mean(target_energy))
    target_energy_array = np.array(target_energy)
    plt.plot(angles_array, trained_energy_array, label='Trained energy')
    plt.plot(angles_array, target_energy_array, label='Target energy')
    plt.ylabel('Energy')
    plt.xlabel('Angle')
    plt.legend()
    if not args.add_gaussian:
        plt.savefig(f'figures/trained_energy_{args.k}_{iteration_n}_{args.beta}.pdf', bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig(f'figures_gaussian/trained_energy_{args.k}_{iteration_n}_{args.beta}.pdf', bbox_inches='tight', pad_inches=0)
    
def scatter_plot_samples(tr_samples, Xtr, args, iteration_n):
    plt.figure(figsize=(6,6))
    tr_samples_dim1 = torch.sign(tr_samples[:,0])*tr_samples[:,0]**2
    tr_samples_dim2 = torch.sign(tr_samples[:,1])*tr_samples[:,1]**2
    Xtr_dim1 = torch.sign(Xtr[:(tr_samples.shape[0]),0])*Xtr[:(tr_samples.shape[0]),0]**2
    Xtr_dim2 = torch.sign(Xtr[:(tr_samples.shape[0]),1])*Xtr[:(tr_samples.shape[0]),1]**2
    plt.scatter(tr_samples_dim1,tr_samples_dim2, s=2, marker='^', facecolor='red', label=f'Generated. Norm std: {torch.std(torch.norm(tr_samples, dim=1))}')
    plt.scatter(Xtr_dim1,Xtr_dim2, s=2, marker='o', facecolor='blue', label=f'Target. Norm std: {torch.std(torch.norm(Xtr, dim=1))}')
    plt.title('Positions of generated samples')
    plt.legend()
    if not args.add_gaussian:
        plt.savefig(f'figures/generated_samples_{args.k}_{iteration_n}_{args.beta}.pdf', bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig(f'figures_gaussian/generated_samples_{args.k}_{iteration_n}_{args.beta}.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()
    
def empirical_CDF(tr_samples, Xtr, args, iteration_n):
    plt.figure(figsize=(6,6))
    #tr_samples_dim1 = torch.sign(tr_samples[:,0])*tr_samples[:,0]**2
    #tr_samples_dim2 = torch.sign(tr_samples[:,1])*tr_samples[:,1]**2
    #complex_tr_samples = tr_samples_dim1 + tr_samples_dim2*1j
    complex_tr_samples = torch.view_as_complex(tr_samples)
    train_angles = torch.angle(complex_tr_samples)
    ecdf_train = ECDF(train_angles)
    #Xtr_dim1 = torch.sign(Xtr[:,0])*Xtr[:,0]**2
    #Xtr_dim2 = torch.sign(Xtr[:,1])*Xtr[:,1]**2
    #complex_Xtr = Xtr_dim1 + Xtr_dim2*1j
    complex_Xtr = torch.view_as_complex(Xtr)
    target_angles = torch.angle(complex_Xtr)
    ecdf_target = ECDF(target_angles)
    
    angle_axis = np.linspace(-np.pi,np.pi,1000)
    ecdf_train_points = ecdf_train(angle_axis)
    ecdf_target_points = ecdf_target(angle_axis)
    plt.plot(angle_axis, ecdf_train_points, label=f'Generated empirical CDF. {torch.std(torch.norm(tr_samples, dim=1))}')
    plt.plot(angle_axis, ecdf_target_points, label=f'Target empirical CDF. {torch.std(torch.norm(Xtr, dim=1))}')
    plt.title('Empirical CDFs')
    plt.legend()
    
    if not args.add_gaussian:
        plt.savefig(f'figures/samples_ecdf_{args.k}_{iteration_n}_{args.beta}.pdf', bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig(f'figures_gaussian/samples_ecdf_{args.k}_{iteration_n}_{args.beta}.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()
        
def scatter_plot_additional_samples(tr_samples, additional_samples, args, iteration_n):
    plt.figure(figsize=(6,6))
    tr_samples_dim1 = tr_samples[:,0]
    tr_samples_dim2 = tr_samples[:,1]
    additional_samples_dim1 = additional_samples[:,0]
    additional_samples_dim2 = additional_samples[:,1]
    plt.scatter(tr_samples_dim1,tr_samples_dim2, s=2, marker='^', facecolor='blue')
    plt.scatter(additional_samples_dim1,additional_samples_dim2, s=2, marker='o', facecolor='red')
    plt.title('Positions of generated samples')
    if not args.add_gaussian:
        plt.savefig(f'figures/generated_additional_samples_{args.k}_{iteration_n}_{args.beta}.pdf', bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig(f'figures_gaussian/generated_additional_samples_{args.k}_{iteration_n}_{args.beta}.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLE training')
    parser.add_argument('--name', default='Wasserstein EBM', help='exp name')
    parser.add_argument('--d', type=int, default=12, help='dimension of the data')
    parser.add_argument('--k', type=int, default=6, help='degree of Legendre polynomial')
    parser.add_argument('--beta', type=float, default=2.00, help='inverse temperature target')
    parser.add_argument('--beta_train', type=float, default=2.00, help='inverse temperature training')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--n_samples', type=int, default=10000, help='number of target samples used')
    parser.add_argument('--total_n_samples', type=int, default=1000000, help='total number of target samples')
    parser.add_argument('--total_n_samples_2', type=int, default=1000000, help='total number of target samples')
    parser.add_argument('--n_samples_val', type=int, default=10000, help='number of validation samples')
    parser.add_argument('--n_samples_te', type=int, default=10000, help='number of test samples')
    parser.add_argument('--n_samples_tr', type=int, default=1000, help='number of training samples')
    parser.add_argument('--pre_target_neurons', type=int, default=1000000, help='number of neurons')
    parser.add_argument('--target_neurons', type=int, default=10000, help='number of neurons')
    parser.add_argument('--n_iterations', type=int, default=1000, help='number of iterations')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--base_variance', type=float, default=1.00, help='learning rate')
    parser.add_argument('--recompute_data', action='store_true', help='recompute teacher samples')
    parser.add_argument('--sphere', action='store_true', help='domain of distribution is sphere instead of Euclidean')
    #parser.add_argument('--rmsprop', action='store_true', help='RMSProp instead of GD')
    parser.add_argument('--EBM_flow', action='store_true', help='run Wasserstein gradient flow on dual EBM loss (with sqrt)')
    parser.add_argument('--SGD', action='store_true', help='train using SGD instead of ERM')
    parser.add_argument('--batch_size_SGD', type=int, default=5000, help='size of batch for SGD')
    parser.add_argument('--compute_likelihood', action='store_true', help='compute the likelihood of the model')
    parser.add_argument('--log_partition_from_uniform', action='store_true', help='use uniform samples to compute log-partition')
    parser.add_argument('--add_gaussian', action='store_true', help='add gaussian energy to the target energy')
    parser.add_argument('--target_log_partition_samples', type=int, default=1000000, help='number of samples used in target log partition computation')
    parser.add_argument('--initialize_target', action='store_true', help='initialize samples at target particles')
    parser.add_argument('--signed_quadratic', action='store_true', help='use signed quadratic transformation of samples')
    parser.add_argument('--plot_figures', action='store_true', help='plot figures in dim 2')
    parser.add_argument('--additional_samples', action='store_true', help='sample additional samples')
    parser.add_argument('--data_2', action='store_true', help='use teacher_samples_2')
    
    #explicit training arguments
    parser.add_argument('--explicit_training', action='store_true', help='perform explicit training to compare')
    parser.add_argument('--neurons', action='store_true', help='number of neurons for explicit training')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay for EBMs')
    parser.add_argument('--eval_unif_samples', type=int, default=100000, help='number of uniform samples for evaluation')
    parser.add_argument('--eval_reps', type=int, default=1, help='number of averaging samples for cross entropy evaluation')
    parser.add_argument('--mcmc_samples', type=int, default=1000,
                        help='number of mcmc samples for computing gradients')
    
    
    args = parser.parse_args()
    
    #N_kd = (2*args.k + args.d - 2) * math.factorial(args.k + args.d - 3) / (math.factorial(args.k) * math.factorial(args.d -2))
    #print(f'N_kd: {N_kd}. sqrt(2/N_kd): {np.sqrt(2/N_kd)}. beta*sqrt(2/N_kd): {args.beta*np.sqrt(2/N_kd)}')
    if not args.data_2:
        f2_norm = compute_f2_norm(args)
        print(f'Beta of target measure: {args.beta}. F2 norm of target measure: {args.beta*f2_norm}')
    
    if not args.SGD:
        Xtr, Xval, Xte, positive_neurons, negative_neurons = teacher_samples(args)
    else:
        if not args.data_2:
            X, positive_neurons, negative_neurons = teacher_samples(args)
        else:
            X, Xtr_target, Xtr_target_negative = teacher_samples_2(args)
            _, positive_neurons, negative_neurons = teacher_samples(args)
            
    if args.data_2:
        f2_norm = args.beta*np.sqrt(torch.mean(f2_kernel_evaluation(Xtr_target, Xtr_target, args)))
        print(f'Teacher samples 2. F2 norm: {f2_norm}')
        
    if args.plot_figures:
        if not args.add_gaussian and not os.path.exists('figures'):
            os.makedirs('figures')
        if args.add_gaussian and not os.path.exists('figures_gaussian'):
            os.makedirs('figures_gaussian')
        
        if not args.data_2:
            plt.figure(figsize=(6,6))
            positive_neuron_dim1 = positive_neurons[0,:]
            positive_neuron_dim2 = positive_neurons[1,:]
            negative_neuron_dim1 = negative_neurons[0,:]
            negative_neuron_dim2 = negative_neurons[1,:]
            plt.scatter(positive_neuron_dim1,positive_neuron_dim2, s=2, marker='^', facecolor='blue')
            plt.scatter(negative_neuron_dim1,negative_neuron_dim2, s=2, marker='o', facecolor='red')
            plt.title('Positive and negative neurons')
            if not args.add_gaussian:
                plt.savefig(f'figures/target_neurons_{args.k}_{args.beta}.pdf', bbox_inches='tight', pad_inches=0)
            else:
                plt.savefig(f'figures_gaussian/target_neurons_{args.k}_{args.beta}.pdf', bbox_inches='tight', pad_inches=0)
            plt.close()
        
        plt.figure(figsize=(6,6))
        if args.SGD:
            print('X shape 0', X.shape[0])
            indices = torch.tensor(np.random.choice(X.shape[0], 30000, replace=False))
            Xtr = torch.index_select(X, 0, indices)
        Xtr_dim1 = torch.sign(Xtr[:,0])*Xtr[:,0]**2
        Xtr_dim2 = torch.sign(Xtr[:,1])*Xtr[:,1]**2
        plt.scatter(Xtr_dim1,Xtr_dim2, s=2, marker='^', facecolor='blue')
        plt.title('Samples from target')
        if not args.add_gaussian:
            plt.savefig(f'figures/target_samples_{args.k}_{args.beta}.pdf', bbox_inches='tight', pad_inches=0)
        else:
            plt.savefig(f'figures_gaussian/target_samples_{args.k}_{args.beta}.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()
        
        angles = torch.tensor(np.linspace(0,2*np.pi,1000))
        X_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1).float()
        #print(X_angles.dtype, positive_neurons.dtype)
        if not args.data_2:
            energy_angles = energy_sphere(X_angles, positive_neurons, negative_neurons, args)
        else:
            energy_angles = args.beta*(torch.mean(f2_kernel_evaluation(X_angles, Xtr_target, args), dim=1) - torch.mean(f2_kernel_evaluation(X_angles, Xtr_target_negative, args), dim=1))
        print('energy_angles', energy_angles.shape, torch.min(energy_angles), torch.max(energy_angles))
        target_energy_average = torch.mean(energy_angles)
        plt.figure(figsize=(4,3))
        angles_array = np.array(angles)
        energy_array = np.array(energy_angles)
        plt.plot(angles_array, energy_array, label='Target energy')
        plt.ylabel('Energy')
        plt.xlabel('Angle')
        plt.legend()
        if not args.add_gaussian:
            plt.savefig(f'figures/target_energy_{args.k}_{args.beta}.pdf', bbox_inches='tight', pad_inches=0)
        else:
            plt.savefig(f'figures_gaussian/target_energy_{args.k}_{args.beta}.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()
        
        compare_kernels(args)
        
        if args.SGD:
            indices2 = torch.tensor(np.random.choice(X.shape[0], 30000, replace=False))
            Xtr2 = torch.index_select(X, 0, indices[:15000])
            Xtr3 = torch.index_select(X, 0, indices[15000:])
            plot_trained_energy(Xtr2, Xtr3, energy_array, args, -1)    
        
    if args.compute_likelihood:
        start = time.time()
        log_partition_target_computation = target_log_partition(positive_neurons, negative_neurons, args)
        if args.SGD:
            indices = torch.tensor(np.random.choice(X.shape[0], 50000, replace=False))
            Xtr = torch.index_select(X, 0, indices)
        shifted_likelihood_computation = shifted_likelihood_target_model(Xtr, positive_neurons, negative_neurons, args)
        likelihood = shifted_likelihood_computation - log_partition_target_computation[0]
        likelihood_error = log_partition_target_computation[1]
        print(f'Likelihood of target energy: {likelihood} ({shifted_likelihood_computation},{- log_partition_target_computation[0]}). Error of likelihood: {likelihood_error}. Duration of likelihood computation: {time.time() - start}.')
    
    if args.sphere:
        if args.initialize_target:
            if args.SGD:
                indices = torch.tensor(np.random.choice(X.shape[0], args.n_samples_tr, replace=False))
                tr_samples = torch.index_select(X, 0, indices)
                #print('mean tr_samples', torch.mean(tr_samples), tr_samples.shape)
            else:
                indices = torch.tensor(np.random.choice(Xtr.shape[0], args.n_samples_tr, replace=False))
                tr_samples = torch.index_select(Xtr, 0, indices)
        else:
            tr_samples = torch.randn(args.n_samples_tr,args.d)/(args.d*50)
            if args.additional_samples:
                additional_samples = torch.randn(5000,args.d)/(args.d*50)
            #tr_samples = torch.nn.functional.normalize(tr_samples, p=2, dim=1)
    else:
        if args.initialize_target:
            if args.SGD:
                indices = torch.tensor(np.random.choice(X.shape[0], args.n_samples_tr, replace=False))
                tr_samples = torch.index_select(X, 0, indices)
            else:
                indices = torch.tensor(np.random.choice(Xtr.shape[0], args.n_samples_tr, replace=False))
                tr_samples = torch.index_select(Xtr, 0, indices)
        else:
            tr_samples = torch.randn(args.n_samples_tr,args.d-1)/args.n_samples_tr
            #tr_samples = torch.nn.functional.normalize(tr_samples, p=2, dim=1)
    print('Initialization complete')
    
    if args.plot_figures:
        if args.SGD:
            indices = torch.tensor(np.random.choice(X.shape[0], 10000, replace=False))
            Xtr = torch.index_select(X, 0, indices)
        plot_trained_energy(tr_samples, Xtr, energy_array, args, -2)
        if not args.additional_samples:
            scatter_plot_samples(tr_samples, Xtr, args, -2)
            empirical_CDF(tr_samples, Xtr, args, -2)
        else:
            scatter_plot_additional_samples(tr_samples, additional_samples, args, -2)
            
    print('Initial plots done')
    
    #print('beta', args.beta, 'beta train', args.beta_train)
    
    if args.explicit_training:
        Win = torch.randn(args.d, args.neurons)
        wout = torch.zeros(1, args.neurons)
    
    for t in range(args.n_iterations):
        tstart = time.time()
        #kernel_eval = f2_kernel_evaluation(tr_samples, tr_samples, args)
        #print('kernel evaluation:', kernel_eval[0:5,0:5])
        
        #Implicit updates
        if args.SGD:
            start = time.time()
            #indices = torch.randint(0, X.shape[0], (args.batch_size_SGD,))
            indices = torch.tensor(np.random.choice(X.shape[0], args.batch_size_SGD, replace=False))
            Xtr = torch.index_select(X, 0, indices)
        gradient = gradient_function(tr_samples, tr_samples, Xtr, target_energy_average, args)
        #print('gradient mean', torch.mean(gradient))
        if args.sphere:
            if args.EBM_flow:
                sqrt_regression_loss = np.sqrt(regression_loss(tr_samples,Xtr))
                gradient = gradient/(2*sqrt_regression_loss)
            if not args.signed_quadratic:
                gradient = gradient + tr_samples/args.beta_train #args.base_variance
            else:
                gradient = gradient + 2*tr_samples**3/args.beta_train - 1/(args.beta_train*tr_samples) #the last term corresponds to the gradient of the log-determinant
            noise = np.sqrt(2*args.lr/args.beta_train)*torch.randn(args.n_samples_tr,args.d)
            tr_samples.sub_(args.lr*gradient - noise)
            #tr_samples = torch.nn.functional.normalize(tr_samples, p=2, dim=1)
        else:
            if args.EBM_flow:
                sqrt_regression_loss = np.sqrt(regression_loss(tr_samples,Xtr))
                gradient = gradient/sqrt_regression_loss
            gradient = gradient + tr_samples/(args.base_variance*args.beta_train)
            noise = np.sqrt(2*args.lr/args.beta_train)*torch.randn(args.n_samples_tr,args.d-1)
            tr_samples.sub_(args.lr*gradient - noise)
            
        if args.SGD and args.additional_samples:
            gradient = gradient_function(additional_samples, tr_samples, Xtr, args)
            if args.sphere:
                if args.EBM_flow:
                    gradient = gradient/sqrt_regression_loss
                if not args.signed_quadratic:
                    gradient = gradient + additional_samples/args.beta_train #args.base_variance
                else:
                    gradient = gradient + 2*additional_samples**3/args.beta_train - 1/(args.beta_train*additional_samples) #the last term corresponds to the gradient of the log-determinant
                noise = np.sqrt(2*args.lr/args.beta_train)*torch.randn(5000,args.d)
                additional_samples.sub_(args.lr*gradient - noise)
        '''    
        #Explicit updates
        if args.explicit_training:
            tstart = time.time()
            potential = lambda x: explicit_energy(x, Win, wout)

            Xmcmc_all = torch.tensor(get_samples_uniform_proposal(potential, args.d, n_samples=args.mcmc_samples, step=0.1, burn=2, burn_in=5000))
            Xmcmc = Xmcmc_all[:args.mcmc_samples]
            dt_mcmc = time.time() - tstart

            gradout, gradout_model = loss_gradient_weights(Win, wout, X, Xmcmc)
            out_step = args.lr * args.neurons * (gradout.data + args.wd * wout.data)
            wout.sub_(out_step)
        '''
        print(f'Std of target: {torch.std(torch.norm(Xtr, dim=1))}. Std of trained model: {torch.std(torch.norm(tr_samples, dim=1))}')
        if torch.std(torch.norm(tr_samples, dim=1)) > 1 or torch.max(torch.abs(gradient)) > 4000:
            print('high norm or high gradient', torch.max(torch.abs(gradient)), torch.max(torch.abs(2*tr_samples**3/args.beta_train)), -torch.max(torch.abs(1/(args.beta_train*tr_samples))), torch.max(torch.abs(gradient- 2*tr_samples**3/args.beta_train + 1/(args.beta_train*tr_samples))))
            scatter_plot_samples(tr_samples, Xtr, args, t)
        time_passed = time.time()-tstart
        if t%5 == 0:
            if not args.SGD:
                print(f'Iteration: {t}/{args.n_iterations}. Time elapsed: {time_passed}.')
            else:
                print(f'Iteration: {t}/{args.n_iterations}. Time elapsed: {time_passed}. Xtr size: {Xtr.shape[0]},{Xtr.shape[1]}')
            print(f'Regression loss: {regression_loss(tr_samples,Xtr)}')
            if args.additional_samples:
                print(f'Average norm of additional_samples: {torch.mean(torch.norm(additional_samples, dim=1))}')
                print(f'Average norm of tr_samples: {torch.mean(torch.norm(tr_samples, dim=1))}')
                print(additional_samples[:10,:])
        if t%100 == 0 and args.compute_likelihood:
            start = time.time()
            if args.SGD:
                indices = torch.tensor(np.random.choice(X.shape[0], 10000, replace=False))
                Xtr = torch.index_select(X, 0, indices)
            shifted_likelihood_computation = shifted_likelihood_trained_model(tr_samples, Xtr, positive_neurons, negative_neurons, args) 
            likelihood = shifted_likelihood_computation[0] - log_partition_target_computation[0]
            likelihood_error = shifted_likelihood_computation[1] + log_partition_target_computation[1]
            print(f'Likelihood: {likelihood} ({shifted_likelihood_computation[0]},{- log_partition_target_computation[0]}). Error : {likelihood_error} ({shifted_likelihood_computation[1]}+{log_partition_target_computation[1]}). Duration of likelihood computation: {time.time() - start}.')
            start = time.time()
            reg_regression, reg_regression_error, first_term, second_term = regularized_regression(tr_samples, Xtr, positive_neurons, negative_neurons, args)
            print(f'Regularized regression loss: {reg_regression} ({first_term},{second_term}). Error: {reg_regression_error}. Duration of regression computation: {time.time() - start}.')
            #print(f'Variance of target: {torch.std(torch.norm(Xtr, dim=1))}. Variance of trained model: {torch.std(torch.norm(tr_samples, dim=1))}')
        if t%100 == 0 and args.plot_figures:
            if not args.data_2:
                energy_angles = energy_sphere(X_angles, positive_neurons, negative_neurons, args)
            else:
                energy_angles = args.beta*(torch.mean(f2_kernel_evaluation(X_angles, Xtr_target, args), dim=1) - torch.mean(f2_kernel_evaluation(X_angles, Xtr_target_negative, args), dim=1))
            energy_array = np.array(energy_angles)
            plot_trained_energy(tr_samples, Xtr, energy_array, args, t)
            scatter_plot_samples(tr_samples, Xtr, args, t)
            empirical_CDF(tr_samples, Xtr, args, t)
            if args.additional_samples:
                print(f'tr_samples has {torch.sum(torch.isnan(tr_samples))} Nan values.' )
                print(f'additional_samples has {torch.sum(torch.isnan(additional_samples))} Nan values.' )
                additional_samples[additional_samples != additional_samples] = 0
                print(f'additional_samples has {torch.sum(torch.isnan(additional_samples))} Nan values.' )
                scatter_plot_additional_samples(tr_samples, additional_samples, args, t)
                
            '''
            if args.explicit_training:
                start = time.time()
                ce_f, ce_f_stuff = free_energy_sampling(Win, wout, n_uniform_samples=args.eval_unif_samples, reps=args.eval_reps, get_stuff=True)
                train_cea = cross_entropy_avgterm(Win, wout, Xtr)
                train_ce = train_cea + ce_f
                print(f'Likelihood of the explicit model: {-train_ce}. Duration of explicit model likelihood computation: {time.time() - start}')
            '''

    
