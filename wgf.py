import argparse
import numpy as np
import os, sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.special as ss
import time

#from mala import get_samples, get_samples_uniform_proposal
#from util import get_teacher, teacher_samples, to_bool

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
    return args.beta * (torch.sum(nn.functional.relu(torch.matmul(X, positive_Win)), dim=1) - torch.sum(nn.functional.relu(torch.matmul(X, negative_Win)), dim=1))/neurons

def energy_sum_sphere(X, positive_Win, negative_Win, args):
    neurons = positive_Win.shape[1] + negative_Win.shape[1]
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
            acceptance_prob = torch.nn.functional.relu(torch.from_numpy(0.99*legendre_k_d(X0[:,args.d-3]) + 0.495*legendre_k_d(X0[:,args.d-2])))
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
            acceptance_prob = torch.nn.functional.relu(torch.from_numpy(-0.99*legendre_k_d(X0[:,args.d-3]) -0.495*legendre_k_d(X0[:,args.d-2])))
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

def teacher_samples(args):
    if args.sphere:
        fname = os.path.join('data', f'data_sphere_{args.target_neurons}_{args.beta}_{args.d}_{args.k}_{args.pre_target_neurons}_{args.seed}.pkl')
    else:
        fname = os.path.join('data', f'data_{args.target_neurons}_{args.beta}_{args.d}_{args.k}_{args.pre_target_neurons}_{args.seed}.pkl')
    
    if not args.recompute_data and os.path.exists(fname):
        positive_neurons = get_positive_target_neurons(args)
        if positive_neurons.shape[1] > 3000:
            positive_neurons = positive_neurons[:,:3000]
        negative_neurons = get_negative_target_neurons(args)
        if negative_neurons.shape[1] > 3000:
            negative_neurons = negative_neurons[:,:3000]
        Xtr, Xval, Xte = pickle.load(open(fname, 'rb'))
        print(f'Xtr shape: {Xtr.shape[0]},{Xtr.shape[1]}, Xval: {Xval.shape[0]},{Xval.shape[1]}, Xte: {Xte.shape[0]},{Xte.shape[1]}')
        return Xtr, Xval, Xte, positive_neurons, negative_neurons
    else:
        positive_neurons = get_positive_target_neurons(args)
        if positive_neurons.shape[1] > 3000:
            positive_neurons = positive_neurons[:,:3000]
        negative_neurons = get_negative_target_neurons(args)
        if negative_neurons.shape[1] > 3000:
            negative_neurons = negative_neurons[:,:3000]
        n_gd_samples = 10000
        if args.sphere:
            gd_particles = torch.randn(n_gd_samples,args.d)
            gd_particles = torch.nn.functional.normalize(gd_particles, p=2, dim=1)
        else:
            gd_particles = torch.randn(n_gd_samples,args.d-1)
        for gd_iteration in range(2000):
            gd_particles.requires_grad_()
            if args.sphere:
                fun_value = energy_sum_sphere(gd_particles, positive_neurons, negative_neurons, args)
            else:
                fun_value = energy_sum(gd_particles, positive_neurons, negative_neurons, args)
            gradient = torch.autograd.grad(fun_value, gd_particles)[0]
            gd_particles.detach_()
            gd_particles.sub_(0.2*gradient)
            if args.sphere:
                gd_particles = torch.nn.functional.normalize(gd_particles, p=2, dim=1)
            if gd_iteration%500==0:
                if args.sphere:
                    final_values = energy_sphere(gd_particles, positive_neurons, negative_neurons, args)
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
        
        for j in range(args.total_n_samples//100000):
            start = time.time()
            if args.sphere:
                X0 = torch.randn(100000,args.d)
                #X = torch.nn.functional.normalize(X, p=2, dim=1)
                X_energy = energy_sphere(X0, positive_neurons, negative_neurons, args) + 0.5*torch.norm(X0, dim=1)**2
            else:
                X0 = torch.randn(100000,args.d-1)
                #X = torch.nn.functional.normalize(X, p=2, dim=1)
                X_energy = energy(X0, positive_neurons, negative_neurons, args)
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
                accepted_rows_tensor = torch.tensor(accepted_rows).unsqueeze(1).expand([len(accepted_rows),args.d])
            else:
                accepted_rows_tensor = torch.tensor(accepted_rows).unsqueeze(1).expand([len(accepted_rows),args.d-1])
            if j==0:
                X = torch.gather(X0, 0, accepted_rows_tensor)
                print(f'Sample batch {j+1}/{args.total_n_samples//100000} done in {time.time()-start}. {X.shape[0]} more samples.')
            else:
                samples = torch.gather(X0, 0, accepted_rows_tensor)
                X = torch.cat((X,samples),0)
                print(f'Sample batch {j+1}/{args.total_n_samples//100000} done in {time.time()-start}. {samples.shape[0]} more samples.')
        print(f'X size: {X.shape[0]}')
        
        Xtr = X[:args.n_samples]
        Xval = X[args.n_samples:(args.n_samples+args.n_samples_val)]
        Xte = X[(args.n_samples+args.n_samples_val):(args.n_samples+args.n_samples_val+args.n_samples_te)]
        
        if not os.path.exists('data'):
            os.makedirs('data')
        pickle.dump((Xtr, Xval, Xte), open(fname, 'wb'))
        return Xtr, Xval, Xte, positive_neurons, negative_neurons
    
def f2_kernel_evaluation(X0, X1, args, fill_diag = True):
    if not args.sphere:
        ones_d0 = torch.ones(X0.shape[0],1).float()
        X0 = torch.cat((X0,ones_d0), 1)
        ones_d1 = torch.ones(X1.shape[0],1).float()
        X1 = torch.cat((X1,ones_d1), 1)
    inner_prod = torch.matmul(X0,X1.t())
    normsX0 = torch.norm(X0, dim=1, p=2)
    normsX1 = torch.norm(X1, dim=1, p=2)
    if fill_diag:
        cosines = (inner_prod/(normsX0.unsqueeze(1)*normsX1.unsqueeze(0))).fill_diagonal_(fill_value = 1)
    else:
        cosines = inner_prod/(normsX0.unsqueeze(1)*normsX1.unsqueeze(0))
    #print(cosines[0:5,0:5], )
    values = normsX0.unsqueeze(1)*normsX1.unsqueeze(0)*((np.pi-torch.acos(cosines))*inner_prod \
            + torch.sqrt(1-cosines*cosines))/(2*np.pi*(args.d+1))
    return values

def gradient_function(X0, X1, X2, args):
    X0.requires_grad_()
    #print(f'X0 shape: {X0.shape}, X1 shape: {X1.shape}, X2 shape: {X2.shape}.')
    fun_value = 2*torch.sum(torch.mean(f2_kernel_evaluation(X0, X1, args), dim=1) - torch.mean(f2_kernel_evaluation(X0, X2, args), dim=1))
    gradient = torch.autograd.grad(fun_value, X0)[0]
    X0.detach_()
    return gradient

def regression_loss(tr_samples,Xtr):
    return torch.mean(f2_kernel_evaluation(tr_samples,tr_samples, args)) + torch.mean(f2_kernel_evaluation(Xtr,Xtr, args)) - 2*torch.mean(f2_kernel_evaluation(tr_samples,Xtr, args))
    
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
    parser.add_argument('--rmsprop', action='store_true', help='RMSProp instead of GD')
    
    args = parser.parse_args()
    
    Xtr, Xval, Xte, positive_neurons, negative_neurons = teacher_samples(args)
    
    if args.sphere:
        tr_samples = torch.randn(args.n_samples_tr,args.d)
        #tr_samples = torch.nn.functional.normalize(tr_samples, p=2, dim=1)
    else:
        tr_samples = torch.randn(args.n_samples_tr,args.d-1)/args.n_samples_tr
        #tr_samples = torch.nn.functional.normalize(tr_samples, p=2, dim=1)
    
    #print('beta', args.beta, 'beta train', args.beta_train)
    
    for t in range(args.n_iterations):
        tstart = time.time()
        #kernel_eval = f2_kernel_evaluation(tr_samples, tr_samples)
        #print('kernel evaluation:', kernel_eval[0:5,0:5])
        gradient = gradient_function(tr_samples, tr_samples, Xtr, args)
        #print(gradient)
        if args.sphere:
            gradient = gradient + tr_samples/args.base_variance
            noise = np.sqrt(2*args.lr/args.beta_train)*torch.randn(args.n_samples_tr,args.d)
            tr_samples.sub_(args.lr*gradient - noise)
            #tr_samples = torch.nn.functional.normalize(tr_samples, p=2, dim=1)
        else:
            gradient = gradient + tr_samples/(args.base_variance*args.beta_train)
            noise = np.sqrt(2*args.lr/args.beta_train)*torch.randn(args.n_samples_tr,args.d-1)
            tr_samples.sub_(args.lr*gradient - noise)
        if t%5 == 0:
            print(f'Iteration: {t}/{args.n_iterations}')
            print(f'Regression loss: {regression_loss(tr_samples,Xtr)}')
        

    
