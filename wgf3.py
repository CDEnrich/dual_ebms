import argparse
import numpy as np
import math
import os, sys
import pickle
import torch
import torch.nn as nn
import scipy.special as ss
import matplotlib.pyplot as plt
import time
from statsmodels.distributions.empirical_distribution import ECDF

beta = 10
d = 5
t = 0
tmax = 2e2
dt = 1e-1

Nx = 2**12
x = torch.randn(Nx,d)
x = torch.nn.functional.normalize(x, p=2, dim=1)
#x = torch.linspace(0,1, steps=Nx+1)
#x = x[:Nx]

#Kernel
a = 10
#K1 = lambda x: ss.iv(0,2*a*torch.cos(np.pi*x))*np.exp(-2*a)
#dK1 = lambda x: -ss.iv(1,2*a*torch.cos(np.pi*x))*2*a*np.pi*torch.sin(np.pi*x)*np.exp(-2*a)
#phi1 = lambda x: torch.exp(a*torch.cos(2*np.pi*x)-a)
phi1 = lambda x,y: torch.exp(a*(torch.matmul(x,y.t()))-a)
dphi1 = lambda x,y: torch.exp(a*(torch.matmul(x,y.t()))-a).unsqueeze(2)*a*y.unsqueeze(0)
K1 = lambda x,y: ss.iv(0,2*a*torch.sqrt((1+torch.matmul(x,y.t()))/2))*np.exp(-2*a)
dK1 = lambda x,y: ss.iv(1,2*a*torch.sqrt((1+torch.matmul(x,y.t()))/2)).unsqueeze(2)*a*np.exp(-2*a)*y.unsqueeze(0)/(np.sqrt(2)*torch.sqrt(1+torch.matmul(x,y.t())).unsqueeze(2))
#def dK1(x,y):
#    x.requires_grad_()
#    gradient = torch.autograd.grad(K1(x,y), x)[0]
#    x.detach_()
#    return gradient

#Target
new_t = 1
save = True
load = False
if load:
    fname = 'exp_cos_target_high/target'
    n_t, x_t, c_t = pickle.load(open(fname, 'rb'))
if new_t == 1:
    n_t = 4
    x_t = torch.randn(n_t,d)
    x_t = torch.nn.functional.normalize(x_t, p=2, dim=1)
    c_t = torch.bernoulli(0.5*torch.ones(n_t))
    if save:
        if not os.path.exists('exp_cos_target_high'):
            os.makedirs('exp_cos_target_high')
        fname = 'exp_cos_target_high/target'
        pickle.dump((n_t, x_t, c_t), open(fname, 'wb'))

Vp = torch.zeros(Nx)
#print('Vp', Vp.shape, (c_t[0]*phi1(x,x_t[0].unsqueeze(0))/n_t).shape)
#print(x_t[0].shape)
for i in range(n_t):
    Vp = Vp + c_t[i]*phi1(x,x_t[i].unsqueeze(0)).squeeze(1)/n_t
nablaVp = torch.zeros(Nx,d)
for i in range(n_t):
    nablaVp = nablaVp + c_t[i]*dphi1(x,x_t[i].unsqueeze(0)).squeeze(1)/n_t
#print('nablaVp', nablaVp.shape)
    
up = torch.exp(-beta*Vp)
up = Nx*up/torch.sum(up)
print('Target generated')

#Data generation
nr = int(1e4) #Number of data points
nr0 = int(1e6)
x2 = torch.randn(nr0,d)
x2 = torch.nn.functional.normalize(x2, p=2, dim=1)
#print('x2 shape', x2.shape)
Vp2 = torch.zeros(nr0)
for i in range(n_t):
    #print('before')
    #a = c_t[i]*phi1(x2,x_t[i].unsqueeze(0))/n_t
    #print('update shape', (c_t[i]*phi1(x2,x_t[i].unsqueeze(0))/n_t).shape, x_t[i].unsqueeze(0).shape, torch.matmul(x2,x_t[i].unsqueeze(0).t()).shape, c_t[i].shape)
    Vp2 = Vp2 + c_t[i]*phi1(x2,x_t[i].unsqueeze(0)).squeeze(1)/n_t
#print('Vp2 shape', Vp2.shape)
#print('Vp2 content', Vp2[:10,:])
up2m = torch.max(torch.exp(-beta*Vp2))
up2n = torch.exp(-beta*Vp2)/up2m
accept = torch.bernoulli(up2n)
accepted_rows = []
for i in range(nr0):
    #print('accept', accept[i])
    if accept[i] == 1:
        accepted_rows.append(i)
accepted_tensor = torch.tensor(accepted_rows).unsqueeze(1).expand([len(accepted_rows),d])
x2 = torch.gather(x2, 0, accepted_tensor)
if x2.shape[0] > nr:
    x2 = x2[:nr,:]
#print('x2 shape', x2.shape)
print('Data generated')

nablaVpx2 = torch.zeros(nr,d)
for i in range(n_t):
    nablaVpx2 = nablaVpx2 + c_t[i]*dphi1(x2,x_t[i].unsqueeze(0)).squeeze(1)/n_t

#Initialization of the particles
npa = int(1e3)
p = torch.randint(0,nr,(npa,1))
#print('x2[p].shape', x2[p].shape)
x1 = x2[p].squeeze(1) + 0.01*torch.randn(npa, 1)
x1 = torch.nn.functional.normalize(x1, p=2, dim=1)
#print('x1 shape', x1.shape, 'x2 shape', x2.shape)
#a = torch.mean(K1(x1, x1))
#b = torch.mean(K1(x2, x2))
#c = torch.mean(K1(x1, x2))
kT1 = torch.sqrt(torch.mean(K1(x1, x1)) + torch.mean(K1(x2, x2)) - 2*torch.mean(K1(x1, x2)))
V2 = (2*torch.mean(K1(x, x1), dim = 1) - 2*torch.mean(K1(x, x2), dim = 1))/kT1
u2 = torch.exp(-beta*V2)
u2 = Nx*u2/torch.sum(u2)
print('Particles initialized')

#Initial figure
A2 = x2 - torch.mean(x2, dim=0).unsqueeze(0) #centered matrix
(U, S, V) = torch.pca_lowrank(A2, q=2, center=True, niter=2)
tf2 = torch.matmul(A2, V)
A1 = x1 - torch.mean(x1, dim=0).unsqueeze(0)
tf1 = torch.matmul(A1, V)

fig, axs = plt.subplots(2)
print('x2 shape', x2.shape, 'x1 shape', x1.shape, 'up shape', up.shape, 'u2 shape', u2.shape)
axs[0].scatter(tf2[:,0], tf2[:,1], marker='o', s=3, label='data', edgecolor='purple', facecolor='none', zorder=1)
#axs[0].hist(x2.squeeze(1), bins=60, label='histo', density='True')
#axs[0].plot(x,up, label='target PDF')
#axs[0].plot(x,u2, label='exp(-V/kT)')
#axs[0].set_ylabel('PDF')
#axs[0].set_xlabel('x')
#axs[0].set_ylim([0,8])
axs[0].legend()
axs[0].set_title(f'time = {t}, dt = {dt}')
axs[1].scatter(tf1[:,0], tf1[:,1], marker='o', s=3, label='walkers', edgecolor='purple', facecolor='none', zorder=1)
#axs[1].plot(x,-torch.log(up)/beta, label='-log target PDF')
#axs[1].plot(x,V2, label='-V/kT(u)')
#axs[1].set_ylabel('potential')
#axs[1].set_xlabel('x')
#axs[1].set_ylim([-.4,.6])
axs[1].legend()
fig.tight_layout(pad=1.0)
if not os.path.exists('plots_exp_cos_high'):
    os.makedirs('plots_exp_cos_high')
fig.savefig(f'plots_exp_cos_high/initial_plot.pdf', bbox_inches='tight', pad_inches=0)
plt.close('all')

#Function for plots
def plot_figure():
    A1 = x1 - torch.mean(x1, dim=0).unsqueeze(0)
    tf1 = torch.matmul(A1, V)
    fig, axs = plt.subplots(2)
    axs[0].scatter(tf2[:,0], tf2[:,1], marker='o', s=3, label='data', edgecolor='purple', facecolor='none', zorder=1)
    axs[0].legend()
    axs[0].set_title(f'time = {t}, dt = {dt}')
    axs[1].scatter(tf1[:,0], tf1[:,1], marker='o', s=3, label='walkers', edgecolor='purple', facecolor='none', zorder=1)
    axs[1].legend()
    fig.tight_layout(pad=1.0)
    fig.savefig(f'plots_exp_cos_high/time_{round(t, 5)}.pdf', bbox_inches='tight', pad_inches=0)
    plt.close('all')

#Main loop
i = 0
n_plot = 1e1
V2avg = 0
nablaV2x2avg = torch.zeros(nr,d)
kT1avg = 0
n_save = int(1e3)
x1s = torch.zeros(npa, d, n_save)
#print('x1 shape', x1.shape)
x1s[:,:,0] = x1
n_batch = int(1e3)
start = time.time()
while t < tmax:
    i = i + 1
    p = torch.randint(0,nr,(n_batch,1))
    x2b = x2[p].squeeze(1)
    kT1 = torch.sqrt(torch.mean(K1(x1, x1)) + torch.mean(K1(x2b,x2b)) - 2*torch.mean(K1(x1, x2b)))
    kT1avg = (i-1)/i*kT1avg + kT1/i
    V2 = (2*torch.mean(K1(x, x1), dim = 1) - 2*torch.mean(K1(x, x2b), dim = 1))/kT1
    V2avg = (i-1)/i*V2avg + V2/i
    #nablaV2x2 = (2*torch.mean(dK1(x2, x1), dim = 1) - 2*torch.mean(dK1(x2, x2b), dim = 1))/kT1
    #nablaV2x2avg = (i-1)/i*nablaV2x2avg + nablaV2x2/i
    t = t + dt
    x1 = x1 - 2*dt*(torch.mean(dK1(x1, x1), dim = 1) - torch.mean(dK1(x1, x2b), dim = 1)) + np.sqrt(2*dt*kT1/beta)*torch.randn(npa,1)
    x1 = torch.nn.functional.normalize(x1, p=2, dim=1)
    #print('x1 shape', x1.shape)
    if i < n_save-1:
        x1s[:,:,i+1] = x1
    if i%10 == 0:
        print(f'Gradient norm average: {torch.mean(torch.abs(2*dt*(torch.mean(dK1(x1, x1), dim = 1) - torch.mean(dK1(x1, x2b), dim = 1))))}. Noise standard deviation: {np.sqrt(2*dt*kT1/beta)}.')
        print(f'Time {t}/{tmax} elapsed. Real time elapsed: {time.time() - start}.')
        #fisher_information = torch.mean(torch.norm(nablaVpx2-nablaV2x2avg, dim=1)**2)
        #print(f'Relative Fisher information: {fisher_information}.')
        u2avg = torch.exp(-beta*V2avg)
        u2avg = Nx*u2avg/torch.sum(u2avg)
        l2_energy_diff = torch.mean((V2avg-torch.mean(V2avg)-Vp+torch.mean(Vp))**2*torch.exp(-beta*V2avg))/torch.mean(torch.exp(-beta*V2avg))
        print(f'Centered energy L2 difference: {l2_energy_diff}.')
        l1_density_diff = torch.mean(torch.abs(u2avg-up))
        print(f'Density L1 difference: {l1_density_diff}.')
        plot_figure()