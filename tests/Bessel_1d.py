import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy.special as ss
from energyflow.training import train
from energyflow.kernels import Bessel_1d

d = 1
# t = 0
# tmax = 2e1
dt = 1e-1
beta = 10

Nx = 2 ** 4
x = torch.linspace(0,1, steps=Nx+1)
x = x[:Nx]


nr = int(1e1) #Number of data points
nr0 = int(1e3)
npa = Nx


#Kernel
a = 10
K1 = lambda x: ss.iv(0,2*a*torch.cos(np.pi*x))*np.exp(-2*a)
dK1 = lambda x: -ss.iv(1,2*a*torch.cos(np.pi*x))*2*a*np.pi*torch.sin(np.pi*x)*np.exp(-2*a)
phi1 = lambda x: torch.exp(a*torch.cos(2*np.pi*x)-a)

#Target
new_t = 1
if new_t == 1:
    n_t = 4
    x_t = torch.rand(n_t,d)
    c_t = torch.sign(x_t-0.5)

Vp = torch.zeros_like(x)
for i in range(n_t):
    Vp = Vp + c_t[i]*phi1(x-x_t[i])/n_t
    
up = torch.exp(-beta*Vp)
up = Nx*up/torch.sum(up)
print('Target generated')

#Data generation

x2 = torch.rand(nr0, d)
Vp2 = torch.zeros_like(x2)
for i in range(n_t):
    Vp2 = Vp2 + c_t[i]*phi1(x2-x_t[i])/n_t
up2m = torch.max(torch.exp(-beta*Vp2))
up2n = torch.exp(-beta*Vp2)/up2m
accept = torch.bernoulli(up2n)
accepted_rows = []
for i in range(nr0):
    if accept[i] == 1:
        accepted_rows.append(i)
accepted_tensor = torch.tensor(accepted_rows).unsqueeze(1).expand([len(accepted_rows),d])
x2 = torch.gather(x2, 0, accepted_tensor)
if x2.shape[0] > nr:
    x2 = x2[:nr,:]
print('Data generated')

#Initialization of the particles
p = torch.randint(0,nr,(npa,1))
x1 = torch.remainder(x2[p].squeeze(2) + 0.01*torch.randn(npa, 1), 1)
kT1 = torch.sqrt(torch.mean(K1(x1.expand(npa,npa) - x1.t().expand(npa,npa))) + torch.mean(K1(x2.expand(nr,nr) - x2.t().expand(nr,nr))) - 2*torch.mean(K1(x1.expand(npa,nr) - x2.t().expand(npa,nr))))
V2 = (2*torch.mean(K1(x.unsqueeze(1).expand(Nx, npa) - x1.t().expand(Nx, npa)), dim = 1) - 2*torch.mean(K1(x.unsqueeze(1).expand(Nx, nr) - x2.t().expand(Nx, nr)), dim = 1))/kT1

u2 = torch.exp(-beta*V2)
u2 = Nx*u2/torch.sum(u2)
print('Particles initialized')


##########################################################################

kernel = Bessel_1d(a=10)
niter = 10
xs = x1
xt = x2
xval = x.reshape(Nx,1)

_ = train(xs, xt, kernel, niter, dt=dt, beta=beta, 
                            sphere=True, xval=xval)

xss, V2avgs, u2avgs = (_[key] for key in ['xss', 'V2avgs', 'u2avgs'])

##########################################################################

 #Function for plots
def plot_figure(x2, x, up, u2avg, V2avg, beta, t=0):
    fig, axs = plt.subplots(2)
    #print('x2 shape', x2.shape, 'x1 shape', x1.shape, 'up shape', up.shape, 'u2 shape', u2.shape)
    axs[0].scatter(x2,0*x2, marker='o', s=2, label='data', edgecolor='purple', 
                  facecolor='none', zorder=1)
    axs[0].hist(x2.squeeze(1), bins=60, label='histo', density='True')
    axs[0].plot(x,up, label='target PDF')
    axs[0].plot(x,u2avg, label='exp(-V/kT)')
    axs[0].set_ylabel('PDF')
    axs[0].set_xlabel('x')
    axs[0].set_ylim([0,8])
    axs[0].legend()
    axs[0].set_title(f'time = {round(t, 5)}, dt = {dt}')
    axs[1].scatter(x1,0*x1, marker='o', s=2, label='walkers', edgecolor='purple', facecolor='none', zorder=1)
    axs[1].plot(x,-torch.log(up)/beta, label='-log target PDF')
    axs[1].plot(x,V2avg, label='-V/kT(u)')
    axs[1].set_ylabel('potential')
    axs[1].set_xlabel('x')
    axs[1].set_ylim([-.6,.6])
    axs[1].legend()
    fig.tight_layout(pad=1.0)
    # fig.savefig(f'plots_exp_cos/time_{round(t, 5)}.pdf', bbox_inches='tight', pad_inches=0)
    # plt.close('all')


# for split in range(len(V2avgs)):
split = -1
u2avg, V2avg = u2avgs[split], V2avgs[split]
plot_figure(x2, x, up, u2avg, V2avg, beta)