import argparse
import numpy as np 
import torch
from energyflow.models import MLP
from energyflow.training_persistent import train
from energyflow.gaussian_utils import MoG, plot_2d_level
from energyflow.utils_plots import plot_2d_level_f
import matplotlib.pyplot as plt 


parser = argparse.ArgumentParser(description='Prepare experiment')
parser.add_argument('-dim', '--dim', type=int, default=2)
parser.add_argument('-k', '--k-modes', type=int, default=2) 
parser.add_argument('-cov', '--covariances', type=float, nargs='+', default=[1.])
parser.add_argument('-w', '--weights', type=float, nargs='+', default=[1.])
parser.add_argument('-sprd', '--spread-from-o', type=float, default=4)
parser.add_argument('-shft', '--shift', type=float, default=0)

parser.add_argument('-nt', '--n-target', type=int, default=int(1e3))
parser.add_argument('-nw', '--n-walkers', type=int, default=int(1e3))

# energy function parameters
parser.add_argument('-dp', '--depth', type=int, default=1)
parser.add_argument('-hd', '--hidden-width', type=int, default=100)

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dtype = torch.float32

####### CREATE TARGET ###############
dim = args.dim
k = args.k_modes
thetas = np.linspace(0, 2 * np.pi, k + 1)[:-1]
means_2d = [args.spread_from_o * torch.tensor([np.cos(t),np.sin(t)], dtype=dtype, device=device) + args.shift for t in thetas]

if dim > 2:
    means = [torch.zeros(dim, dtype=dtype, device=device) for m in range(k)]
    for m in range(k):
        means[m][0:2] = means_2d[m]
else:
    means = means_2d

if len(args.covariances) == 1:
    covars = [args.covariances[0] * torch.eye(dim, device=device, dtype=dtype)] * k
else:
    covars = [c * torch.eye(dim, device=device, dtype=dtype) for c in args.covariances] 


if len(args.weights) == 1:
    weights = args.weights * k
elif type(args.weights) == list:
    weights = args.weights
else:
    raise NotImplemented

mog = MoG(means, covars, weights=weights, dtype=dtype, device=device)
x_min = -10
x_max = 10
Us_g = plot_2d_level(mog, x_min=x_min, x_max=x_max, n_points=100)  # d>2 plots slice 5

xt = mog.sample(args.n_target)

#######################
p = torch.randint(0, args.n_target, (args.n_walkers,), device=device)
xs = xt[p, :] #+ 0.1 * torch.randn(args.n_walkers, dim)

hdim = args.hidden_width
modelU = MLP([dim] + [hdim] * args.depth + [1], init_scale=1e-3, activation=torch.tanh)
niter = 100
dt = 5e-3
beta = 1.

out = train(xs, xt, modelU, niter, dt=dt, beta=beta, n_batch=int(1e3), 
            coef_speed=1, sphere=False, xval=None)

xss, sumUts = (out[key] for key in ['xss', 'sumUts'])

#######################

xss = [xs.detach() for xs in xss]
xs = xs.detach()

plt.figure(figsize=(15,7))
# plt.scatter(xss[-1][:,0], xss[-1][:,1])  
axs = [plt.subplot(1,2,i+1) for i in range(2)]

plt.sca(axs[0])
cat_xss = torch.stack(xss)
for i in range(cat_xss.shape[1]):
    plt.plot(cat_xss[:, i, 0], cat_xss[:, i, 1], '-', c='gray', alpha=0.1)
    plt.xlim(x_min, x_max)
    plt.ylim(x_min, x_max)

plt.scatter(xs[:, 0], xs[:, 1], marker='o', edgecolor='red', facecolor='None')

plt.sca(axs[1])
plt.plot(sumUts)
# plt.yscale('log')

blu= plot_2d_level_f(modelU, x_min=x_min, x_max=x_max, n_points=100)