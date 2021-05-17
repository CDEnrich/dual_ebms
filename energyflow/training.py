from energyflow.kernels import Bessel, Bessel_1d
import torch
import numpy as np



def compute_mmd(x, y, knl):
    """
    x : N x d1 x ...
    y : N x d1 x ...
    knl : kernel object must handle correctly the dimensions
    """
    grams = knl.forward(x, x).mean() + knl.forward(y, y).mean() 
    grams += - 2 * knl.forward(x, y).mean()
    grams = torch.maximum(torch.tensor(0), grams)
    kT = torch.sqrt(grams)
    return kT


def train(xs, xt, kernel, niter, n_batch=int(1e1), 
          beta=10, dt=1e-1, sphere=None, xval=None):
    """
    xs for x_students, particles beging learned
    xt for x_target, samples from target distribution 
    sphere: project back to the sphere if 'project', remainder if 'pbc'
    """

    Nt = xt.shape[0]  # nb of training samples, dim
    Ns = xs.shape[0]  # nb of particles beging learned
    dims = xt.shape[1:]
    K1 = kernel.forward
    dK1 = kernel.grad
    lbd = 1 # coef of base Gaussian when not on sphere

    xss = []
    
    kT1avgs = []
    kT1avg = 0

    for i in range(1, niter+1):
        batch_indxs = torch.randint(Nt, (n_batch,))
        xtb = xt[batch_indxs, :]

        blu1 = dK1(xs, xtb)
        blu2 = dK1(xs, xs)

        kT1 = compute_mmd(xs, xtb, kernel)
        kT1avg = (i-1) / i * kT1avg + kT1 / i

        diff = dK1(xs, xs).mean(1) - dK1(xs, xtb).mean(1)

        if isinstance(kernel, Bessel_1d):
            ## should have been more careful about implementing 
            diff.unsqueeze_(1)

        xs = xs -  dt * diff
        
        xs += np.sqrt(2 * dt * kT1 / beta) * torch.randn(xs.shape)

        ## if we are working on the sphere
        if sphere == 'pbc':
            xs = torch.remainder(xs, 1) 
        elif sphere == 'project':
            # TO DO for array data
            xs = torch.nn.functional.normalize(xs, p=2, dim=1) 
        else:
            xs -= lbd * dt * kT1 / beta * xs ** 2 
            
        if i % (niter / 10) == 0:
            kT1avgs.append(kT1avg)
            xss.append(xs)

            mean_grad_norm = torch.mean(torch.abs(2 * dt * diff ))
            noise_std = np.sqrt(2*dt*kT1/beta)

            print('Iter: {:d}, Gradient norm average: {:0.2e}, Noise standard deviation: {:0.2e}'.format(i, mean_grad_norm, noise_std))
            
    to_return = {
        'xss': xss,
        'kT1avgs': kT1avgs
    }
    
    return to_return
