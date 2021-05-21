from energyflow.kernels import Bessel, Bessel_1d
import torch
import numpy as np


def train(xs, xt, modelU, niter, n_batch=int(1e1), 
          beta=10, dt=1e-1, coef_speed=1, sphere=None, xval=None):
    """
    xs for x_students, particles beging learned
    xt for x_target, samples from target distribution 
    modelU: nn.Module with parameters taking xs as inputs
    sphere: project back to the sphere if 'project', remainder if 'pbc'
    """

    Nt = xt.shape[0]  # nb of training samples, dim
    Ns = xs.shape[0]  # nb of particles beging learned
    dims = xt.shape[1:]
    lbd = 0.1 # coef of base Gaussian when not on sphere

    xss = [xs.clone()]
    sumUts = []
    modelUs = []

    xs.requires_grad_()

    for i in range(1, niter+1):
        batch_indxs = torch.randint(Nt, (n_batch,))
        xtb = xt[batch_indxs, :]

        ### UPDATE CHAINS
        if xs.grad is not None:
            xs.grad.zero_()
        Ux = modelU(xs).sum()
        Ux.backward()
        xs.data.sub_(dt * xs.grad 
                    + np.sqrt(2 * dt * beta ** -1) * torch.randn(xs.shape))

        ## if we are working on the sphere
        with torch.no_grad():
            if sphere == 'pbc':
                xs = torch.remainder(xs, 1) 
            elif sphere == 'project':
                # TO DO for array data
                xs = torch.nn.functional.normalize(xs, p=2, dim=1) 
            else:
                # some other form of regularization?
                # xs -= 2 * lbd * dt * kT1 / beta * xs 
                pass

        ### UPDATE PARAMETERS
        blu = modelU(xs).shape
        modelU.zero_grad()
        Utheta = (modelU(xs).mean() - modelU(xtb).mean()) 
        Utheta.backward()
        for param in modelU.parameters():
            param.data.sub_(coef_speed * param.grad)

            
        if i % (niter / 10) == 0 or niter < 10:
            xss.append(xs.clone())
            sumUts.append(modelU(xt).sum().item())
            modelUs.append(modelU)
            

            print('Iter: {:d}, loglikelihood: {:0.2e}, grads: {:0.2e}'.format(i, Utheta, xs.grad.norm()))
            
    to_return = {
        'xss': xss,
        'sumUts': sumUts,
        'modelsU': modelUs
    }
    
    return to_return
