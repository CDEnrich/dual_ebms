import numpy as np
import scipy.special as ss
import torch
import torch.nn as nn

class RBF(nn.Module):
    def __init__(self, sigma=1):
        self.sigma = sigma

    def forward(self, x, y):
        """
        inputs have shape (N x d) et (N' x d) 
        N = # samples
        d = dim

        output shape (N, N')
        """
        gram_diff_sum = torch.einsum('ki,li->kl', x, -y)
        out = torch.exp(- 0.5 * (gram_diff_sum ** 2) / self.sigma) 
        return out / np.sqrt(2 * np.pi * self.sigma) ** x.shape[1]

    def grad(self, x, y):
        """
        return dk/dx first argument

        output shape (N, N', d)
        """ 
        gram_diff = x.unsqueeze(1) - y.unsqueeze(0)
        return self.forward(x, y).unsqueeze(2) * 0.5 * gram_diff / self.sigma

class Bessel(nn.Module):
    """
    TODO: test
    """
    def __init__(self, a=10):
        self.a = a

    def forward(self, x, y):
        """
        takes inputs of size N x d
        N = # samples
        d = dim
        """
        z = 2 * self.a * torch.sqrt((1 + x @ y.t()) / 2)
        return ss.iv(0, z) * np.exp(- 2 * self.a)

    def grad(self, x, y):
        z = 2 * self.a * torch.sqrt((1 + x @ y.t()) / 2)
        g = ss.iv(1, z).unsqueeze(2) * self.a * np.exp(- 2 * self.a) * y.unsqueeze(0) 
        g = g / (np.sqrt(2) * torch.sqrt(1 + x @ y.t()).unsqueeze(2))
        return g


class Bessel_1d(nn.Module):
    def __init__(self, a=10):
        self.a = a

    def forward(self, x, y):
        nx = x.shape[0]
        ny = y.shape[0]

        x = x.expand(nx, ny)
        y = y.t().expand(nx, ny)

        x = x - y # or y - x ??
        a = self.a
        return ss.iv(0, 2 * a * torch.cos(np.pi*x)) * np.exp(-2*a)

    def grad(self, x, y):
        nx = x.shape[0]
        ny = y.shape[0]

        x = x.expand(nx, ny)
        y = y.t().expand(nx, ny)

        x = x - y  # or y - x ??
        a = self.a 
        return -ss.iv(1, 2* a* torch.cos(np.pi*x))*2*a*np.pi*torch.sin(np.pi*x)*np.exp(-2*a)