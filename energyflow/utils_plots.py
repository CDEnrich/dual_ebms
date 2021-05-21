import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_2d_level_f(function, x_min=-10, x_max=10,
                    y_min=None, y_max=None,
                    n_points=100, ax=None, title='',
                    device='cpu'):
    """
    Args:
    function take 2d input
    """
    # if 
    dim = 2
    x_range = torch.linspace(x_min, x_max, n_points, device=device)
    if y_min is None:
        y_range = x_range.clone()
    else:
        y_range = torch.linspace(y_min, y_max, n_points, device=device)

    grid = torch.meshgrid(x_range, y_range)
    xys = torch.stack(grid).reshape(2, n_points ** 2).T.to(device)
    
    ## actually can also draw a cut in higher dimension
    if dim > 2:
        blu = torch.zeros(n_points ** 2, dim).to(device)
        blu[:, 0:2] = xys
        xys = blu

    Us = function(xys).reshape(n_points, n_points).T.detach().cpu().numpy()
    
    if ax is None:
        plt.figure()
    else:
        plt.sca(ax)
    plt.imshow(np.exp(- Us[::-1]), 
    # plt.imshow(Us[::-1]
        # norm=matplotlib.colors.LogNorm()
        )
    plt.axis('off')
    plt.colorbar()
    plt.title(title)
    return Us