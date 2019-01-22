import torch
import numpy as np
import matplotlib.pyplot as plt

def scal(u, v):
    """Scalar product between two vectors."""
    return torch.dot(u.view(-1), v.view(-1))

def distances(x, y):
    """Matrix of distances, C_ij = |x_i - y_j|."""
    return (x.unsqueeze(1) - y.unsqueeze(0)).norm(p=2, dim=2)

def sqdistances(x, y):
    """Matrix of squared distances, C_ij = |x_i-y_j|^2."""
    return ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(2)


def K_xx(x, σ = .1):
    """Kernel matrix for x."""
    return (-sqdistances(x, x)/σ**2 ).exp()


def K_xy(x, y, σ = .1):
    """Kernel matrix between x and y."""
    return (-sqdistances(x, y)/σ**2 ).exp()


def grid2vec(x, y):
    """Convert a grid of points (such as given by torch.meshgrid) to a tensor of vectors."""
    return torch.cat([x.contiguous().view(1, -1), y.contiguous().view(1, -1)], 0).t()


def vec2grid(vec, nx, ny):
    """Convert a tensor of vectors to a grid of points."""
    return vec.t()[0].view(nx, ny), vec.t()[1].view(nx, ny)


def plotTensorScatter(x, alpha=1., scale=64.):
    "Scatter plot points in the format: ([x, y], scale) or ([x, y]) (in that case you can specify scale)"""
    if(isinstance(x, tuple)):
        #plt.scatter(x[0].detach().numpy()[:,1], x[0].detach().numpy()[:,0], s=50.*x[1].shape[0]*x[1].detach().numpy(), marker='o', alpha=alpha)
        plt.scatter(x[0].detach().numpy()[:,1], x[0].detach().numpy()[:,0], s=scale, marker='o', alpha=alpha)
    else:
        plt.scatter(x.detach().numpy()[:,1], x.detach().numpy()[:,0], s=scale, marker='o', alpha=alpha)
