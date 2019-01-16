import torch
import numpy as np


def scal(u, v) :
    """Scalar product between two vectors."""
    return torch.dot( u.view(-1), v.view(-1) )


def sqdistances(x, y) :
    """Matrix of squared distances, C_ij = |x_i-y_j|^2."""
    return ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(2)


def K_xx(x, σ = .1) :
    """Kernel matrix for x."""
    return (-sqdistances(x, x)/σ**2 ).exp()


def K_xy(x, y, σ = .1) :
    """Kernel matrix between x and y."""
    return (-sqdistances(x, y)/σ**2 ).exp()


def grid2vec(x, y):
    """Convert a grid of points (such as given by torch.meshgrid) to a tensor of vectors."""
    return torch.cat([x.contiguous().view(1, -1), y.contiguous().view(1, -1)], 0).t()


def vec2grid(vec, nx, ny):
    """Convert a tensor of vectors to a grid of points."""
    return vec.t()[0].view(nx, ny), vec.t()[1].view(nx, ny)
