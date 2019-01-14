import torch
import numpy as np


def scal( α, f ) :
    "Scalar product between two vectors."
    return torch.dot( α.view(-1), f.view(-1) )


def sqdistances(x_i, y_j) :
    "Matrix of squared distances, C_ij = |x_i-y_j|^2."
    return ( (x_i.unsqueeze(1) - y_j.unsqueeze(0)) ** 2).sum(2)


def K_xx(x_i, σ = .1) :
    return (-sqdistances(x_i,x_i)/σ**2 ).exp()


def K_xy(x_i, y_i, σ = .1) :
    return (-sqdistances(x_i,y_i)/σ**2 ).exp()


def grid2vec(x, y):
    return torch.cat([x.contiguous().view(1, -1), y.contiguous().view(1, -1)], 0).t()


def vec2grid(vec, nx, ny):
    return vec.t()[0].view(nx, ny), vec.t()[1].view(nx, ny)
