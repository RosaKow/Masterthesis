import torch

def scal(x, y):
    """Scalar product between two vectors."""
    return torch.dot(x.view(-1), y.view(-1))


def distances(x, y):
    """Matrix of distances, C_ij = |x_i - y_j|."""
    return (x.unsqueeze(1) - y.unsqueeze(0)).norm(p=2, dim=2)


def sqdistances(x, y):
    """Matrix of squared distances, C_ij = |x_i-y_j|^2."""
    return ((x.unsqueeze(1) - y.unsqueeze(0))**2).sum(2)


def K_xx(x, sigma = .1):
    """Kernel matrix for x."""
    return (-sqdistances(x, x)/sigma**2).exp()


def K_xy(x, y, sigma = .1):
    """Kernel matrix between x and y."""
    return (-sqdistances(x, y)/sigma**2).exp()

