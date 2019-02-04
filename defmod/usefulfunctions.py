import torch
import numpy as np
import matplotlib.pyplot as plt


class AABB:
    """Class used to represent an Axis Aligned Bounding Box"""
    def __init__(self, xmin=0., xmax=0., ymin=0., ymax=0.):
        self.__xmin = xmin
        self.__xmax = xmax
        self.__ymin = ymin
        self.__ymax = ymax

    @staticmethod
    def build_from_points(points):
        """Compute the AABB from points"""

        return AABB(torch.min(points[:, 0]), torch.max(points[:, 0]),
                    torch.min(points[:, 1]), torch.max(points[:, 1]))

    def __getitem__(self, key):
        return self.get_list()[key]

    def get_list(self):
        """Returns the AABB as a list, 0:xmin, 1:xmax, 2:ymin, 3:ymax."""
        return [self.__xmin, self.__xmax, self.__ymin, self.__ymax]

    @property
    def xmin(self):
        return self.__xmin

    @property
    def ymin(self):
        return self.__ymin

    @property
    def xmax(self):
        return self.__xmax

    @property
    def ymax(self):
        return self.__ymax

    @property
    def width(self):
        return self.__xmax - self.__xmin

    @property
    def height(self):
        return self.__ymax - self.__ymin


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


def grid2vec(x, y):
    """Convert a grid of points (such as given by torch.meshgrid) to a tensor of vectors."""
    return torch.cat([x.contiguous().view(1, -1), y.contiguous().view(1, -1)], 0).t()


def vec2grid(vec, nx, ny):
    """Convert a tensor of vectors to a grid of points."""
    return vec.t()[0].view(nx, ny), vec.t()[1].view(nx, ny)


def plotTensorScatter(x, alpha=1., scale=64.):
    """Scatter plot points in the format: ([x, y], scale) or ([x, y]) (in that case you can specify scale)"""
    if(isinstance(x, tuple) or isinstance(x, list)):
        #plt.scatter(x[0].detach().numpy()[:,1], x[0].detach().numpy()[:,0], s=50.*x[1].shape[0]*x[1].detach().numpy(), marker='o', alpha=alpha)
        plt.scatter(x[0].detach().numpy()[:,1], x[0].detach().numpy()[:,0], s=64.*x[1].shape[0]*x[1], marker='o', alpha=alpha)
    else:
        plt.scatter(x.detach().numpy()[:,1], x.detach().numpy()[:,0], s=scale, marker='o', alpha=alpha)
