from pykeops.torch import Kernel, kernel_product, Genred
from pykeops.torch.kernel_product.formula import *


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


def gauss_kernel(sigma, dtype=torch.float32):
    p = torch.tensor([1/sigma/sigma])
    def K(x, y, b):
        d = 2
        formula = 'Exp(-p*SqDist(x, y))*b'
        variables = ['x = Vx('+str(d)+')',
                     'y = Vy('+str(d)+')',
                     'b = Vy('+str(d)+')',
                     'p = Pm(1)']

        cuda_type = "float32"
        if(dtype is torch.float64):
            cuda_type = "float64"

        my_routine = Genred(formula, variables, reduction_op='Sum', axis=1, cuda_type=cuda_type)
        return my_routine(x, y, b, p, backend="auto")
    return K

