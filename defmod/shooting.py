import torch
import numpy as np
from torch.autograd import grad
#from torchdiffeq import odeint_adjoint as odeint


def shoot(gd, mom, h, n):
    step = 1. / n
    for i in range(n):
        [d_gd, d_mom] = grad(h(gd, mom, h.geodesic_controls(gd, mom)), [gd, mom], create_graph=True)
        gd = gd + step*d_mom
        mom = mom - step*d_gd
    return gd, mom


# def shootTorchdiffeq(GD, MOM, H):
#     x = torch.cat([GD.view(1, -1), MOM.view(1, -1)])
#     result = odeint(H, x, torch.linspace(0., 1., 100))
#     return result[-1, 0, :], result[-1, 1, :]


# class deqH(Hamiltonian.Hamilt):
#     def __init__(self, DefModule):
#         super().__init__(DefModule)
        
#     def __call__(self, t, x):
#         with torch.enable_grad():
#             Cont = self.Cont_geo(x[0], x[1])
#             a = super().__call__(x[0], x[1], Cont), [x[0], x[1]]
#         print(a)
#         return x

