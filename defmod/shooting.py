import torch
import numpy as np
from torch.autograd import grad
from .hamiltonian import Hamiltonian
from torchdiffeq import odeint_adjoint as odeint


def shoot(h, it=10):
    step = 1. / it

    intermediate = [h.module.manifold.copy()]
    for i in range(it):
        h.geodesic_controls()
        d_gd, d_mom = grad(h(), [h.module.manifold.gd, h.module.manifold.cotan], create_graph=True)
        h.module.manifold.muladd_gd(d_mom, step)
        h.module.manifold.muladd_cotan(-d_gd, step)
        intermediate.append(h.module.manifold.copy())

    return intermediate


# def shoot(h, it=10, method='rk4'):
#     # Wrapper class used by TorchDiffEqb
#     # TODO: __call__ of hamiltonian should give the hamiltonian value and not his grad wrt gd, mom.
#     class TorchDiffEqHamiltonian(Hamiltonian):
#         def __init__(self, def_module):
#             super().__init__(def_module)

#         def __call__(self, t, x):
#             with torch.enable_grad():
#                 self.module.manifold.fill_gd(x[0].detach().requires_grad_())
#                 self.module.manifold.fill_cotan(x[1].detach().requires_grad_())
#                 self.geodesic_controls()
#                 delta = grad(super().__call__(),
#                              [self.module.manifold.gd, self.module.manifold.cotan], create_graph=True)
#                 return torch.cat([delta[1], -delta[0]], dim=0).view(2, -1)

#     intermediate = [h.module.manifold.copy()]

#     x = torch.cat([h.module.manifold.gd.view(1, -1), h.module.manifold.cotan.view(1, -1)], dim=0)
#     c = TorchDiffEqHamiltonian.from_hamiltonian(h)
#     result = odeint(c, x, torch.linspace(0., 1., it), method=method)

#     h.module.manifold.fill_gd(result[-1, 0, :])
#     h.module.manifold.fill_cotan(result[-1, 1, :])

#     # TODO: very very dirty, change this
#     for i in range(0, it):
#         intermediate.append(intermediate[-1].copy())
#         intermediate[-1].fill_gd(result[i, 0, :])
#         intermediate[-1].fill_cotan(result[i, 1, :])

#     return intermediate

