import torch
import numpy as np
from torch.autograd import grad
from .hamiltonian import Hamiltonian
from torchdiffeq import odeint_adjoint


def shoot_euler(h, it=10):
    step = 1. / it

    intermediate = [h.module.manifold.copy()]
    for i in range(it):
        h.geodesic_controls()
        l = [*h.module.manifold.gd, *h.module.manifold.cotan]
        grad_out = grad(h(), l, create_graph=True)
        d_gd = grad_out[:int(len(grad_out)/2)]
        d_mom = grad_out[int(len(grad_out)/2):]
        h.module.manifold.muladd_gd(d_mom, step)
        h.module.manifold.muladd_cotan(d_gd, -step)
        intermediate.append(h.module.manifold.copy())

    return intermediate


def shoot(h, it=2, method='rk4'):
    # Wrapper class used by TorchDiffEqb
    # TODO: __call__ of hamiltonian should give the hamiltonian value and not his grad wrt gd, mom.
    class TorchDiffEqHamiltonian(Hamiltonian, torch.nn.Module):
        def __init__(self, def_module):
            super().__init__(def_module)

        def __call__(self, t, x):
            with torch.enable_grad():
                gd, mom = [], []
                index = 0
                for m in self.module:
                    gd.append(x[0][index:index+m.manifold.dim_gd].requires_grad_())
                    mom.append(x[1][index:index+m.manifold.dim_gd].requires_grad_())
                    index = index + m.manifold.dim_gd
                self.module.manifold.fill_gd(gd)
                self.module.manifold.fill_cotan(mom)
                self.geodesic_controls()
                delta = grad(super().__call__(),
                             [*self.module.manifold.gd, *self.module.manifold.cotan], create_graph=True)
                gd_out = delta[:self.module.nb_module]
                mom_out = delta[self.module.nb_module:]
                return torch.cat(list(map(lambda x: x.view(-1), [*mom_out, *list(map(lambda x: -x, gd_out))])), dim=0).view(2, -1)

    intermediate = [h.module.manifold.copy()]

    x_0 = torch.cat(list(map(lambda x: x.view(-1), [*h.module.manifold.gd, *h.module.manifold.cotan])), dim=0).view(2, -1)
    x_1 = odeint_adjoint(TorchDiffEqHamiltonian.from_hamiltonian(h), x_0, torch.linspace(0., 1., it), method=method)

    gd, mom = [], []
    index = 0
    for m in h.module:
        gd.append(x_1[-1, 0, index:index+m.manifold.dim_gd])
        mom.append(x_1[-1, 1, index:index+m.manifold.dim_gd])
        index = index + m.manifold.dim_gd

    h.module.manifold.fill_gd(gd)
    h.module.manifold.fill_cotan(mom)

    # TODO: very very dirty, change this
    for i in range(0, it):
        gd, mom = [], []
        index = 0
        for m in h.module:
            gd.append(x_1[-1, 0, index:index+m.manifold.dim_gd])
            mom.append(x_1[-1, 1, index:index+m.manifold.dim_gd])
            index = index + m.manifold.dim_gd

        intermediate.append(intermediate[-1].copy())
        intermediate[-1].fill_gd(gd)
        intermediate[-1].fill_cotan(mom)

    return intermediate

