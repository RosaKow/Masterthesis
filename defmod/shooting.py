import torch
import numpy as np
from torch.autograd import grad
from .hamiltonian import Hamiltonian
from torchdiffeq import odeint_adjoint as odeint

# def shoot(gd, mom, h, n=10):
#     # TODO: Assertion on the dimension of gd and mom (should be 1D)
#     step = 1. / n
#     gd_out = gd.clone()
#     mom_out = mom.clone()
#     for i in range(n):
#         controls = h.geodesic_controls(gd_out, mom_out)
#         [d_gd, d_mom] = grad(h(gd_out, mom_out, controls), [gd_out, mom_out], create_graph=True)
#         gd_out = gd_out + step*d_mom
#         mom_out = mom_out - step*d_gd
#     return gd_out, mom_out


def shoot(gd, mom, h, it=2, reverse=False, intermediate=False, output_list=False):
    assert len(gd.shape) == 1
    assert len(mom.shape) == 1
    
    # Wrapper class used by TorchDiffEqb
    # TODO: __call__ of hamiltonian should give the hamiltonian value and not his grad wrt gd, mom.
    class TorchDiffEqHamiltonian(Hamiltonian):
        def __init__(self, def_module):
            super().__init__(def_module)

        def __call__(self, t, x):
            with torch.enable_grad():
                gd = x[0]
                mom = x[1]

                gd.requires_grad_()
                mom.requires_grad_()
                g = grad(super().__call__(gd, mom, self.geodesic_controls(gd, mom)),
                         [gd, mom], create_graph=True)

                return torch.cat([g[1], -g[0]], dim=0).view(2, -1)

    t0, t1 = 0., 0.
    if(not reverse):
        t1 = 1.
    else:
        t0 = 1.

    x = torch.cat([gd.view(1, -1), mom.view(1, -1)], dim=0)
    result = odeint(TorchDiffEqHamiltonian.from_hamiltonian(h),
                            x, torch.linspace(t0, t1, it), method='rk4')
    if(intermediate):
        result_gd_list = []
        result_mom_list = []
        for i in range(0, it):
            result_gd_list.append(result[i, 0, :])
            result_mom_list.append(result[i, 1, :])

        return result_gd_list, result_mom_list
    else:
        if(output_list):
            return [result[-1, 0, :]], [result[-1, 1, :]]
        else:
            return result[-1, 0, :], result[-1, 1, :]


