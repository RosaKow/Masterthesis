import torch
import numpy as np
from torch.autograd import grad
from .hamiltonian import Hamiltonian
from torchdiffeq import odeint as odeint
from .usefulfunctions import make_grad_graph
from .deformationmodules import SilentPoints
from .manifold import Landmarks
from .multishape_silentpoints import MultiShapeModule_silent
import copy

import defmod as dm


def shoot(h, it, method):
    if method == "torch_euler":
        return shoot_euler(h, it)
    else:
        return shoot_torchdiffeq(h, it, method)
    

def shoot_euler(h, it):
    step = 1. / it

    intermediate_states = [h.module.manifold.copy()]
    intermediate_controls = []
        
    for i in range(it):
        h.geodesic_controls()
        #print('constr',h.constraints(h.module))
        #mod = h.module.copy()
        
        #speed_action = [gdi.action(modulei).tan for gdi, modulei in zip(h.module.manifold.manifold_list, h.module)] 
        
        l = [*h.module.manifold.unroll_gd(), *h.module.manifold.unroll_cotan()]
        delta = grad(h(), l, create_graph=True, allow_unused=True)
        
        d_gd = h.module.manifold.roll_gd(list(delta[:int(len(delta)/2)]))
        d_mom = h.module.manifold.roll_cotan(list(delta[int(len(delta)/2):]))
        
        
        #print('d_gd Hamiltonian______________________')
        #print(d_gd)        
        
        #mod.manifold.fill(intermediate_states[0])
              
        #print('d_gd field generator__________________')
        #print(mod.field_generator()(mod.manifold.gd))
        
        #print('========================================')
        
 
        h.module.manifold.muladd_gd(d_mom, step)
        h.module.manifold.muladd_cotan(d_gd, -step)
        
        intermediate_states.append(h.module.manifold.copy())
        intermediate_controls.append(h.module.controls)
      
    return intermediate_states, intermediate_controls



def shoot_euler_silent(h, points, it):
    step = 1. / it
     
    #compound_silent = []
    #man_grid = dm.manifold.Landmarks(h.module.manifold.dim, nb_pts = len(points), gd=points.clone().view(-1).requires_grad_())
    #SilentGrid = dm.deformationmodules.SilentPoints(man_grid)
    #for m in h.module.module_list[:-1]:
    #    compound_silent.append(dm.deformationmodules.CompoundModule([*m.module_list.copy(), SilentGrid.copy()]))
    
    multi_silent = MultiShapeModule_silent([m.copy() for m in h.module.module_list[:-1]], sigma_background = h.module.background.sigma, silentpoints=points)
    multi_silent.fill_cotan_without_silentpoints(h.module.manifold.cotan)
    
    H_silent = dm.hamiltonian_multishape.Hamiltonian_multi(multi_silent, h.constraints)
    H_silent.module.manifold.fill(H_silent.module.manifold.copy())
    
    
    intermediate_states = [H_silent.module.manifold.copy()]
    intermediate_controls = []
    intermediate_points = [points.view(-1) for i in range(len(H_silent.module.module_list)-1)]
    
    for i in range(it):
        H_silent.geodesic_controls()
        
        l = [*H_silent.module.manifold.unroll_gd(), *H_silent.module.manifold.unroll_cotan()]
        delta = grad(H_silent(), l, create_graph=True, allow_unused=True)
        
        d_gd = H_silent.module.manifold.roll_gd(list(delta[:int(len(delta)/2)]))
        d_mom = H_silent.module.manifold.roll_cotan(list(delta[int(len(delta)/2):]))

        H_silent.module.manifold.muladd_gd(d_mom, step)
        H_silent.module.manifold.muladd_cotan(d_gd, -step)
        
        intermediate_states.append(H_silent.module.manifold.copy())
        intermediate_controls.append(H_silent.module.controls.copy())
        
    intermediate_points = [[intermediate_states[k][i][-1].gd.view(points.shape) for i in range(len(H_silent.module.manifold.manifold_list))] for k in range(it+1)]
                
    return intermediate_states, intermediate_controls, intermediate_points

def shoot_euler_controls(h, controls, it):
    assert len(controls) == it
    step = 1. / it

    intermediate_states = [h.module.manifold.copy()]
    for i in range(it):
        h.module.fill_controls(controls[i])
        l = [*h.module.manifold.unroll_gd(), *h.module.manifold.unroll_cotan()]
        delta = grad(h(), l, create_graph=True, allow_unused=True)
        # TODO: is list() necessary?
        d_gd = h.module.manifold.roll_gd(list(delta[:int(len(delta)/2)]))
        d_mom = h.module.manifold.roll_cotan(list(delta[int(len(delta)/2):]))
        h.module.manifold.muladd_gd(d_mom, step)
        h.module.manifold.muladd_cotan(d_gd, -step)
        intermediate_states.append(h.module.manifold.copy())
        intermediate_controls.append(h.module.controls)

    return intermediate_states, intermediate_controls


def shoot_euler_controls(h, controls, it):
    assert len(controls) == it
    step = 1. / it

    intermediate_states = [h.module.manifold.copy()]
    for i in range(it):
        h.module.fill_controls(controls[i])
        l = [*h.module.manifold.unroll_gd(), *h.module.manifold.unroll_cotan()]
        delta = grad(h(), l, create_graph=True)

        d_gd = h.module.manifold.roll_gd(list(delta[:int(len(delta)/2)]))
        d_mom = h.module.manifold.roll_cotan(list(delta[int(len(delta)/2):]))
        d_gd2 = [gdi.action(modulei).tan for gdi, modulei in zip(h.module.manifold.manifold_list, h.module)]
        #print(d_gd)
        #print('----------------------')
        #print(d_gd2)
        
        h.module.manifold.muladd_gd(d_mom, step)
        h.module.manifold.muladd_cotan(d_gd, -step)

        intermediate.append(h.module.manifold.copy())
        
        modules_t = [*modules_t, h.module]

    return intermediate, modules_t



def shoot_torchdiffeq(h, it, method='rk4'):
    # Wrapper class used by TorchDiffEq
    # Returns (\partial H \over \partial p, -\partial H \over \partial q)
    class TorchDiffEqHamiltonianGrad(Hamiltonian, torch.nn.Module):
        def __init__(self, module):
            super().__init__(module)

        def __call__(self, t, x):
            with torch.enable_grad():
                gd, mom = [], []
                index = 0

                for m in self.module:
                    for i in range(m.manifold.len_gd):
                        gd.append(x[0][index:index+m.manifold.dim_gd[i]].requires_grad_())
                        mom.append(x[1][index:index+m.manifold.dim_gd[i]].requires_grad_())
                        index = index + m.manifold.dim_gd[i]

                self.module.manifold.fill_gd(self.module.manifold.roll_gd(gd))
                self.module.manifold.fill_cotan(self.module.manifold.roll_cotan(mom))

                self.geodesic_controls()
                delta = grad(super().__call__(),
                             [*self.module.manifold.unroll_gd(),
                              *self.module.manifold.unroll_cotan()],
                             create_graph=True)

                gd_out = delta[:int(len(delta)/2)]
                mom_out = delta[int(len(delta)/2):]

                return torch.cat(list(map(lambda x: x.view(-1), [*mom_out, *list(map(lambda x: -x, gd_out))])), dim=0).view(2, -1)

    steps = it + 1
    intermediate = []
    init_manifold = h.module.manifold.copy()

    x_0 = torch.cat(list(map(lambda x: x.view(-1), [*h.module.manifold.unroll_gd(), *h.module.manifold.unroll_cotan()])), dim=0).view(2, -1)
    x_1 = odeint(TorchDiffEqHamiltonianGrad.from_hamiltonian(h), x_0, torch.linspace(0., 1., steps), method=method)

    gd, mom = [], []
    index = 0
    for m in h.module:
        for i in range(m.manifold.len_gd):
            gd.append(x_1[-1, 0, index:index+m.manifold.dim_gd[i]])
            mom.append(x_1[-1, 1, index:index+m.manifold.dim_gd[i]])
            index = index + m.manifold.dim_gd[i]

    h.module.manifold.fill_gd(h.module.manifold.roll_gd(gd))
    h.module.manifold.fill_cotan(h.module.manifold.roll_cotan(mom))

    # TODO: very very dirty, change this
    for i in range(0, steps):
        gd, mom = [], []
        index = 0
        for m in h.module:
            for j in range(m.manifold.len_gd):
                gd.append(x_1[i, 0, index:index+m.manifold.dim_gd[j]])
                mom.append(x_1[i, 1, index:index+m.manifold.dim_gd[j]])
                index = index + m.manifold.dim_gd[j]

        intermediate.append(init_manifold.copy())

        intermediate[-1].roll_gd(gd)
        intermediate[-1].roll_cotan(mom)
        intermediate[-1].fill_gd(gd)
        intermediate[-1].fill_cotan(mom)

    return intermediate

