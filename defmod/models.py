import copy
import time

import torch
import torch.nn as nn
import torch.optim
import numpy as np
import matplotlib.pyplot as plt

from .deformationmodules import SilentPoints, Translations, Compound
from .hamiltonian import Hamiltonian
from .shooting import shoot 
from .usefulfunctions import AABB, grid2vec, vec2grid
from .kernels import distances, scal
from .sampling import sample_from_greyscale, sample_from_smoothed_points, resample_image_to_smoothed, deformed_intensities


def fidelity(a, b):
    """Energy Distance between two sampled probability measures."""
    x_i, a_i = a
    y_j, b_j = b
    K_xx = -distances(x_i, x_i)
    K_xy = -distances(x_i, y_j)
    K_yy = -distances(y_j, y_j)
    cost = .5*scal(a_i, torch.mm(K_xx, a_i.view(-1, 1))) - scal(a_i, torch.mm(K_xy, b_j.view(-1, 1))) + .5*scal(b_j, torch.mm(K_yy, b_j.view(-1, 1)))
    return cost


def L2_norm_fidelity(a, b):
    return torch.dist(a, b)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, it=10):
        raise NotImplementedError

    def get_var_tensor(self):
        raise NotImplementedError

    def get_var_list(self):
        raise NotImplementedError

    def get_module_compound(self):
        raise NotImplementedError

    def shoot_tensor(self):
        raise NotImplementedError

    def shoot_list(self):
        raise NotImplementedError

    def fidelity(self, target):
        raise NotImplementedError

    def __call__(self, reverse=False):
        raise NotImplementedError

    def transform_target(self, target):
        return target

    def cost(self, target):
        gd, mom = self.get_var_tensor()
        attach = self.fidelity(target)
        deformation_cost = self.compound.cost(gd, Hamiltonian(self.compound).geodesic_controls(gd, mom))
        return attach, deformation_cost

    def fit(self, target, lr=1e-3, l=1., max_iter=100, tol=1e-7, log_interval=10):
        transformed_target = copy.deepcopy(self.transform_target(target))

        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.05)
        self.nit = -1
        self.break_loop = False
        costs = []

        def closure():
            self.nit += 1
            optimizer.zero_grad()
            attach, deformation_cost = self.cost(transformed_target)
            attach = l*attach
            cost = attach + deformation_cost

            if(self.nit%log_interval == 0):
                print("It: %d, deformation cost: %.6f, attach: %.6f. Total cost: %.6f" % (self.nit, deformation_cost.detach().numpy(), attach.detach().numpy(), cost.detach().numpy()))

            costs.append(cost.item())

            if(len(costs) > 1 and abs(costs[-1] - costs[-2]) < tol) or self.nit >= max_iter:
                self.break_loop = True
            else:
                cost.backward()

            return cost

        for i in range(0, max_iter):
            optimizer.step(closure)

            if(self.break_loop):
                break

        print("End of the optimisation process")
        return costs


class ModelCompound(Model):
    def __init__(self, dim, module_list, gd_list, fixed_gd):
        super(Model, self).__init__()
        self.dim = dim

        self.gd_params, self.mom_params = torch.nn.ParameterList(), torch.nn.ParameterList()
        self.gd_fixed = []
        self.module_list = []

        for i in range(len(module_list)):
            if(fixed_gd[i]):
                self.gd_fixed.append(gd_list[i].clone().requires_grad_())
                self.mom_params.append(torch.nn.Parameter(torch.zeros_like(gd_list[i])))
                self.module_list.append(module_list[i])

        for i in range(len(module_list)):
            if(not fixed_gd[i]):
                self.gd_params.append(torch.nn.Parameter(gd_list[i].clone()))
                self.mom_params.append(torch.nn.Parameter(torch.zeros_like(gd_list[i])))
                self.module_list.append(module_list[i])

        self.compound = Compound(self.module_list)

    def get_var_tensor(self):
        return torch.cat([*self.gd_fixed, *self.gd_params]), torch.cat([*self.mom_params])

    def get_var_list(self):
        return [*self.gd_fixed, *self.gd_params], self.mom_params

    def shoot_tensor(self, it=2, intermediate=False):
        gd_in, mom_in = self.get_var_tensor()

        gd_out, mom_out = shoot(gd_in, mom_in, Hamiltonian(self.compound), it=it, intermediate=intermediate, output_list=True)
        if(intermediate):
            return gd_out, mom_out
        else:
            return [gd_out[-1]], [mom_out[-1]]


    def shoot_list(self, it=2, intermediate=False):
        """Solves the shooting equations and returns the result as lists."""
        gd_in, mom_in = self.get_var_tensor()
        gd, mom = shoot(gd_in, mom_in, Hamiltonian(self.compound), it=it, intermediate=intermediate, output_list=True)
        
        gd_out = []
        mom_out = []

        its = [-1]
        if(intermediate):
            its = range(it)

        for i in its:
            gd_list = []
            mom_list = []
            for j in range(self.compound.nb_module):
                gd_list.append(gd[i][self.compound.indice_gd[j]:self.compound.indice_gd[j+1]])
                mom_list.append(mom[i][self.compound.indice_gd[j]:self.compound.indice_gd[j+1]])

            gd_out.append(gd_list)
            mom_out.append(mom_list)

        return gd_out, mom_out


    def compute_deformation_grid(self, grid_origin, grid_size, grid_resolution, it=2, intermediate=False):
        x, y = torch.meshgrid([
            torch.linspace(grid_origin[0], grid_origin[0]+grid_size[0], grid_resolution[0]),
            torch.linspace(grid_origin[1], grid_origin[1]+grid_size[1], grid_resolution[1])])

        gridpos = grid2vec(x, y)
        gd_model, mom_model = self.get_var_tensor()
        gd = torch.cat([gridpos.view(-1), gd_model])
        mom = torch.cat([torch.zeros_like(gridpos.view(-1)), mom_model])
        modules = Compound([SilentPoints(2, gridpos.shape[0]), *self.module_list])
        gd_out, _ = shoot(gd, mom, Hamiltonian(modules), it=it,
                          intermediate=intermediate, output_list=True)

        grid_x_out = []
        grid_y_out = []

        its = [-1]
        if(intermediate):
            its = range(it)
            
        for it in its:
            grid_x, grid_y = vec2grid(gd_out[it][0:gridpos.view(-1).shape[0]].view(-1, 2), grid_resolution[0], grid_resolution[1])
            grid_x_out.append(grid_x)
            grid_y_out.append(grid_y)

        return grid_x_out, grid_y_out


class ModelCompoundWithPointsRegistration(ModelCompound):
    def __init__(self, dim, source, module_list, gd_list, fixed_gd):
        self.alpha = source[1]
        module_list.insert(0, SilentPoints(dim, source[0].shape[0]))
        gd_list.insert(0, source[0].view(-1))
        fixed_gd.insert(0, True)
        super().__init__(dim, module_list, gd_list, fixed_gd)

    def fidelity(self, target, it=2):
        points = self(it=it)
        return fidelity((points[0][0], points[1][0]), target)
        

    def __call__(self, it=2, intermediate=False):
        gd_list, _ = self.shoot_list(it=it, intermediate=intermediate)
        return [gd[0].view(-1, 2) for gd in gd_list], [self.alpha]


class ModelCompoundImageRegistration(ModelCompound):
    def __init__(self, dim, source_image, module_list, gd_list, fixed_gd, img_transform=lambda x : x):
        self.frame_res = source_image.shape
        self.source = sample_from_greyscale(source_image, 0., centered=False, normalise_weights=False, normalise_position=False)
        self.img_transform = img_transform
        super().__init__(dim, module_list, gd_list, fixed_gd)

    def transform_target(self, target):
        return self.img_transform(target)

    def fidelity(self, target):
        return L2_norm_fidelity(self.img_transform(self(it=5)[-1]), target)

    def __call__(self, it=2, intermediate=False):
        # First, forward step shooting only the deformation modules
        phi_gd, phi_mom = self.shoot_tensor(it=it)

        # Then, reverse shooting in order to get the final deformed image
        silent_gd = self.source[0].view(-1)
        silent_mom = torch.zeros_like(silent_gd)
        gd, mom = torch.cat([silent_gd, phi_gd[-1]]), torch.cat([silent_mom, -phi_mom[-1]])
        
        compound = Compound([SilentPoints(2, self.source[0].shape[0]), *self.module_list])
        
        gd_out, mom_out = shoot(gd, mom, Hamiltonian(compound), it=it, intermediate=intermediate, output_list=True)

        its = [-1]
        if(intermediate):
            its = range(it)

        return [deformed_intensities(gd_out[i][compound.indice_gd[0]:compound.indice_gd[1]].view(-1, 2), self.source[1].view(self.frame_res)) for i in its]

