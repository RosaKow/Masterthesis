import copy

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
    def __init__(self, dim, module_list, gd_list, fixed):
        super(Model, self).__init__()
        self.dim = dim

        self.gd_params, self.mom_params = torch.nn.ParameterList(), torch.nn.ParameterList()
        self.gd_fixed = []
        self.module_list = []

        for i in range(len(module_list)):
            if(fixed[i]):
                self.gd_fixed.append(gd_list[i].clone().requires_grad_())
                self.module_list.append(module_list[i])

        for i in range(len(module_list)):
            if(not fixed[i]):
                self.gd_params.append(torch.nn.Parameter(gd_list[i].clone()))
                self.mom_params.append(torch.nn.Parameter(torch.zeros_like(gd_list[i])))
                self.module_list.append(module_list[i])

        self.compound = Compound(self.module_list)

    def get_var_tensor(self):
        gd_list = [*self.gd_fixed, *self.gd_params]
        mom_list = [torch.zeros(sum(a.shape[0] for a in self.gd_fixed)), *self.mom_params]
        return torch.cat(gd_list), torch.cat(mom_list)

    def get_var_list(self):
        mom_fixed = []
        for fixed in self.gd_fixed:
            mom_fixed.append(torch.zeros_like(fixed))

        return [*self.gd_fixed, *self.gd_params], [*mom_fixed, *self.mom_params]

    def shoot_tensor(self):
        gd_in, mom_in = self.get_var_tensor()
        return shoot(gd_in, mom_in, Hamiltonian(self.compound))

    def shoot_list(self):
        """Solves the shooting equations and returns the result as lists."""
        gd, mom = self.get_var_tensor()
        gd, mom = shoot(gd, mom, Hamiltonian(self.compound))
        gd_list = []
        mom_list = []
        for i in range(self.compound.nb_module):
            gd_list.append(gd[self.compound.indice_gd[i]:self.compound.indice_gd[i+1]])
            mom_list.append(mom[self.compound.indice_gd[i]:self.compound.indice_gd[i+1]])
        return gd_list, mom_list

    def compute_deformation_grid(self, grid_origin, grid_size, grid_resolution):
        x, y = torch.meshgrid([
            torch.linspace(grid_origin[0], grid_origin[0]+grid_size[0], grid_resolution[0]),
            torch.linspace(grid_origin[1], grid_origin[1]+grid_size[1], grid_resolution[1])])

        gridpos = grid2vec(x, y)
        gd_model, mom_model = self.get_var_tensor()
        gd = torch.cat([gridpos.view(-1), gd_model])
        mom = torch.cat([torch.zeros_like(gridpos.view(-1)), mom_model])
        modules = Compound([SilentPoints(2, gridpos.shape[0]), *self.module_list])
        gd_out, _ = shoot(gd, mom, Hamiltonian(modules))

        return vec2grid(gd_out[0:gridpos.view(-1).shape[0]].view(-1, 2), grid_resolution[0], grid_resolution[1])


class ModelCompoundWithPointsRegistration(ModelCompound):
    def __init__(self, dim, source, module_list, gd_list, fixed):
        self.alpha = source[1]
        module_list.insert(0, SilentPoints(dim, source[0].shape[0]))
        gd_list.insert(0, source[0].view(-1))
        fixed.insert(0, True)
        super().__init__(dim, module_list, gd_list, fixed)

    def fidelity(self, target):
        return fidelity(self(), target)

    def __call__(self):
        gd_list, _ = self.shoot_list()
        return gd_list[0].view(-1, 2), self.alpha


class ModelCompoundImageRegistration(ModelCompound):
    def __init__(self, dim, source_image, module_list, gd_list, fixed):
        self.frame_res = source_image.shape
        self.source = sample_from_greyscale(source_image, 0., centered=False, normalise_weights=False, normalise_position=False)
        super().__init__(dim, module_list, gd_list, fixed)

    def transform_target(self, target):
        return target

    def fidelity(self, target):
        return L2_norm_fidelity(self(), target)

    def __call__(self):
        module_gd, module_mom = self.get_var_tensor()
        silent_gd = self.source[0].view(-1)
        silent_mom = torch.zeros_like(silent_gd)
        gd, mom = torch.cat([silent_gd, module_gd]), torch.cat([silent_mom, module_mom])
        compound = Compound([SilentPoints(2, self.source[0].shape[0]), *self.module_list])
        gd_out, mom_out = shoot(gd, mom, Hamiltonian(compound), reverse=True)

        return torch.flip(deformed_intensities(gd_out[compound.indice_gd[0]:compound.indice_gd[1]].view(-1, 2), self.source[1].view(self.frame_res)), dims=[0])


