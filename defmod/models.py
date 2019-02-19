import itertools

import torch
import torch.nn as nn
import torch.optim
import numpy as np
import matplotlib.pyplot as plt

from .deformationmodules import SilentPoints, Translations, Compound
from .hamiltonian import Hamiltonian
from .shooting import shoot 
from .usefulfunctions import AABB
from .kernels import distances, scal
from .sampling import sample_from_greyscale, sample_from_smoothed_points


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
    
    def cost(self, target):
        gd, mom = self.get_var_tensor()
        attach = self.fidelity(target)
        deformation_cost = self.get_module_compound().cost(gd, self.H.geodesic_controls(gd, mom))
        return attach, deformation_cost

    def fit(self, target, lr=1e-3, l=1., max_iter=100, tol=1e-7, log_interval=10):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.05)
        self.nit = -1
        self.break_loop = False
        costs = []

        def closure():
            self.nit += 1
            optimizer.zero_grad()
            attach, deformation_cost = self.cost(target)
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


class ModelTranslationModuleRegistration(Model):
    def __init__(self, dim, source, sigma, translation_gd, fixed_translation_points=True):
        super().__init__()
        self.dim = dim
        self.init_points = source[0].clone()
        self.alpha = source[1].clone()
        self.sigma = sigma

        self.silent_module = SilentPoints(self.dim, self.init_points.shape[0])
        self.translation_module = Translations(self.dim, translation_gd.view(-1, self.dim).shape[0], self.sigma)
        self.compound = Compound([self.silent_module, self.translation_module])
        self.H = Hamiltonian(self.compound)

        self.translation_mom = torch.nn.Parameter(torch.zeros_like(translation_gd).view(-1))
        self.silent_mom = torch.nn.Parameter(torch.zeros_like(self.init_points).view(-1))

        self.translation_gd = None
        if(fixed_translation_points):
            self.translation_gd = translation_gd.clone().requires_grad_()
        else:
            self.translation_gd = torch.nn.Parameter(translationGD.clone())

    def fidelity(self, target):
        return fidelity(self(), target)

    def get_var_tensor(self):
        """Returns the variables from the problem (GDs and MOMs) as tensors."""
        return torch.cat((self.init_points.view(-1), self.translation_gd), 0), torch.cat((self.silent_mom, self.translation_mom), 0)

    def get_var_list(self):
        """Returns the variables from the problem (GDs and MOMs) as lists."""
        return [self.init_points.view(-1), self.translation_gd], [self.silent_mom, self.translation_mom]

    def shoot_tensor(self):
        """Solves the shooting equations and returns the result as tensors."""
        gd, mom = self.get_var_tensor()
        return shoot(gd, mom, self.H)

    def shoot_list(self):
        """Solves the shooting equations and returns the result as lists."""
        gd, mom = self.get_var_tensor()
        gd, mom = shoot(gd, mom, self.H)
        return [gd.view(-1, self.dim)[0:self.init_points.shape[0]].view(-1), gd.view(-1, self.dim)[self.init_points.shape[0]:].view(-1)], [mom.view(-1, self.dim)[0:self.init_points.shape[0]].view(-1), mom.view(-1, self.dim)[self.init_points.shape[0]:].view(-1)]

    def get_module_compound(self):
        return self.compound

    def __call__(self):
        """Returns the projected data by the deformation modules."""
        gd_in, mom_in = self.get_var_tensor()
        gd_out, mom_out = shoot(gd_in, mom_in, self.H)
        return gd_out[0:self.init_points.shape[0]*self.dim].view(-1, self.dim), self.alpha
    

class ModelCompoundRegistration(Model):
    def __init__(self, dim, source, module_list, gd_list, fixed):
        super(Model, self).__init__()
        self.dim = dim
        self.init_points = source[0].clone()
        self.alpha = source[1].clone()

        self.mom_silent = torch.nn.Parameter(torch.zeros_like(self.init_points).view(-1))

        self.gd_params, self.mom_params = torch.nn.ParameterList(), torch.nn.ParameterList()
        self.gd_fixed = []
        self.module_list = [SilentPoints(self.dim, self.init_points.shape[0])]

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
        self.H = Hamiltonian(self.compound)

    def fidelity(self, target):
        return fidelity(self(), target)

    def get_var_tensor(self):
        gd_list = [self.init_points.view(-1), *self.gd_fixed, *self.gd_params]
        mom_list = [self.mom_silent, torch.zeros(sum(a.shape[0] for a in self.gd_fixed)), *self.mom_params]
        return torch.cat(gd_list), torch.cat(mom_list)

    def get_var_list(self):
        mom_fixed = []
        for a in self.gd_fixed:
            mom_fixed.append(torch.zeros_like(a))
            
        return [self.init_points.view(-1), *self.gd_fixed, *self.gd_params], [self.mom_silent, *mom_fixed, *self.mom_params]

    def shoot_tensor(self):
        gd_in, mom_in = self.get_var_tensor()
        return shoot(gd_in, mom_in, self.H)

    def shoot_list(self, it=10):
        """Solves the shooting equations and returns the result as lists."""
        gd, mom = self.get_var_tensor()
        gd, mom = shoot(gd, mom, self.H)
        gd_list = []
        mom_list = []
        for i in range(self.compound.nb_module):
            gd_list.append(gd[self.compound.indice_gd[i]:self.compound.indice_gd[i+1]])
            mom_list.append(mom[self.compound.indice_gd[i]:self.compound.indice_gd[i+1]])
        return gd_list, mom_list

    def get_module_compound(self):
        return self.compound

    def __call__(self):
        gd_in, mom_in = self.get_var_tensor()
        gd_out, mom_out = shoot(gd_in, mom_in, self.H)
        return gd_out.view(-1, self.dim)[0:self.init_points.shape[0]], self.alpha
        

import matplotlib.pyplot as plt
    
class ModelCompoundImageRegistration(ModelCompoundRegistration):
    def __init__(self, dim, source_image, module_list, gd_list, fixed, threshold=0.5):
        source = sample_from_greyscale(source_image, threshold, centered=False, normalise_weights=False, normalise_position=False)
        sampled = sample_from_smoothed_points(source, source_image.shape, sigma=2., normalize=2.)
        self.frame_res = source_image.shape
        super().__init__(dim, source, module_list, gd_list, fixed)

    def fidelity(self, target):
        sampled_image = sample_from_smoothed_points(self(), self.frame_res, sigma=1., normalize=False, aabb=AABB(0., self.frame_res[0], 0., self.frame_res[1]))
        target = sample_from_smoothed_points(sample_from_greyscale(target, threshold=0.5, centered=False, normalise_weights=False, normalise_position=False), self.frame_res, sigma=1., normalize=False, aabb=AABB(0., self.frame_res[0], 0., self.frame_res[1]))
        # plt.subplot(1, 2, 1)
        # plt.imshow(sampled_image.detach())
        # plt.subplot(1, 2, 2)
        # plt.imshow(target)
        # plt.show()
        #target = torch.flip(target, dims=[0])
        return L2_norm_fidelity(sampled_image, target)

