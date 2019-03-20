import copy
import time

import torch
import torch.nn as nn
import torch.optim
import numpy as np
import matplotlib.pyplot as plt

from .deformationmodules import SilentPoints, Translations, CompoundModule
from .manifold import Landmarks
from .hamiltonian import Hamiltonian
from .shooting import shoot 
from .usefulfunctions import AABB, grid2vec, vec2grid, make_grad_graph
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


class Model():
    def __init__(self):
        super().__init__()

    def fidelity(self, target):
        raise NotImplementedError

    def __call__(self, reverse=False):
        raise NotImplementedError

    def transform_target(self, target):
        return target

    def cost(self, target):
        #self.shoot()
        deformation_cost = self.compound.cost()
        attach = self.fidelity(target)
        return attach, deformation_cost

    def fit(self, target, lr=1e-3, l=1., max_iter=100, tol=1e-7, log_interval=10):
        transformed_target = self.transform_target(target)

        optimizer = torch.optim.SGD(self.parameters, lr=lr, momentum=0.05)
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
                cost.backward(retain_graph=True)

            self.clear_shot()

            #make_grad_graph(cost, str("cost" + str(self.nit)))
            
            return cost

        for i in range(0, max_iter):
            optimizer.step(closure)

            if(self.break_loop):
                break

        print("End of the optimisation process.")
        return costs


class ModelCompound(Model):
    def __init__(self, modules, fixed):
        super().__init__()
        self.__module_list = modules
        self.__fixed = fixed

        self.__compound = CompoundModule(self.__module_list)
        self.__init_manifold = self.__compound.manifold.copy()

        self.__parameters = []

        for i in range(len(self.__module_list)):
            self.__parameters.append(self.__init_manifold[i].cotan)
            if(not self.__fixed[i]):
                self.__parameters.append(self.__init_manifold[i].gd)

        self.__shot = False

    @property
    def module_list(self):
        return self.__module_list

    @property
    def fixed(self):
        return self.__fixed

    @property
    def compound(self):
        return self.__compound

    @property
    def init_manifold(self):
        return self.__init_manifold

    @property
    def parameters(self):
        return self.__parameters

    @property
    def shot(self):
        return self.__shot

    def clear_shot(self):
        self.__shot = False

    def reset_state(self):
        self.compound.manifold.fill(self.__init_manifold, copy=True)
        self.__shot = False

    def shoot(self, it=3):
        if not self.__shot:
            self.reset_state()
            h = Hamiltonian(self.__compound)
            shoot(h, it=2, method='rk4')
            self.__shot = True

    def compute_deformation_grid(self, grid_origin, grid_size, grid_resolution, it=2, intermediate=False):
        x, y = torch.meshgrid([
            torch.linspace(grid_origin[0], grid_origin[0]+grid_size[0], grid_resolution[0]),
            torch.linspace(grid_origin[1], grid_origin[1]+grid_size[1], grid_resolution[1])])

        gridpos = grid2vec(x, y)

        self.reset_state()
        grid_landmarks = Landmarks(2, gridpos.shape[0], gd=gridpos.view(-1))
        grid_silent = SilentPoints(grid_landmarks)

        intermediate = shoot(Hamiltonian([grid_silent, *self.module_list]))

        return vec2grid(grid_landmarks.gd.view(-1, 2).detach(), grid_resolution[0], grid_resolution[1])


class ModelCompoundWithPointsRegistration(ModelCompound):
    def __init__(self, source, module_list, fixed):
        self.alpha = source[1]

        module_list.insert(0, SilentPoints(Landmarks(2, source[0].shape[0], gd=source[0].view(-1).requires_grad_())))
        fixed.insert(0, True)

        super().__init__(module_list, fixed)

    def fidelity(self, target, it=10):
        return fidelity(self(), target)

    def __call__(self, it=3):
        self.shoot(it=it)
        return self.compound[0].manifold.gd.view(-1, 2), self.alpha


class ModelCompoundImageRegistration(ModelCompound):
    def __init__(self, source_image, module_list, fixed, img_transform=lambda x: x):
        self.__frame_res = source_image.shape
        self.__source = sample_from_greyscale(source_image.clone(), 0., centered=False, normalise_weights=False, normalise_position=False)
        self.__img_transform = img_transform
        super().__init__(module_list, fixed)

    def transform_target(self, target):
        return self.__img_transform(target)

    def fidelity(self, target):
        return L2_norm_fidelity(self(), target)

    def __call__(self, it=2, intermediate=False):
        if not self.shot:
            # First, forward step shooting only the deformation modules
            self.reset_state()
            self.shoot(it=6)

            # Prepare for reverse shooting
            self.compound.manifold.negate_cotan()

            image_landmarks = Landmarks(2, self.__source[0].shape[0], gd=self.__source[0].view(-1))
            compound = [SilentPoints(image_landmarks), *self.compound]

            # Then, reverse shooting in order to get the final deformed image
            intermediate = shoot(Hamiltonian(compound), it=6)

            self.__output_image = deformed_intensities(compound[0].manifold.gd.view(-1, 2), self.__source[1].view(self.__frame_res))

        return self.__output_image

