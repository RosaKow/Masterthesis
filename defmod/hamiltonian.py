import copy

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import grad

from .kernels import scal

class Hamiltonian(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.__module = module

    @classmethod
    def from_hamiltonian(cls, class_instance):
        module = copy.deepcopy(class_instance.module)
        return cls(module)

    @property
    def module(self):
        return self.__module

    def __call__(self):
        """Computes the hamiltonian."""
        # def_cost = self.module.cost(gd, controls)
        # return self.apply_mom(gd, mom, controls) - def_cost
        return self.apply_mom() - self.__module.cost()
        
    def apply_mom(self):
        """Apply the moment on the geodesic descriptors."""
        # speed = self.module.action(gd, self.module, gd, controls)
        # return scal(mom, speed)
        speed = self.module.action(self.module)
        return scal(self.module.manifold.mom, speed)

    def geodesic_controls(self):
        # Fill initial zero controls
        self.module.fill_controls(torch.zeros(self.module.dim_controls, requires_grad=True))
        controls = grad(self.apply_mom(), [init_controls], create_graph=True)[0]
        self.module.compute_geodesic_control(controls)

    # def geodesic_controls(self, gd, mom):
    #    return self.module.compute_geodesic_control(
    #        self.module.apply_adjoint(gd, self.module, gd, mom), gd)

