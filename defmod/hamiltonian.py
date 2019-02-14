import copy

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import grad

from .kernels import scal

class Hamiltonian(nn.Module):
    def __init__(self, def_module):
        super().__init__()
        self.__def_module = def_module
        self.__init_controls = torch.zeros(def_module.dim_controls, requires_grad=True)

    @classmethod
    def from_hamiltonian(cls, class_instance):
        def_module = copy.deepcopy(class_instance.def_module)
        return cls(def_module)

    @property
    def def_module(self):
        return self.__def_module

    @property
    def init_controls(self):
        return self.__init_controls

    def __call__(self, gd, mom, controls):
        """Computes the hamiltonian."""
        def_cost = self.__def_module.cost(gd, controls)
        return self.apply_mom(gd, mom, controls) - def_cost

    def apply_mom(self, gd, mom, controls):
        """Apply the moment on the geodesic descriptors."""
        speed = self.__def_module.action(gd, self.__def_module, gd, controls)
        return scal(mom, speed)

    # TODO: Find a better name for this function.
    # TODO: Manualy compute the gradient so we can use the torchdifeq library for the shooting.
    # def geodesic_controls(self, gd, mom):
    #     controls = grad(self.apply_mom(gd, mom, self.__init_controls),
    #                     [self.__init_controls], create_graph=True)[0]
    #     return self.__def_module.compute_geodesic_control(controls, gd)

    def geodesic_controls(self, gd, mom):
       return self.__def_module.compute_geodesic_control(
           self.__def_module.apply_adjoint(gd, self.__def_module, gd, mom), gd)

