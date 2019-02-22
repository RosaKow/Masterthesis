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

    @classmethod
    def from_hamiltonian(cls, class_instance):
        def_module = copy.deepcopy(class_instance.def_module)
        return cls(def_module)

    @property
    def def_module(self):
        return self.__def_module

    def __call__(self, gd, mom, controls):
        """Computes the hamiltonian."""
        def_cost = self.__def_module.cost(gd, controls)
        return self.apply_mom(gd, mom, controls) - def_cost

    def apply_mom(self, gd, mom, controls):
        """Apply the moment on the geodesic descriptors."""
        speed = self.__def_module.action(gd, self.__def_module, gd, controls)
        return scal(mom, speed)

    def geodesic_controls(self, gd, mom):
        init_controls = torch.zeros(self.__def_module.dim_controls, requires_grad=True)
        controls = grad(self.apply_mom(gd, mom, init_controls),
                        [init_controls], create_graph=True)[0]
        return self.__def_module.compute_geodesic_control(controls, gd)

    # def geodesic_controls(self, gd, mom):
    #    return self.__def_module.compute_geodesic_control(
    #        self.__def_module.apply_adjoint(gd, self.__def_module, gd, mom), gd)

