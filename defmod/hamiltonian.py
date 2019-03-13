import copy

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import grad

from .kernels import scal
from .usefulfunctions import make_grad_graph

class Hamiltonian(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.__module = module

    @classmethod
    def from_hamiltonian(cls, class_instance):
        module = class_instance.module
        return cls(module)

    @property
    def module(self):
        return self.__module

    def __call__(self):
        """Computes the hamiltonian."""
        return self.apply_mom() - self.__module.cost()

    def apply_mom(self):
        """Apply the moment on the geodesic descriptors."""
        a = self.__module.manifold.inner_prod_module(self.__module)
        make_grad_graph(a, "apply_mom")
        return a

    def apply_mom_controls(self, controls):
        self.__module.fill_controls(controls)
        a = self.__module.manifold.inner_prod_module(self.__module)
        return a

    def geodesic_controls(self):
        # Fill initial zero controls
        #self.__module.fill_controls(torch.zeros(self.__module.dim_controls, requires_grad=True))
        #a = self.__module.controls
        init_controls = torch.zeros(self.__module.dim_controls, requires_grad=True)
        a = self.apply_mom_controls(init_controls)
        controls = grad(a, [init_controls], create_graph=True)[0]
        self.module.compute_geodesic_control(controls)

