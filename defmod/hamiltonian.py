import copy
from collections import Iterable

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import grad

from .kernels import scal
from .usefulfunctions import make_grad_graph
from .deformationmodules import CompoundModule

class Hamiltonian(nn.Module):
    def __init__(self, module_list):
        assert isinstance(module_list, Iterable)
        super().__init__()
        self.__module = CompoundModule(module_list)

    @classmethod
    def from_hamiltonian(cls, class_instance):
        return cls(class_instance.module)

    @property
    def module(self):
        return self.__module

    def __call__(self):
        """Computes the hamiltonian."""
        return self.apply_mom() - self.__module.cost()

    def apply_mom(self):
        """Apply the moment on the geodesic descriptors."""
        return self.__module.manifold.inner_prod_module(self.__module)

    def geodesic_controls(self):
        self.__module.fill_controls_zero()
        controls = grad(self.apply_mom(), self.__module.controls, create_graph=True, allow_unused=True)
        self.module.compute_geodesic_control(list(controls))

