import copy

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import grad

from .kernels import scal
from .usefulfunctions import make_grad_graph
from .deformationmodules import CompoundModule

class Hamiltonian_multi:
    def __init__(self, modules, constr):
            self.__modules = modules
            self.__constr = constr

    @property
    def module(self):
        return self.__modules

    def __call__(self):
        """Computes the hamiltonian."""
        return self.apply_mom() - self.__modules.cost() - self.apply_constr()

    def apply_mom(self):
        """Apply the moment on the geodesic descriptors."""     
        return sum([mod.manifold.inner_prod_module(mod) for mod in self.__modules])

    def geodesic_controls(self):
        self.__modules.compute_geodesic_variables(self.__modules.manifold, self.__constr)
        
    def apply_constr(self):
        """ Apply Constraints on the generated vectorfields"""
        return torch.dot(self.__modules.l.view(-1,1).squeeze(), self.__constr().view(-1,1).squeeze())
