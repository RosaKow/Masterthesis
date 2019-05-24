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
    @property
    def constraints(self):
        return self.__constr

    def __call__(self):
        """Computes the hamiltonian."""
        return self.apply_mom() - self.__modules.cost() - self.apply_constr()

    def apply_mom(self):
        """Apply the moment on the geodesic descriptors."""     
        return sum([self.__modules.manifold.manifold_list[i].inner_prod_field(self.__modules[i].field_generator()) for i in range(len(self.__modules.module_list))])
    #sum([mod.manifold.inner_prod_module(mod) for mod in self.__modules])

    def geodesic_controls(self):
        self.__modules.compute_geodesic_variables_silent(self.__constr)
        
    def apply_constr(self):
        """ Apply Constraints on the generated vectorfields"""
        appl_constr = torch.dot(self.__modules.l.view(-1,1).squeeze(), self.__constr(self.__modules).view(-1,1).squeeze())
        return (appl_constr)