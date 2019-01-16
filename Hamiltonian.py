import torch
import numpy as np
import torch.nn as nn
from torch.autograd import grad

import UsefulFunctions as fun

class Hamilt(nn.Module):
    def __init__(self, DefModule):
        super(Hamilt, self).__init__()
        self.DefModule = DefModule
        self.init_cont = torch.zeros(DefModule.nb_pts, 2, requires_grad=True)

    def apply_Mom(self, GD, MOM, Cont):
        speed = self.DefModule.action(GD, self.DefModule, GD, Cont)
        return fun.scal(MOM, speed)

    def __call__(self, GD, MOM, Cont):
        "Computes the hamiltonian."
        
        co = self.DefModule.cost(GD, Cont)
        return self.apply_Mom(GD, MOM, Cont) - co

    def Cont_geo(self, GD, MOM):
        cont0 = grad(self.apply_Mom(GD, MOM, self.init_cont), [self.init_cont], create_graph=True)[0]
        return self.DefModule.compute_geodesic_control(cont0, GD) 


def H_r(Mod, GD, MOM, H):
    return H(GD, MOM, H.Cont_geo(Mod, GD, MOM))

