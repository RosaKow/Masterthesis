import torch
import numpy as np
import torch.nn as nn
from torch.autograd import grad

from kernels import scal

class Hamiltonian_Multishape(nn.Module):
    def __init__(self, module_list, dim_controls):
        super().__init__()
        self.__module_list = module_list
        self.nb_modules = len(module_list)
        self.dim_controls = dim_controls
        self.__init_controls = [torch.zeros(dim_controls[0], requires_grad=True), torch.zeros(dim_controls[1], requires_grad=True), torch.zeros(dim_controls[2], requires_grad=True)]
    

    def __call__(self, gd_list, mom_list, controls_list, z, l_list, Constr):
        """Computes the hamiltonian."""
        gd_module_list = [z[0], z[1], gd_list[2]]
        cost = self.def_cost(gd_list, controls_list, z)
        applmom = self.apply_mom(gd_list, gd_module_list, mom_list, controls_list)
        applconstr = self.apply_constraints(gd_list, gd_module_list, l_list, controls_list, Constr)
        
        return applmom - cost - applconstr

    
    def def_cost(self, gd_list, controls_list, z):
        gd = [z[0], z[1], gd_list[2]]
        
        cost = 0
        for i in range(self.nb_modules):
            cost += self.__module_list[i].cost(gd[i], controls_list[i])
        return cost
        
    
    def apply_mom(self, gd_list, gd_module_list, mom_list, controls_list):
        """Apply the moment on the geodesic descriptors."""
        A = 0
        speed = []
        for i in range(self.nb_modules):
            speed.append(self.__module_list[i].action(gd_list[i], self.__module_list[i], gd_module_list[i], controls_list[i]))
            A = A + scal(mom_list[i], speed[i])
        return A
    

    def apply_constraints(self, gd_list, gd_module_list, l_list, controls_list, Constr):
        A = 0
        speed = []
        for i in range(self.nb_modules):
            speed.append(self.__module_list[i].action(gd_list[i], self.__module_list[i], gd_module_list[i], controls_list[i]).contiguous().view(len(gd_list[i]),2))
        
        v = torch.cat([speed[0], speed[1], speed[2]],0)
        A1 = scal(l_list[0].contiguous(), torch.mm(Constr[0],v).contiguous())
        A2 = scal(l_list[1].contiguous(), torch.mm(Constr[1],v).contiguous())
        return A1 + A2
    
##########################################################################
    # TO DO
    def lambda_qp(self, gd, mom):
        # compute zeta^ast xi^ast p
        zetaxip = grad(self.apply_mom(gd, mom, self.__init_controls),
                        [self.__init_controls], create_graph=True)[0]
        # apply inverse cost operator Z^{-1}
        # by solving equation Z X = zetaxip
        K_q = K_xx(gd.view(-1, self.__dim), self.__sigma)       #???
        X, _ = torch.gesv(zetaxip.view(-1, self.__dim), K_q)
        X = X.contiguous().view(-1)
        # apply field generator
        Y = self.__module_list.action(gd, self.__module_list, gd, X)
        # apply constraints operator
        #CY = 
        # compute alpha = zeta^ast C^ast
        
        # apply inverse cost operator
        
        # apply field generator
        
        # apply constraints function
        #B = 
        # apply B^{-1}
        # by solving equation B l = CY
        l, _ = torch_gesv(CY.view(-1, self.__dim))
        return l
    
    # TODO 
    # (= geodesic_controls)
    def h_qp(self, gd, mom, l):
        zetaxip = grad(self.apply_mom(gd, mom, self.__init_controls),
                        [self.__init_controls], create_graph=True)[0]
        zetaClambda = grad(self.apply_constraints(gd, l, self.__init_controls),
                        [self.__init_controls], create_graph=True)[0]
        return self.__module_list.inverse_cost(zetaxip - zetaClambda, gd)
    

