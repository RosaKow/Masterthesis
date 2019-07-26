import torch
#import multimodule_usefulfunctions as mm 
#import numpy as np
from defmod.shooting import shoot_euler#, shoot_euler_source
from defmod.attachement import L2NormAttachement_multi, L2NormAttachement


class EnergyFunctional():
    def __init__(self, modules, h, Constr, source, target, dim=2, gamma=1, attach=None):
        super().__init__()
        self.__modules = modules
        self.h = h
        self.Constr = Constr
        self.source = source
        self.target = target
        self.attach_func = attach
        self.dim = dim
        self.nb_pts = [modules.module_list[0].manifold.nb_pts, modules.module_list[1].manifold.nb_pts]
        self.gamma = gamma
        
    @property
    def modules(self):
        return self.__modules
        
    def attach(self):
        if self.attach_func == None:
            return sum([L2NormAttachement()( self.modules.module_list[i][0].manifold.gd, self.target[i]) for i in range(len(self.target))])
        else:
            return self.attach_func(self.modules, self.target)

        
    def cost(self):   
        return self.modules.cost()
      
    def shoot(self):       
        intermediate_states, intermediate_controls = shoot_euler(self.h, it=10)
        return intermediate_states, intermediate_controls    
    
    def shoot_grid(self, gridpoints):
        intermediate_states, intermediate_controls = shoot_euler_silent(self.h, gridpoints, it=10)
        shot_grid = [m.gd[0] for m in intermediate_states[-1]]
        return shot_grid
    
    
    def energy_tensor(self, gd0, mom0):
        ''' Energy functional for tensor input
            (to compute the automatic gradient the input is needed as tensor, not as list) '''
        
        self.h.module.manifold.fill_gd(gd0)
        self.h.module.manifold.fill_cotan(mom0)
        self.h.geodesic_controls()

        self.shoot()
                        
        cost = self.cost()
        attach = self.attach()
        print('cost:', self.gamma * cost.detach().numpy(), 'attach:', attach.detach().numpy())
        
        return self.gamma*cost + attach
        
    
    def energy(self, gd0, mom0):
        gd_t,_,controls_t = self.shoot(gd0, mom0)
        
        geodesicControls0 = controls_t[0]
        gd1 = gd_t[-1]
        
        z = [mm.computeCenter(gd0[0]), mm.computeCenter(gd0[1]), gd0[2]]
        
        cost = self.cost(z, geodesicControls0)
        attach = self.attach(gd1, self.target)
        
        return self.gamma*cost + attach
    
    def gradE(self, gd0_tensor, mom0_tensor):
        E = self.energy_tensor(gd0_tensor, mom0_tensor)
        E.backward(create_graph = True)
        grad = mom0_tensor.grad
        #mom0_tensor.grad.data.zero_()         <- doesn't seem to be needed
        return grad
    
    def gradE_autograd(self, gd0_tensor, mom0_tensor):
        grad = torch.autograd.grad(self.energy_tensor(gd0_tensor, mom0_tensor), mom0_tensor)[0]
        return grad
    
    def tensor2list(self, x):
        x_list = [x[0:self.nb_pts[0]*self.dim], x[self.nb_pts[0]*self.dim:(self.nb_pts[0]+self.nb_pts[1])*self.dim], [x[(self.nb_pts[1]+self.nb_pts[0])*self.dim:(self.nb_pts[1]+2*self.nb_pts[0])*self.dim], x[(self.nb_pts[1]+2*self.nb_pts[0])*self.dim:]]]
        return x_list
    
    def list2tensor(self, x):
        a = [*[a for a in x[:-1]], *[a for a in x[-1]]]
        return torch.cat(a,0).requires_grad_().view(-1).double()
    
    
    
############################################################
class EnergyFunctional_unconstrained():
    def __init__(self, modules, h, source, target, dim=2, gamma=1, attach=None):
        super().__init__()
        self.__modules = modules
        self.h = h
        self.source = source
        self.target = target
        self.attach_func = attach
        self.dim = dim
        self.gamma = gamma
        
    @property
    def modules(self):
        return self.__modules
        
    def attach(self):
        if self.attach_func == None:
            return sum([L2NormAttachement()( self.modules.manifold.gd.view(-1,2), torch.cat(self.target)) ])
        
        else:
            return self.attach_func(self.modules, self.target)

        
    def cost(self):   
        return self.modules.cost()
      
    def shoot(self):       
        intermediate_states, intermediate_controls = shoot_euler(self.h, it=10)
        return intermediate_states, intermediate_controls    
    
    def shoot_grid(self, gridpoints):
        intermediate_states, intermediate_controls = shoot_euler_silent(self.h, gridpoints, it=10)
        shot_grid = [m.gd[0] for m in intermediate_states[-1]]
        return shot_grid
    
    
    def energy_tensor(self, gd0, mom0):
        ''' Energy functional for tensor input
            (to compute the automatic gradient the input is needed as tensor, not as list) '''
        
        self.h.module.manifold.fill_gd(gd0)
        self.h.module.manifold.fill_cotan(mom0)
        self.h.geodesic_controls()

        cost = self.cost()
        self.shoot()
                     
        attach = self.attach()
        print('cost:', self.gamma * cost.detach().numpy(), 'attach:', attach.detach().numpy())
        
        return self.gamma*cost + attach
        
    
    def gradE_autograd(self, gd0_tensor, mom0_tensor):
        grad = torch.autograd.grad(self.energy_tensor(gd0_tensor, mom0_tensor), mom0_tensor)[0]
        return grad
    
    