import torch
import numpy as np
import defmod as dm

from .manifold import Landmarks, CompoundManifold


#from .kernels import K_xx, K_xy

class MultiShapeModule(torch.nn.Module):
    """ Input: List of modules, 
               List of functions taking points as input, that check if the module acts on the point.
               sigma_backgroud
        Creates background module with sigma_background"""
    def __init__(self, module_list, sigma_background=0.5):
        #assert isinstance(module_list)
        super().__init__()
        self.__sigma_background = sigma_background
        ##self.__points_in_region = [*points_in_region]
        self.__module_list = [mod.copy() for mod in module_list]
        self.__background = dm.deformationmodules.Background(self.__module_list, self.__sigma_background)
        self.__module_list = [*self.__module_list, self.__background]
        self.__manifold_list = [m.manifold for m in self.__module_list]
        self.__manifold = CompoundManifold(self.__manifold_list)
        
        self.__background.fill_controls_zero()
        
    def copy(self):
        return MultiShapeModule([mod.copy() for mod in self.__module_list], self.__sigma_background)

    @property
    def module_list(self):
        return self.__module_list
    
    @property
    def nb_pts(self):
        return [m.nb_pts for m in self.__manifold.manifold_list[:-1]]
    
    @property
    def numel_gd(self):
        return [m.numel_gd for m in self.__manifold.manifold_list]
    
    @property
    def background(self):    
        return self.__background
        
    @property
    def points_in_region(self):
        return self.__points_in_region

    def __getitem__(self, index):
        return self.__module_list[index]

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current >= len(self.__module_list):
            raise StopIteration
        else:
            self.current = self.current + 1
            return self.__module_list[self.current - 1]

    @property
    def nb_module(self):
        """ number of modules (without background)"""
        return len(self.__module_list) -1

    @property
    def dim_controls(self):
        return sum([mod.dim_controls for mod in self.__module_list])

    def __get_controls(self):
        controls = [m.controls for m in self.__module_list]
        return controls
    
    def get_controls_background(self):
        return self.__background.controls
    
  
    def fill_controls(self, controls):
        assert len(controls) == self.nb_module +1
        for i in range(self.nb_module):
            self.__module_list[i].fill_controls(controls[i])#.detach().clone().requires_grad_())
        self.__module_list[-1].fill_controls(controls[-1])#.copy())
            
    controls = property(__get_controls, fill_controls)

    def fill_controls_zero(self):
        #for m in self.__module_list:
        #    m.fill_controls_zero()  
        self.fill_controls([*[torch.zeros(m.dim_controls) for m in self.module_list[:-1]], [torch.zeros(m.dim_controls) for m in self.module_list[:-1]]])
            
    def __get_l(self):
        return self.__l
    
    def fill_l(self, l, copy=False):
        #assert len(l) == self.nb_module
        # used as dim: (nb_pts*dim x 1)
        if copy:
            self.__l = l.detach().clone().requires_grad_()
        else:
            self.__l = l
        
        
    l = property(__get_l, fill_l)
            

    @property
    def manifold(self):
        return self.__manifold
    
    #def boollist2tensor(boollist):
    #    return torch.tensor(np.asarray(boollist).astype(np.uint8))

    def __call__(self, points) :
        """Applies the generated vector field on given points."""
        vs = self.field_generator()
        return vs(points)
    
    
    def field_generator(self):
        """ generates vector fields depending on region of the points"""
        fields = []
        for m in self.__module_list:
            fields = [*fields, m.field_generator()]
        return fields
    
    
    def cost(self):
        return sum([m.cost() for m in self.__module_list])
    
    
    def autoaction(self):
        """ """
        n = self.manifold.gd[0].shape[0]
        A = self.__module_list[0].autoaction()
        for m in self.__module_list[1:]:
            ni = m.manifold.numel_gd
            A = torch.cat([torch.cat([A, torch.zeros(n, ni)], 1), torch.cat([torch.zeros(ni, n), m.autoaction()], 1)], 0)
            n = n+ni
        return A
    
    def compute_geodesic_variables(self, constr):

        self.compute_geodesic_control_from_self(self.manifold)
        fields = self.field_generator()
        constr_mat = constr.constraintsmatrix(self)
        
        gd_moved = [f(p) for f,p in zip(fields, dm.multimodule_usefulfunctions.gdlist_reshape(self.manifold.gd, [-1,self.__manifold.dim]))]
        B = torch.mm(constr_mat, torch.cat( [*gd_moved[:-1], *gd_moved[-1]]).view(-1,1))
        
        #trying to make it more general

        #gd_action = [man.action(mod).tan for mod,man in zip(self.module_list, self.manifold)]
        #gd2 = self.action_on_self().tan


        B1 = torch.mm(constr_mat, dm.multimodule_usefulfunctions.gdlist2tensor(gd_moved).view(-1,1))
      
        #print(gd_moved)
        #print(gd_action)
        #print(gd2)
        #print('____________')
        #############################
        
        A = torch.mm(torch.mm(constr_mat, self.autoaction()), torch.transpose(constr_mat,0,1))
                
        lambda_qp,_ = torch.gesv(B, A)
        self.fill_l(lambda_qp)
             
        p = torch.cat([*self.manifold.cotan[:-1], *self.manifold.cotan[-1]],0).view(-1,1) # cotan 2 
        cotan = dm.multimodule_usefulfunctions.gdtensor2list((p - torch.mm(torch.transpose(constr_mat,0,1), lambda_qp)).view(-1), self.numel_gd[:-1])
        
        man = self.manifold.copy()
        man.cotan = cotan
        self.compute_geodesic_control_from_self(man)
        h_qp = self.controls
                                        
        return lambda_qp, h_qp
      
    
    def compute_geodesic_control(self, manifold): 

        self.fill_controls([m.controls for m in self.__module_list])
        
            
    def compute_geodesic_control_from_self(self, manifold):
        # TODO: check manifold and self.manifold have the same type
        for m, man in zip(self.__module_list, manifold.manifold_list):            
            m.compute_geodesic_control_from_self(man)

            
         