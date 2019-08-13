import torch
import numpy as np
import defmod as dm

from .manifold import Landmarks, CompoundManifold
from .structuredfield import StructuredField_multi
from .deformationmodules import CompoundModule, SilentPoints, Translations
from .multimodule_usefulfunctions import block_diag


class MultiShapeModule(torch.nn.Module):
    """ Input: List of compound modules, one compound module per shape, 
               the first module of each compound is a silent module with boundary points as gd
               List of functions taking points as input, that check if the module acts on the point.
               sigma_backgroud
        Creates background module with sigma_background"""
    def __init__(self, module_list, sigma_background, reduce_background=False, silentpoints=None):
        super().__init__()
        self.__nb_shapes = len(module_list)
        self.__sigma_background = sigma_background
        self.__reduce_background = reduce_background
        self.__module_list = [mod.copy() for mod in module_list]
        self.__silent_list = [mod.module_list[0].copy(retain_grad=True) for mod in module_list]
        
        if reduce_background:
            print('reduced')
            self.__background = dm.deformationmodules.Background_reduced(self.__silent_list, self.__sigma_background, boundary_labels)
            print(self.__silent_list)
            print('red background gd',self.__background.manifold.gd)
        else:
            self.__background = dm.deformationmodules.Background(self.__silent_list, self.__sigma_background)
            
            
        self.__module_list = [*self.__module_list, self.__background]
        
        if silentpoints is not None:
            man_silent = Landmarks(dim=self.__module_list[0].manifold.dim, nb_pts=len(silentpoints), gd=silentpoints.view(-1).requires_grad_())
            mod_silent = SilentPoints(man_silent)
            mod_list = []
            for mod in self.__module_list[:-1]:
                mod_list.append(CompoundModule([ *mod.module_list, mod_silent.copy()]))
            mod_list.append(CompoundModule([self.__background, mod_silent.copy()]))    
            self.__module_list = mod_list
                                     
        
        self.__manifold_list = [m.manifold for m in self.__module_list]
        self.__manifold = CompoundManifold(self.__manifold_list)
        
        
        self.__background.fill_controls_zero()
        
    def copy(self):        
        mod_copy = MultiShapeModule([mod.copy() for mod in self.__module_list[:-1]], self.__sigma_background, reduce_background = self.__reduce_background)
        mod_copy.module_list[-1] = self.__background.copy()
        return mod_copy

    @property
    def module_list(self):
        return self.__module_list
    
    @property
    def silent_list(self):
        return self.__silent_list
    
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
    def reduced_gd(self):
        return self.__reduce_background
        

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
    
    def fill_controls(self, controls, copy=False, retain_grad=False):
        #assert len(controls) == self.nb_module +1
        if copy:
            for i in range(self.nb_module):
                self.__module_list[i].__controls = controls[i].clone().detach().requires_grad_()
            if retain_grad:
                self.__module_list[i].__controls = controls[i].clone().requires_grad_()        
        else:
            for i in range(self.nb_module):
                self.__module_list[i].fill_controls(controls[i])
            
    controls = property(__get_controls, fill_controls)

    def fill_controls_zero(self):
        for m in self.__module_list:
            m.fill_controls_zero()         
            
    def __get_l(self):
        return self.__l
    
    def fill_l(self, l, copy=False):
        #assert len(l) == self.nb_module
        if copy:
            self.__l = l.detach().clone().requires_grad_()
        else:
            self.__l = l
        
    l = property(__get_l, fill_l)
            

    @property
    def manifold(self):
        return self.__manifold
    

    def __call__(self, points) :
        """Applies the generated vector field on given points."""
        ## TODO: apply field generator corresponding to label of point
        vs = self.field_generator()
        
        raise NotImplementedError
    
    
    def field_generator(self):
        """ generates vector fields depending on region of the points"""
        fields = []
        for m in self.__module_list:
            fields = [*fields, m.field_generator()]
        #return fields
        return StructuredField_multi(fields)
    
    
    def cost(self):
        return sum([m.cost() for m in self.__module_list])
    
    
    def autoaction(self):
        """ """
        n = self.module_list[0].manifold.numel_gd
        A = self.__module_list[0].autoaction()
        for m in self.__module_list[1:]:
            ni = m.manifold.numel_gd
            A = torch.cat([torch.cat([A, torch.zeros(n, ni)], 1), torch.cat([torch.zeros(ni, n), m.autoaction()], 1)], 0)
            n = n+ni
        return A
    
    def autoaction_silent(self):
        """ """
        n = self.module_list[0].manifold.manifold_list[0].numel_gd
        A = self.__module_list[0].autoaction_silent()
        
        for m in self.__module_list[1:-1]:
            ni = m.manifold.manifold_list[0].numel_gd
            A = torch.cat([torch.cat([A, torch.zeros(n, ni)], 1), torch.cat([torch.zeros(ni, n), m.autoaction_silent()], 1)], 0)
            n = n+ni
        ni = self.module_list[-1].manifold.numel_gd
        A = torch.cat([torch.cat([A, torch.zeros(n, ni)], 1), torch.cat([torch.zeros(ni, n), self.module_list[-1].autoaction_silent()], 1)], 0)
        return A
    
    def autoaction_blockdiag(self):
        ''' building autoaction matrix by blockdiagonal'''
        blocklist = [m.autoaction() for m in self.__module_list]
        return block_diag(blocklist)
            
    
    
    def compute_geodesic_variables(self, constr):
        
        self.compute_geodesic_control(self.manifold)

        fields = self.field_generator().fieldlist
        constr_mat = constr.constraintsmatrix(self)
        
        gd_action = torch.cat([*[torch.cat(man.manifold_list[0].action(mod).unroll_tan()) for mod,man in zip(self.module_list[:-1], self.manifold)], 
                                torch.cat(self.manifold.manifold_list[-1].action(self.module_list[-1]).unroll_tan())]).view(-1, self.manifold.dim)

        B = torch.mm(constr_mat, gd_action.view(-1,1))
        A = torch.mm(torch.mm(constr_mat, self.autoaction_silent()), torch.transpose(constr_mat,0,1))
            
        lambda_qp,_ = torch.gesv(B, A)
        self.fill_l(lambda_qp)

        tmp = torch.mm(torch.transpose(constr_mat,0,1), lambda_qp)
        
        # build compound manifold of all silent points (that belong to the boundary constraints) and the background manifold
        man_list = [man[0].copy(retain_grad=True) for man in self.manifold.manifold_list[:-1]]
        man = dm.manifold.CompoundManifold([*man_list, self.background.manifold.copy(retain_grad=True)])
        
        c = 0
        for m in man.manifold_list[:-1]:
            m.cotan = tmp[c:c+m.numel_gd].view(-1)
            c = c+m.numel_gd
        for m in man.manifold_list[-1].manifold_list:
            m.cotan = tmp[c:c+m.numel_gd].view(-1)
            c = c+m.numel_gd
        
        #self.compute_geodesic_control(man)
        
        for mod, m in zip(self.module_list, man.manifold_list):
            mod.compute_geodesic_control(m)
            h1 = mod.controls
            mod.compute_geodesic_control(mod.manifold)
            h2 = mod.controls
            print('h1', h1)
            print('h2', h2)
            print('================')
            mod.fill_controls([h1[i]- h2[i] for i in range(len(h1))])
        
        h_qp = self.controls
                                        
        return lambda_qp, h_qp
      
    
    def compute_geodesic_control(self, manifold): 
        for mod, man in zip(self.__module_list, manifold):
            mod.compute_geodesic_control(man)
        self.fill_controls([m.controls for m in self.__module_list])
        
            
    def compute_geodesic_control_from_self(self, manifold):
        # TODO: check manifold and self.manifold have the same type
        for m, man in zip(self.__module_list, manifold.manifold_list):            
            m.compute_geodesic_control_from_self(man)


class MultiShapeLDDMM(MultiShapeModule):
    ''' creates a multishape module that uses only local translations, corresponding to the multishape LDDMM framework '''
    def __init__(self, man_list, sigma_list):
        module_list = [CompoundModule([Translations(man, sigma)]) for man, sigma in zip(man_list, sigma_list[:-1]) ]
        super().__init__(module_list, sigma_background = sigma_list[-1], reduce_background=False)