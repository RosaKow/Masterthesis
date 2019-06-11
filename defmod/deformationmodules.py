import copy
from collections import Iterable

import torch
import numpy as np

from .structuredfield import StructuredField_Null, StructuredField_0, CompoundStructuredField
from .kernels import gauss_kernel, K_xx, K_xy, compute_sks
from .manifold import Landmarks, CompoundManifold

from .usefulfunctions import make_grad_graph

from .multimodule_usefulfunctions import kronecker_I2, CirclePoints
import math




class DeformationModule:
    """Abstract module."""
    def __init__(self):
        super().__init__()

    def copy(self):
        return copy.copy(self)

    def __call__(self, gd, controls, points):
        """Applies the generated vector field on given points."""
        raise NotImplementedError

    def cost(self, gd, controls):
        """Returns the cost."""
        raise NotImplementedError


class Translations(DeformationModule):
    """Module generating sum of translations."""
    def __init__(self, manifold, sigma):
        #assert isinstance(manifold, Landmarks)
        super().__init__()
        self.__manifold = manifold
        self.__sigma = sigma
        self.__dim_controls = self.__manifold.dim*self.__manifold.nb_pts
        self.__controls = torch.zeros(self.__dim_controls, requires_grad=True)

    @classmethod
    def build_and_fill(cls, dim, nb_pts, sigma, gd=None, tan=None, cotan=None):
        """Builds the Translations deformation module from tensors."""
        return cls(Landmarks(dim, nb_pts, gd=gd, tan=tan, cotan=cotan), sigma)

    @property
    def manifold(self):
        return self.__manifold

    @property
    def sigma(self):
        return self.__sigma

    @property
    def dim_controls(self):
        return self.__dim_controls

    def __get_controls(self):
        return self.__controls

    def fill_controls(self, controls):
        self.__controls = controls

    controls = property(__get_controls, fill_controls)

    def fill_controls_zero(self):
        self.__controls = torch.zeros(self.__dim_controls)

    def __call__(self, points):
        #return self.field_generator().Apply(points)
        """Applies the generated vector field on given points."""
        K_q = K_xy(points, self.__manifold.gd.view(-1, self.__manifold.dim), self.__sigma)
        return torch.mm(K_q, self.__controls.view(-1, self.__manifold.dim))

    def cost(self):
        """Returns the cost."""
        K_q = K_xx(self.manifold.gd.view(-1, self.__manifold.dim), self.__sigma)
        m = torch.mm(K_q, self.__controls.view(-1, self.__manifold.dim))
        return 0.5*torch.dot(m.view(-1), self.__controls.view(-1))

    def compute_geodesic_control(self, man):
        """Computes geodesic control from StructuredField vs."""
        vs = self.adjoint(man)
        K_q = K_xx(self.manifold.gd.view(-1, self.__manifold.dim), self.__sigma)
        controls, _ = torch.gesv(vs(self.manifold.gd.view(-1, self.manifold.dim)), K_q)
        self.__controls = controls.contiguous().view(-1)
        
    def compute_geodesic_control_from_self(self, manifold):
        """ Computes geodesic control on manifold of same type as self.manifold"""
        # TODO check manifold has the same type as self.manifold
        self.compute_geodesic_control(manifold)

    def field_generator(self):
        return StructuredField_0(self.__manifold.gd.view(-1, self.__manifold.dim),
                                 self.__controls.view(-1, self.__manifold.dim), self.__sigma)

    def adjoint(self, manifold):
        return manifold.cot_to_vs(self.__sigma)

    def autoaction(self):
        """ computes matrix for autoaction = xi zeta Z^-1 zeta^\ast xi^\ast """
        ## Kernelmatrix K_qq
        return kronecker_I2(K_xx(self.manifold.gd.view(-1, self.__manifold.dim), self.__sigma))
    
    def costop_inv(self):
        return torch.eye(self.__dim_controls)

class SilentPoints(DeformationModule):
    """Module handling silent points."""
    def __init__(self, manifold):
        assert isinstance(manifold, Landmarks)
        super().__init__()
        self.__manifold = manifold

    @classmethod
    def build_from_points(cls, pts):
        """Builds the Translations deformation module from tensors."""
        return cls(Landmarks(pts.shape[1], pts.shape[0], gd=pts.view(-1)))

    @property
    def dim_controls(self):
        return 0

    @property
    def manifold(self):
        return self.__manifold
    

    def __get_controls(self):
        return torch.tensor([], requires_grad=True)

    def fill_controls(self, controls):
        pass

    controls = property(__get_controls, fill_controls)

    def fill_controls_zero(self):
        pass

    def __call__(self, points):
        """Applies the generated vector field on given points."""
        return torch.zeros_like(points)

    def cost(self):
        """Returns the cost."""
        return torch.tensor(0.)

    def compute_geodesic_control(self, man):
        """Computes geodesic control from StructuredField vs. For SilentPoints, does nothing."""
        pass
    
    def compute_geodesic_control_from_self(self, man):
        pass

    def field_generator(self):
        return StructuredField_Null()
    
    def adjoint(self, manifold):
        return StructuredField_Null()

    def costop_inv(self):
        return torch.tensor([])

class CompoundModule(DeformationModule, Iterable):
    """Combination of modules."""
    def __init__(self, module_list):
        assert isinstance(module_list, Iterable)
        super().__init__()
        self.__module_list = []
        self.numel_controls = 0
        for mod in module_list:
            if isinstance(mod, CompoundModule):
                self.__module_list.extend(mod.module_list)
                self.numel_controls = self.numel_controls + mod.numel_controls
            else:
                self.__module_list.append(mod)
                self.numel_controls = self.numel_controls + list(mod.controls.shape)[0]
            
    @property
    def module_list(self):
        return self.__module_list

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
        return len(self.__module_list)

    @property
    def dim_controls(self):
        return sum([mod.dim_controls for mod in self.__module_list])

    def __get_controls(self):
        return [m.controls for m in self.__module_list]

    def fill_controls(self, controls):
        assert len(controls) == self.nb_module
        for i in range(self.nb_module):
            self.__module_list[i].fill_controls(controls[i])

    controls = property(__get_controls, fill_controls)

    def fill_controls_zero(self):
        for m in self.__module_list:
            m.fill_controls_zero()

    @property
    def manifold(self):
        return CompoundManifold([m.manifold for m in self.__module_list])

    def __call__(self, points) :
        """Applies the generated vector field on given points."""
        app_list = []
        for m in self.__module_list:
            app_list.append(m(points))

        return sum(app_list).view(-1, self.manifold.dim)

    def cost(self):
        """Returns the cost."""
        cost_list = []
        for m in self.__module_list:
            cost_list.append(m.cost())

        return sum(cost_list)

    def compute_geodesic_control(self, man):
        """Computes geodesic control from \delta \in H^\ast."""
        for i in range(self.nb_module):
            self.__module_list[i].compute_geodesic_control(man)

    def compute_geodesic_control_from_self(self, manifold):
        """ Computes geodesic control on self.manifold"""
        self.compute_geodesic_control(manifold)
            
    def field_generator(self):
        return CompoundStructuredField([m.field_generator() for m in self.__module_list])
    
    def autoaction(self):
        """ only if cost operator = Id """
        actionmat = torch.zeros(self.manifold.numel_gd, self.numel_controls)
        tmp = 0
        for m in range(len(self.module_list)):
            for i in range(list(self.module_list[m].controls.shape)[0]):
                self.fill_controls_zero()
                c = self.controls
                c[m][i] = 1
                self.fill_controls(c)
                actionmat[:,tmp+i] = torch.cat(self.manifold.action(self.module_list[m]).unroll_tan())
            tmp = tmp + list(self.module_list[m].controls.shape)[0]
        #A = torch.mm(actionmat, torch.transpose(actionmat, 0,1))
        A = torch.mm(actionmat, torch.mm(self.costop_inv(), torch.transpose(actionmat, 0,1)))

        return A
    
    def autoaction_silent(self):
        actionmat = torch.zeros(self.manifold.manifold_list[0].numel_gd, self.numel_controls)
        tmp = 0
        for m in range(len(self.module_list)):
            for i in range(list(self.module_list[m].controls.shape)[0]):
                self.fill_controls_zero()
                c = self.controls
                c[m][i] = 1
                self.fill_controls(c)
                actionmat[:,tmp+i] = torch.cat(self.manifold.manifold_list[0].action(self.module_list[m]).unroll_tan())
            tmp = tmp + list(self.module_list[m].controls.shape)[0]
        #A = torch.mm(actionmat, torch.transpose(actionmat, 0,1))
        A = torch.mm(actionmat, torch.mm(self.costop_inv(), torch.transpose(actionmat, 0,1)))
        return A
    
    def costop_inv(self):
        # blockdiagonal matrix of inverse cost operators of each module
        Z = self.module_list[0].costop_inv()
        n = len(Z)
        for m in self.module_list[1:]:
            Zi = m.costop_inv()
            ni = len(Zi)
            Z = torch.cat([torch.cat([Z, torch.zeros(n, ni)], 1), torch.cat([torch.zeros(ni, n), Zi], 1)], 0)
            n = n + ni
        return Z
    
class Background(DeformationModule):
    """ Creates the background module for the multishape framework"""
    def __init__(self, module_list, sigma, boundary_labels=None):
        import copy
        super().__init__()
        
        self.__module_list = [mod.copy() for mod in module_list]
        self.__boundary_labels = boundary_labels

        if (boundary_labels==None):
            self.__manifold = CompoundManifold([m.manifold.copy(retain_grad=True) for m in self.__module_list]) 
        
        else:
            man_list = []
            dim = module_list[0].manifold.dim
            for mod, label in zip(module_list, boundary_labels):
                if isinstance(mod.manifold, Landmarks):
                    gd = mod.manifold.gd.view(-1,2)[np.where(label==1)[0].tolist(),:]
                    man_list.append(Landmarks(dim, len(gd), gd.view(-1)))
                #TODO:  elif isinstance(mod.manifold, CompoundModule):
                    
                else:
                    raise NotImplementedError
            self.__manifold = CompoundManifold(man_list)
            
        self.__controls = [ man.roll_gd([torch.zeros(x.shape) for x in man.unroll_gd()]) for man in self.__manifold.manifold_list ] 
        self.__sigma = sigma
        
    @property
    def module_list(self):
        return self.__module_list
    
    @property
    def nb_module(self):
        return len(self.__module_list)

    @property
    def manifold(self):
        return self.__manifold
    
    @property
    def dim_controls(self):
        if self.__boundary_labels==None:
            return sum([mod.manifold.numel_gd for mod in self.__module_list])
        else: 
            return sum([np.sum(labels) for labels in self.__boundary_labels])
    
    @property 
    def dim(self):
        return self.__module_list[0].manifold.dim

    def __get_controls(self):
        return self.__controls

    def fill_controls(self, controls, copy=False):
        assert len(controls) == self.nb_module
        #for i in range(len(controls)):
        #    if self.__boundary_labels == None:
        #        assert controls[i].shape == self.__module_list[i].manifold.dim_gd  
        #    else: 
        #        assert controls[i].shape == self.manifold.dim * np.sum(self.__boundary_labels[i])
        if copy:
            for i in range(self.nb_module):
                self.__controls[i] = controls[i].clone().detach().requires_grad_()
        else:
            for i in range(self.nb_module):
                self.__controls = controls
        
    def fill_controls_zero(self):
        self.__controls = [ man.roll_gd([torch.zeros(x.shape) for x in man.unroll_gd()]) for man in self.__manifold.manifold_list ] 
            
    controls = property(__get_controls, fill_controls)
    

    def __call__(self, points):
        vs = self.field_generator()
        return vs(points)
    
    def K_q(self):
        """ Kernelmatrix which is used for cost and autoaction"""
        return K_xx(torch.cat(self.manifold.unroll_gd()).view(-1, self.dim), self.__sigma)
    
    def cost(self):
        """Returns the cost."""
        cont = torch.cat(self.controls)
        K_q = self.K_q()
        m = torch.mm(K_q, cont.view(-1, self.dim))
        cost = 0.5*torch.dot(m.view(-1), cont.view(-1))
        return cost
             
    def field_generator(self):
        man = self.manifold.copy()
        man.fill_gd(self.manifold.gd)
        #for i in range(len(self.module_list)):
        #    man[i].fill_cotan(self.__controls[i].view(-1))
        man.fill_cotan(self.__controls)
        return man.cot_to_vs(self.__sigma)
    
    def compute_geodesic_control_from_self(self, manifold):
        """ assume man is of the same type and has the same gd as self.__man"""
        # TODO: check manifold and self.manifold have the same type
        self.fill_controls(manifold.cotan.copy())   
        
    def compute_geodesic_control(self, man):
        """ assume man is of the same type and has the same gd as self.__man"""
        raise NotImplementedError
        
    def autoaction(self):
        """ computes matrix for autoaction = xi zeta Z^-1 zeta^\ast xi^\ast """
        return kronecker_I2(self.K_q())
    
    def autoaction_silent(self):
        """ computes matrix for autoaction = xi zeta Z^-1 zeta^\ast xi^\ast """
        return kronecker_I2(self.K_q())
    
    def costop_inv(self):
        return self.K_q()
        
        
class GlobalTranslation(DeformationModule):
    ''' Global Translation Module for Multishapes
        Corresponds to a Translation Module where the translation is carried by the mean value of geometric descriptors'''
    def __init__(self, manifold, sigma):
        super().__init__()
        self.__sigma = sigma
        self.__dim = manifold.dim
        self.__manifold = manifold
        self.__manifold_trans = Landmarks(manifold.dim, 1)
        self.__translationmodule = Translations(self.__manifold_trans, sigma)
   
    @property
    def manifold(self):
        return self.__manifold

    @property
    def sigma(self):
        return self.__sigma
    
    @property
    def dim_controls(self):
        return self.__dim

    def __get_controls(self):
        return self.__controls

    def fill_controls(self, controls):
        self.__controls = controls
        self.__translationmodule.fill_controls(controls)

    controls = property(__get_controls, fill_controls)
    
    def fill_controls_zero(self):
        self.__controls = torch.zeros(self.dim_controls, requires_grad=True)
        self.__translationmodule.fill_controls_zero()
        
    
    def __call__(self, points) :
        """Applies the generated vector field on given points."""
        self.__translationmodule.manifold.fill_gd(self.z().view(-1))
        return self.__translationmodule(points)
    
    def z(self):
        ''' Computes the center (mean) of gd'''
        gd = self.manifold.gd
        if len(gd.shape) == 1:
            gd = gd.unsqueeze(0)
        return torch.mean(gd.view(-1,self.manifold.dim),0).view(1,self.manifold.dim)

    def cost(self) :
        """Returns the cost."""
        return self.__translationmodule.cost()

    def compute_geodesic_control(self, man):
        """Computes geodesic control from StructuredField."""
        self.__translationmodule.manifold.fill_gd(self.z().view(-1))
        self.__translationmodule.compute_geodesic_control(man) 
        self.__controls = self.__translationmodule.controls

    def compute_geodesic_control_from_self(self, manifold):
        """ Computes geodesic control on self.manifold"""
        # TODO: check manifold has the same type as self.manifold
        self.compute_geodesic_control(manifold)
        
    def field_generator(self):
        self.__translationmodule.manifold.fill_gd(self.z().view(-1))
        return self.__translationmodule.field_generator()
    
    def autoaction(self):
        """ computes matrix for autoaction = xi zeta Z^-1 zeta^\ast xi^\ast """
        K = K_xy(self.manifold.gd.view(-1, self.manifold.dim), self.z(), self.__sigma)
        return kronecker_I2(torch.mm(K, torch.transpose(K,0,1)))
    
    def costop_inv(self):
        return torch.eye(self.dim_controls)

    
class LocalConstraintTranslation(DeformationModule):
    """ local constraint translation module 
        f_support and f_vectors are functions that give the support and translationvectors from the gd """
    def __init__(self, manifold, sigma, f_support, f_vectors):
        super().__init__()
        self.__sigma = sigma
        self.__dim = manifold.dim
        self.__manifold = manifold
        self.__controls = torch.tensor([1.], requires_grad=True)
        self.__f_support = f_support
        self.__f_vectors = f_vectors
    
            
    @property
    def f_support(self):
        return self.__f_support
    
    @property
    def f_vectors(self):
        return self.__f_vectors
    @property
    def manifold(self):
        return self.__manifold

    @property
    def sigma(self):
        return self.__sigma
    
    @property
    def dim_controls(self):
        return 1

    def __get_controls(self):
        return self.__controls

    def fill_controls(self, control):
        self.__controls = control

    controls = property(__get_controls, fill_controls)
    
    def fill_controls_zero(self):
        self.__controls = torch.zeros(1, requires_grad=True)        
    
    def __call__(self, points) :
        """Applies the generated vector field on given points."""
        return self.field_generator()(points)
 
    def cost(self) :
        """Returns the cost."""
        return self.__controls *self.__controls

    def compute_geodesic_control(self, man):
        """Computes geodesic control from StructuredField."""
        self.fill_controls(1)
        
        v = self.field_generator()
        controls = man.inner_prod_field(v)
        #controls = torch.mm(self.field_generator()(man.gd.view(-1,self.manifold.dim)).view(1,-1), man.cotan.view(-1,1))
        self.__controls = controls.contiguous().view(-1)
        
    def compute_geodesic_control_from_self(self, manifold):
        """ Computes geodesic control on self.manifold"""
        # TODO: check manifold has the same type as self.manifold
        self.compute_geodesic_control(manifold)
        
    def field_generator(self):
        """  """
        vectors = self.__f_vectors(self.__manifold.gd)
        support = self.__f_support(self.__manifold.gd)
        return StructuredField_0(support.view(-1, self.__manifold.dim),
                                 self.controls * vectors.view(-1, self.__manifold.dim), self.__sigma)
    
    def autoaction(self):
        """ computes matrix for autoaction = xi zeta Z^-1 zeta^\ast xi^\ast """
        K_q = sum([K_xy(self.__support.view(-1, self.__manifold.dim)[i,:], self.manifold.gd.view(-1, self.__manifold.dim), self.__sigma) for i in range(len(self.__support))])
        
        return torch.mm(K_q.view(2,1), K_q.view(1,2))
    
    def costop_inv(self):
        return torch.eye(self.dim_controls)
        
class LocalScaling(LocalConstraintTranslation):
    def __init__(self, manifold, sigma):
        
        def f_vectors(gd):
            pi = math.pi
            return torch.tensor([[math.cos(2*pi/3*x), math.sin(2*pi/3*x)] for x in range(3)])
        def f_support(gd):
            return gd.repeat(3, 1) + sigma/3 * f_vectors(gd)
        
        super().__init__(manifold, sigma, f_support, f_vectors)
        
class LocalRotation(LocalConstraintTranslation):
    def __init__(self, manifold, sigma):
        
        def f_support(gd):
            pi = math.pi
            return gd.repeat(3, 1) + sigma/3 * torch.tensor([[math.cos(2*pi/3*x), math.sin(2*pi/3*x)] for x in range(3)])
        def f_vectors(gd):
            pi = math.pi
            return torch.tensor([[-math.sin(2*pi/3*(x)), math.cos(2*pi/3*(x))] for x in range(3)])
        
        super().__init__(manifold, sigma, f_support, f_vectors)
        
        
class GlobalConstraintTranslation(DeformationModule):
    def __init__(self, manifold, sigma, f_support, f_vectors):
        self.__manifold = manifold
        self.__sigma = sigma
        self.__controls = torch.tensor([1.], requires_grad=True)
        self.__f_support = f_support
        self.__f_vectors = f_vectors
        #self.__vectors = f_vectors(self.__manifold.gd)
        #self.__support = f_support(self.__manifold.gd)        
        
        man = Landmarks(manifold.dim, 1)
        gd = manifold.gd.view(-1,manifold.dim)
        man.fill_gd(torch.mean(gd.view(-1,manifold.dim),0).view(1,manifold.dim).view(-1))  

        self.__localTranslation = LocalConstraintTranslation(man, sigma, f_support, f_vectors)
        super().__init__()    
            
    @property
    def f_support(self):
        return self.__f_support
    
    @property
    def f_vectors(self):
        return self.__f_vectors
    @property
    def manifold(self):
        return self.__manifold

    @property
    def sigma(self):
        return self.__sigma
    
    @property
    def dim_controls(self):
        return 1

    def __get_controls(self):
        return self.__controls
    
    def fill_controls_zero(self):
        self.__controls = torch.zeros(1, requires_grad=True)   
        
        
    def fill_controls(self, controls):
        self.__controls = controls
        self.__localTranslation.fill_controls(controls)

    controls = property(__get_controls, fill_controls)
    
    def fill_controls_zero(self):
        self.__controls = torch.zeros(self.dim_controls, requires_grad=True)
        self.__localTranslation.fill_controls_zero()
    
    def __call__(self, points) :
        """Applies the generated vector field on given points."""
        return self.__localTranslation.field_generator()(points)
 
    def cost(self) :
        """Returns the cost."""
        return self.__localTranslation.cost()
    
    def compute_geodesic_control(self, man):
        """Computes geodesic control from StructuredField."""
        self.fill_controls(1)
        #controls = torch.mm(self.field_generator()(man.gd.view(-1,self.manifold.dim)).view(1,-1), man.cotan.view(-1,1))
        controls = man.inner_prod_field(self.field_generator())
        self.__controls = controls.contiguous().view(-1)
        
    def compute_geodesic_control_from_self(self, manifold):
        """ Computes geodesic control on self.manifold"""
        # TODO: check manifold has the same type as self.manifold
        self.compute_geodesic_control(manifold)
        
    def field_generator(self):
        """  """
        return self.__localTranslation.field_generator()
    
    def autoaction(self):
        """ computes matrix for autoaction = xi zeta Z^-1 zeta^\ast xi^\ast """
        K_q = sum([torch.mm( K_xy(self.manifold.gd.view(-1, self.__manifold.dim), self.__support.view(-1, self.__manifold.dim)[i,:].view(-1, self.__manifold.dim), self.__sigma), self.__vectors[i,:].view(1,-1)) for i in range(len(self.__support))], 0)
        
        return torch.mm(K_q.view(-1,1), K_q.view(1,-1))
    
    def costop_inv(self):
        return torch.eye(self.dim_controls)

    
class GlobalScaling(GlobalConstraintTranslation):
    def __init__(self, manifold, sigma):
        
        def f_vectors(gd):
            pi = math.pi
            gd = torch.mean(gd.view(-1,manifold.dim),0).view(1,manifold.dim).view(-1)
            return torch.tensor([[math.cos(2*pi/3*x), math.sin(2*pi/3*x)] for x in range(3)])
        def f_support(gd):
            gd = torch.mean(gd.view(-1,manifold.dim),0).view(1,manifold.dim).view(-1)
            return gd * torch.ones(3,manifold.dim) + sigma/3 * f_vectors(gd)
        
        super().__init__(manifold, sigma, f_support, f_vectors)
              
class GlobalRotation(GlobalConstraintTranslation):
    def __init__(self, manifold, sigma):
        
        def f_support(gd):
            pi = math.pi
            return gd.repeat(3, 1) + sigma/3 * torch.tensor([[math.cos(2*pi/3*x), math.sin(2*pi/3*x)] for x in range(3)])
        def f_vectors(gd):
            pi = math.pi
            return torch.tensor([[-math.sin(2*pi/3*(x)), math.cos(2*pi/3*(x))] for x in range(3)])
        super().__init__(manifold, sigma, f_support, f_vectors)