import copy
from collections import Iterable

import torch
import numpy as np

from .structuredfield import StructuredField_Null, StructuredField_0, CompoundStructuredField
from .kernels import gauss_kernel, K_xx, K_xy, compute_sks
from .manifold import Landmarks, CompoundManifold

from .usefulfunctions import make_grad_graph

from .multimodule_usefulfunctions import kronecker_I2, CirclePoints



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


class CompoundModule(DeformationModule, Iterable):
    """Combination of modules."""
    def __init__(self, module_list):
        assert isinstance(module_list, Iterable)
        super().__init__()
        self.__module_list = [*module_list]

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

    def compute_geodesic_control_from_self(self):
        """ Computes geodesic control on self.manifold"""
        self.compute_geodesic_control(self.__manifold)
            
    def field_generator(self):
        return CompoundStructuredField([m.field_generator() for m in self.__module_list])
    
    
class Background(DeformationModule):
    """ Creates the background module for the multishape framework"""
    def __init__(self, module_list, sigma, boundary_labels=None):
        import copy
        super().__init__()
        
        self.__module_list = [mod.copy() for mod in module_list]
        self.__boundary_labels = boundary_labels

        if (boundary_labels==None):
            self.__manifold = CompoundManifold([m.manifold.copy(retain_grad=True) for m in self.__module_list]) 
        
        else:### for boundary labels:
            man_list = []
            dim = module_list[0].manifold.dim
            for mod, label in zip(module_list, boundary_labels):
                if isinstance(mod.manifold, Landmarks):
                    gd = mod.manifold.gd.view(-1,2)[np.where(label==1)[0].tolist(),:]
                    man_list.append(Landmarks(dim, len(gd), gd.view(-1)))
                else:
                    raise NotImplementedError
            self.__manifold = CompoundManifold(man_list)
            
        self.__controls = [ torch.zeros(man.gd.shape) for man in self.__manifold.manifold_list ] 
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
            return sum([mod.manifold.dim_gd for mod in self.__module_list])
        else: 
            return sum([np.sum(labels) for labels in self.__boundary_labels])
    
    @property 
    def dim(self):
        return self.__module_list[0].manifold.dim

    def __get_controls(self):
        return self.__controls

    def fill_controls(self, controls, copy=False):
        assert len(controls) == self.nb_module
        for i in range(len(controls)):
            if self.__boundary_labels == None:
                assert controls[i].shape == self.__module_list[i].manifold.dim_gd  
            else: 
                print('controls shape', controls[i].shape)
                print('boundary', np.sum(self.__boundary_labels[i]))
                assert controls[i].shape == self.manifold.dim * np.sum(self.__boundary_labels[i])
        if copy:
            for i in range(self.nb_module):
                self.__controls[i] = controls[i].clone().detach().requires_grad_()
        else:
            for i in range(self.nb_module):
                self.__controls = controls
        
    def fill_controls_zero(self):
        self.__controls = [ torch.zeros(mod.manifold.gd.shape) for mod in self.__module_list ] 
            
    controls = property(__get_controls, fill_controls)
    

    def __call__(self, points):
        vs = self.field_generator()
        return vs(points)
    
    def K_q(self):
        """ Kernelmatrix which is used for cost and autoaction"""
        return K_xx(torch.cat(self.manifold.gd).view(-1, self.dim), self.__sigma)
    
    def cost(self):
        """Returns the cost."""
        cont = torch.cat(self.__controls,0)
        K_q = self.K_q()
        m = torch.mm(K_q, cont.view(-1, self.dim))
        cost = 0.5*torch.dot(m.view(-1), cont.view(-1))
        return cost
     
        
    def field_generator(self):
        man = self.manifold.copy()
        man.fill_gd(self.manifold.gd)
        for i in range(len(self.module_list)):
            man[i].fill_cotan(self.__controls[i].view(-1))
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
        return self.manifold.dim

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
        self.__translationmodule.compute_geodesic_control(man) 
        self.__controls = self.__translationmodule.controls

    def compute_geodesic_control_from_self(self, manifold):
        """ Computes geodesic control on self.manifold"""
        # TODO: check manifold has the same type as self.manifold
        self.compute_geodesic_control(manifold)
        
    def field_generator(self):
        return self.__translationmodule.field_generator()
    
    ## check if working
    def autoaction(self):
        """ computes matrix for autoaction = xi zeta Z^-1 zeta^\ast xi^\ast """

        K = K_xy(torch.cat(self.manifold.gd).view(-1, self.dim), self.z, self.__sigma)
        return torch.mm(torch.transpose(K,0),K)

    
class Scaling(DeformationModule):
    """ local scaling """
    def __init__(self, manifold, sigma):
        super().__init__()
        self.__sigma = sigma
        self.__dim = manifold.dim
        self.__manifold = manifold
        self.__scalvec = CirclePoints([0,0], 1, 3)
        self.__controls = torch.tensor(1., requires_grad=True)

        
    @property
    def scal_vec(self):
        return self.__scalvec
    
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
    
    def z(self):
        ''' Computes the center (mean) of gd'''
        gd = self.manifold.gd
        if len(gd.shape) == 1:
            gd = gd.unsqueeze(0)
        return torch.mean(gd.view(-1,self.manifold.dim),0).view(1,self.manifold.dim)
 
    def cost(self) :
        """Returns the cost."""
        # return self.__controls
        raise NotImplementedError

    def compute_geodesic_control(self, man):
        """Computes geodesic control from StructuredField."""
        raise NotImplementedError

    def compute_geodesic_control_from_self(self, manifold):
        """ Computes geodesic control on self.manifold"""
        # TODO: check manifold has the same type as self.manifold
        raise NotImplementedError
        
    def field_generator(self):
        """  """
        # Not working! why does it return zero?
        z = self.z()
        vectors = self.__controls * (self.__scalvec + z)
        support = z * torch.ones(vectors.shape)
        return StructuredField_0(support.view(-1, self.__manifold.dim),
                                 vectors.view(-1, self.__manifold.dim), self.__sigma), support, vectors
    
    def autoaction(self):
        """ computes matrix for autoaction = xi zeta Z^-1 zeta^\ast xi^\ast """
        raise NotImplementedError