import torch
import numpy as np

from kernels import K_xx, K_xy

class MultiShapeModule(torch.nn.Module, Iterable):
    """ Input: List of modules, 
               List of functions taking points as input, that check if the module acts on the point.
        Creates background module """
    def __init__(self, module_list, points_in_region):
        assert isinstance(module_list, Iterable)
        super().__init__()
        self.__module_list = [*module_list]
        self.__points_in_region = [*points_in_region]
        self.__points_in_background = lambda points: np.logical_not(np.logical_or(self.__points_in_region1(points),points_in_region2(points)))
        
    def __points_in_background(self, points):
        label = [False]*len(points)
        for i in range(len(self.__points_in_region)):
            label = np.logical_or(self.__points_in_region[i](points), label)
        return np.logical_not(label)

    @property
    def module_list(self):
        return self.__module_list
    
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
        return CompoundManifold([m.manifold for m in self.__module_list]) ## plus background!
    
    def boollist2tensor(boollist):
        return torch.tensor(np.asarray(boollist).astype(np.uint8))

    ## Check if this is working
    def __call__(self, points) :
        """Applies the generated vector field on given points."""
        app_list = []
        for m in self.__module_list:
            label = boollist2tensor(self.__points_in_region[m](points))
            
            tmp = torch.zeros([1, len(points), dim])
            tmp[:,label==0,:] = m(points(label==0))
            
            app_list.append(tmp)
            
        return sum(app_list).view(-1, self.manifold.dim)
    
    
    def compute_geodesic_control(self, manifold, l ):        
        for m in self.__module_list:            
            manifold.cotan = manifold.cotan - 'Cq \lambda'
            self.__controls = m.compute_geodesic_control(manifold)
            
            