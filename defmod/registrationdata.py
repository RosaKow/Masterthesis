import math
pi = math.pi
import torch
import defmod as dm


class RegistrationData:
    """ Builds different kind of datasets for matching"""
    def __init__(self):
        super().__init__()

        
class PointCircles(RegistrationData):
    """ Builds source and target points for translation and scaling of circles """
    def __init__(self):
        super().__init__()       
        
        self.__source = None
        self.__target = None
        
    @property
    def source(self):
        return self.__source
    
    @property
    def target(self):
        return self.__target

    def CirclePoints(self, origin, r,n):
        points = []
        for x in range (0,n):
            points.append([origin[0] + math.cos(2*pi/n*x)*r,origin[1] + math.sin(2*pi/n*x)*r])
        return torch.tensor(points, requires_grad=True)

    def multipleCircles(self, origin, radius, numberPoints):
        circles = []
        for o,r,n in zip(origin, radius, numberPoints):
            circles.append(self.CirclePoints(o, r, n))
        return circles
    
    def __call__(self, nb_pts, origin1, radius1, transvec, scal=1):
        origin1 = torch.tensor(origin1)
        origin2 = [o + torch.tensor(v) for o,v in zip(origin1, transvec)]
        radius2 = radius1 * scal
        self.__source = self.multipleCircles(origin1, radius1, nb_pts)
        self.__target = self.multipleCircles(origin2, radius2, nb_pts)

        
class organs(RegistrationData):
    def __init__(self):
        super().__init__()
        self.__source = None
        self.__target = None
        self.__modules = None
        self.__dim = 2
        self.__nb_pts = 30
        
    @property
    def source(self):
        return self.__source
    
    @property
    def target(self):
        return self.__target
    
    @property
    def modules(self):
        return self.__modules
        
    def EllipsePoints(self, origin, a, b, n):
        points = []
        for x in range(0,n):
            points.append([origin[0] + math.cos(2*pi*x/n)*a, origin[1] + math.sin(2*pi*x/n)*b])
        return torch.tensor(points, requires_grad=True)
    
    def build_modules(self):
        self.build_source()
        
        trans_local = dm.deformationmodules.Translations(dm.manifold.Landmarks(self.__dim, 1, gd = torch.tensor([0.,6.], requires_grad=True)), sigma=10)
        rot = dm.deformationmodules.LocalRotation(dm.manifold.Landmarks(self.__dim, 1), sigma=5)
        scal = dm.deformationmodules.LocalScaling(dm.manifold.Landmarks(self.__dim,1), sigma=1.2)
        
        rot.fill_controls(torch.tensor([5.]))
        rot_center = torch.tensor([-8.,-1.], requires_grad=True).view(-1)
        rot.manifold.fill_gd(rot_center)
        silentpoints1 = self.source[0].view(-1)
        silentpoints1.requires_grad
        silent1 = dm.deformationmodules.SilentPoints(dm.manifold.Landmarks(self.__dim, self.__nb_pts, gd = silentpoints1))
        mod1 = dm.deformationmodules.CompoundModule([silent1, rot])
        
        scal_center = torch.tensor([-7.2,-7.])
        scal.fill_controls(torch.tensor([4.], requires_grad=True))
        scal.manifold.fill_gd(scal_center)
        trans_global = dm.deformationmodules.GlobalTranslation(dm.manifold.Landmarks(self.__dim, self.__nb_pts), sigma=10)
        trans_global.fill_controls(torch.tensor([1.5, 1.]))
        trans_global.manifold.fill_gd(self.__source[1].view(-1))
        silentpoints2 = self.source[1].view(-1)
        silentpoints2.requires_grad
        silent2 = dm.deformationmodules.SilentPoints(dm.manifold.Landmarks(self.__dim, self.__nb_pts, gd = silentpoints2))
        mod2 = dm.deformationmodules.CompoundModule([silent2, scal, trans_global])
        
        self.__modules = [mod1, mod2]
    
    def build_source(self):
        origin = [0.,0.]
        a = 2.
        b = 8.
        X = self.EllipsePoints(origin, a, b, self.__nb_pts).detach()
        Y = self.EllipsePoints(origin, a, b/2, self.__nb_pts).detach()
        
        trans_global1 = dm.deformationmodules.GlobalTranslation(dm.manifold.Landmarks(self.__dim, self.__nb_pts), sigma=10)
        trans_global1.fill_controls(torch.tensor([-15.,0.]))
        trans_global1.manifold.fill_gd(X.view(-1))
        trans_local = dm.deformationmodules.Translations(dm.manifold.Landmarks(self.__dim, 1, gd = torch.tensor([0.,6.], requires_grad=True)), sigma=10)
        trans_local.fill_controls(torch.tensor([3.,-5.]))
        mod = dm.deformationmodules.CompoundModule([trans_global1, trans_local])
        v = mod.field_generator()(X)
        source = X + v
        
        trans_global2 = dm.deformationmodules.GlobalTranslation(dm.manifold.Landmarks(self.__dim, self.__nb_pts), 50)
        trans_global2.fill_controls(torch.tensor([-7.,-5.]))
        trans_global2.manifold.fill_gd(Y.view(-1))
        mod = dm.deformationmodules.CompoundModule([trans_global2])
        v = mod.field_generator()(Y)
        source2 = Y + v
        
        self.__source = [source, source2]
        
    def build_target(self):
        self.build_modules()
        v = self.__modules[0].field_generator()(self.__source[0])
        target = self.__source[0] + v
        
        v = self.__modules[1].field_generator()(self.__source[1])
        target2 = self.__source[1] + v

        self.__target = [target, target2]
    
    def __call__(self):
        self.build_target()
        

        