import math
pi = math.pi
import torch
import defmod as dm
import defmod.hamiltonian_multishape as hamiltonian

from .multimodule_usefulfunctions import kronecker_I2, is_inside_shape
from .usefulfunctions import grid2vec


class RegistrationData:
    """ Builds different kind of datasets for matching"""
    def __init__(self):
        super().__init__()

        
class PointCircles(RegistrationData):
    """ Builds source and target points for translation and scaling of circles """
    def __init__(self, nb_pts, origin1, radius1, transvec, scal, dim = 2):
        super().__init__()       
        
        self.__nb_pts = nb_pts
        self.__origin1 = origin1
        self.__radius1 = radius1
        self.__transvec = transvec
        self.__scal = scal
        self.__dim = 2
        self.__source = None
        self.__target = None
        self.__modules = None
        
    @property
    def source(self):
        return self.__source
    
    @property
    def target(self):
        return self.__target
    
    @property
    def modules(self):
        return self.__modules

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
    
    def build_source(self):
        origin1 = torch.tensor(self.__origin1)
        self.__source = self.multipleCircles(self.__origin1, self.__radius1, self.__nb_pts)

    def build_target(self):
        origin1 = torch.tensor(self.__origin1)
        origin2 = [o + v for o,v in zip(origin1, self.__transvec)]
        radius2 = [r*s for r,s in zip(self.__radius1,self.__scal)]
        self.__target = self.multipleCircles(origin2, radius2, self.__nb_pts)
        
    def build_modules(self):
        self.build_source()
        self.build_target()
                
        manifold1 = dm.manifold.Landmarks(self.__dim, self.__nb_pts[0], gd=self.__source[0].view(-1))
        manifold2 = dm.manifold.Landmarks(self.__dim, self.__nb_pts[1], gd=self.__source[1].view(-1))

        silent1 = dm.deformationmodules.SilentPoints(manifold1)
        trans1 = dm.deformationmodules.GlobalTranslation(manifold1, sigma=20)
        trans1.fill_controls_zero()
        scal1 = dm.deformationmodules.GlobalScaling(manifold1, sigma=20)
        mod1 = dm.deformationmodules.CompoundModule([silent1, trans1, scal1])
        
        silent2 = dm.deformationmodules.SilentPoints(manifold2)
        trans2 = dm.deformationmodules.GlobalTranslation(manifold2, sigma=20)
        trans2.fill_controls_zero()
        scal2 = dm.deformationmodules.GlobalScaling(manifold2,sigma=20)
        mod2 = dm.deformationmodules.CompoundModule([silent2, trans2, scal2])
        
        self.__modules = [mod1, mod2]
    
    def __call__(self, nb_pts, origin1, radius1, transvec, scal=1):
        origin1 = torch.tensor(origin1)
        origin2 = [o + torch.tensor(v) for o,v in zip(origin1, transvec)]
        radius2 = radius1 * scal
        self.__source = self.multipleCircles(origin1, radius1, nb_pts)
        self.__target = self.multipleCircles(origin2, radius2, nb_pts)
        
    def grid_label(self, xmin, xmax, ymin, ymax, dx, dy):
        """ returns grid and labels for each gridpoint """
        x, y = torch.meshgrid([torch.arange(xmin, xmax, dx), torch.arange(ymin, ymax, dy)])
        nx, ny = x.shape[0], x.shape[1] 

        gridpoints = dm.usefulfunctions.grid2vec(x, y).type(torch.DoubleTensor)

        label = torch.zeros(len(gridpoints))
        for i in range(len(gridpoints)): 
            p = gridpoints[i,:]
            if (torch.norm(p-torch.tensor(self.__origin1[0]))) < self.__radius1[0]:
                label[i] = 1
            elif (torch.norm(p-torch.tensor(self.__origin1[1]))) < self.__radius1[1]:
                label[i] = 2
            else:
                label[i]=-1
                
        return x,y, label.view(x.shape)
        
class part_rigid(RegistrationData):
    def __init__(self, source_vertices, target_vertices, nb_pts, width=0.1, dim=2):
        super().__init__()
        self.__source = None
        self.__target = None
        self.__modules = None
        self.__dim = dim
        self.__nb_pts = nb_pts
        
        self.__source_vertices = source_vertices
        self.__target_vertices = target_vertices
        
    @property
    def source(self):
        return self.__source
    
    @property
    def target(self):
        return self.__target
    
    @property
    def modules(self):
        return self.__modules
    
    def compute_edges(self, v):
        a = v[1,:] - v[0,:]
        tmp = (a/torch.norm(a)).view(1,2)
        b = torch.cat([tmp[:,1], -tmp[:,0]])
        x = torch.cat([v[0,:] + b,
                       v[0,:] - b,
                       v[1,:] + b,
                       v[1,:] - b],0).view(-1,2)
        a = v[1,:] - v[2,:]
        tmp = (a/torch.norm(a)).view(1,2)
        b = torch.cat([tmp[:,1], -tmp[:,0]])
        y = torch.cat([v[1,:] - b,
                       v[1,:] + b,
                       v[2,:] - b,
                       v[2,:] + b],0).view(-1,2)
        q = []
        for i in range(2):
            x1 = x[i::2,:]
            y1 = y[i::2,:]
            p = torch.cat([x[i::2,:], y[i::2,:]],0).view(-1,2)
            A = torch.transpose(torch.mm(torch.tensor([[-1.,1., 0.,0.], [0.,0.,1.,-1.]]),p),0,1)
            B = torch.mm(torch.tensor([[-1.,0.,1.,0.]]),p).view(2,1)
            c,_ = torch.gesv(B, A)
            q.append((x1[0,:]+c[0,0]*(x1[1,:]-x1[0,:])).view(1,2))
                            
        return torch.cat([q[0], x[0:2,:].view(-1,2),q[1],q[0]],0) ,torch.cat([q[0], y[2:,:].view(-1,2), q[1],q[0]],0), q
    
    def points_on_line(self,a,b, n):
        points = []
        k = torch.tensor(range(n)).view(-1,1).double()
        s = (b - a).view(-1,self.__dim).double()
        return 1/(n-1)*s*k+a*torch.ones(k.shape)
    
    def flip(self, x, dim):
        dim = x.dim() + dim if dim < 0 else dim
        inds = tuple(slice(None, None) if i != dim
                 else x.new(torch.arange(x.size(i)-1, -1, -1).tolist()).long()
                 for i in range(x.dim()))
        return x[inds]

    def build_source(self):
        self.__source = []
        v1, v2,_ = self.compute_edges(self.__source_vertices)
        nb_pts = [self.__nb_pts[0], self.__nb_pts[1], self.__nb_pts[0], self.__nb_pts[1]]
        
        for v in [v1,v2]:
            source = torch.zeros([0,2],requires_grad=True)
            for i in range(len(v1)-1):
                source = torch.cat([source,self.points_on_line(v[i,:],v[i+1,:],nb_pts[i])[:-1]],0)
            self.__source.append(source)
            
        # for points_in_shape function
        self.__source[0] = self.flip(self.__source[0], 0)
      
    def build_target(self):
        self.__target = []
        v1, v2,_ = self.compute_edges(self.__target_vertices)
        nb_pts = [self.__nb_pts[0], self.__nb_pts[1], self.__nb_pts[0], self.__nb_pts[1]]
        
        for v in [v1,v2]:
            target = torch.zeros([0,2],requires_grad=True)
            for i in range(len(v1)-1):
                target = torch.cat([target,self.points_on_line(v[i],v[i+1],nb_pts[i])[:-1]],0)
            self.__target.append(target)
            
            
    def build_shot_target(self):
        self.build_target()
        self.build_modules()
        
        modules = dm.multishape.MultiShapeModule([self.modules[0], self.modules[1]], sigma_background=0.5, reduce_background=True)
        modules.fill_controls_zero()
        constr_Id = dm.constraints.Identity_Silent_reduced()
        H = hamiltonian.Hamiltonian_multi(modules, constr_Id)

        modules.module_list[0].module_list[-1].manifold.cotan.view(-1,2)[5:9,:] = torch.tensor([0,5])
        modules.module_list[0].module_list[-1].manifold.gd.view(-1,2)

        states, controls = dm.shooting.shoot_euler(H, it=10)

        self.__target = [states[-1][0][-1].gd.view(-1,2), states[-1][1][-1].gd.view(-1,2)]
        
        # adding noise
        noise1 = torch.zeros_like(self.__target[0])
        noise1[3:-1,:] = 0.1 * torch.rand_like(self.__target[0][3:-1,:])
        self.__target[0] = self.__target[0]+ noise1
        noise2 = torch.zeros_like(self.__target[1])
        noise2[1:9,:] = 0.1 * torch.rand_like(self.__target[1][1:9,:])
        self.__target[1] = self.__target[1] + noise2
        
        self.__source = [states[0][0][-1].gd.view(-1,2), states[0][1][-1].gd.view(-1,2)] 
        
            
    def intersection(self):
        l=[]
        l1 = [self.source[0][i,:] for i in range(len(self.source[0]))]
        l2 = [self.source[1][i,:] for i in range(len(self.source[0]))]
        for x in l1:
            for y in l2:
                if torch.all(torch.eq(x,y)):
                    l.append(x.view(1,2))
        return torch.cat(l)

    def build_modules(self):
        self.__modules = []
        _,_,q = self.compute_edges(self.__source_vertices)
        gd_silent = self.intersection()
        man_silent = dm.manifold.Landmarks(2, self.__nb_pts[1], gd = gd_silent.view(-1))    

        #gd_silent = self.__source
        #man_silent = dm.manifold.Landmarks(2, 2*sum(self.__nb_pts), gd = gd_silent[i].view(-1))    

        
        for i in range(2):
            man = dm.manifold.Landmarks(2, len(self.__source[i]), gd=self.__source[i].view(-1))
            rot = dm.deformationmodules.GlobalRotation(man.copy(retain_grad=True), sigma=2., coeff=0.5)
            trans = dm.deformationmodules.GlobalTranslation(man.copy(retain_grad=True), sigma=20.,coeff=2.)
            trans1 = dm.deformationmodules.Translations(man_silent.copy(retain_grad=True), sigma=0.1, coeff=5.)
            trans.fill_controls_zero()
            silent = dm.deformationmodules.SilentPoints(man_silent.copy(retain_grad=True))
            silent_source = dm.deformationmodules.SilentPoints(dm.manifold.Landmarks(2, len(self.__source[i]), gd=self.__source[i].view(-1)))
            mod = dm.deformationmodules.CompoundModule([silent, trans, rot, trans1, silent_source])
            self.__modules.append(mod)
            
    def grid_label(self, xmin, xmax, ymin, ymax, dx, dy):
        """ returns grid and labels for each gridpoint """
        x, y = torch.meshgrid([torch.arange(xmin, xmax, dx), torch.arange(ymin, ymax, dy)])
        nx, ny = x.shape[0], x.shape[1] 

        gridpoints = dm.usefulfunctions.grid2vec(x, y).type(torch.DoubleTensor)

        label = torch.zeros(len(gridpoints))
        
        for shape, i in zip(self.__source, list(range(len(self.__source)+1))):
            print(i, torch.sum(is_inside_shape(shape, gridpoints)))
            
            label[is_inside_shape(shape, gridpoints) == 1] = i+1
        label[label==0] = -1
        return x, y, label.view(x.shape)
        
            
        
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
        

        
import pickle

class Nut(RegistrationData):
    def __init__(self, sigma_scaling=1.):
        super().__init__()
        self.__source = None
        self.__target = None
        self.__modules = None
        self.__sigma_scaling = sigma_scaling
        
    @property
    def source(self):
        return self.__source
    
    @property
    def target(self):
        return self.__target
    
    @property
    def modules(self):
        return self.__modules
        
    def close_loop(self, oc):
        cc = np.zeros((np.shape(oc)[0]+1,2))
        cc[0:-1,:] = oc
        cc[-1][:] = oc[0][:]
        return cc

    def build_modules(self):
        dty = torch.float64
        gd0 = torch.tensor([-1., 0.], requires_grad=True, dtype=dty)
        gd1 = torch.tensor([1., 0.], requires_grad=True, dtype=dty)
        
        nb_pts = self.__source[0].shape[0]
        man_silent = dm.manifold.Landmarks(2, nb_pts, gd=self.__source[0].view(-1).requires_grad_())
        man_scal1 = dm.manifold.Landmarks(2,1,gd=gd0)
        man_scal2 = dm.manifold.Landmarks(2,1,gd=gd1)
        
        silent = dm.deformationmodules.SilentPoints(man_silent)
        #scal1 = dm.deformationmodules.LocalScaling(man_scal1, sigma = self.__sigma_scaling)
        #scal2 = dm.deformationmodules.LocalScaling(man_scal2, sigma = self.__sigma_scaling)
        
        scal1 = dm.deformationmodules.ConstrainedTranslations_Scaling(man_scal1, sigma = self.__sigma_scaling)
        scal2 = dm.deformationmodules.ConstrainedTranslations_Scaling(man_scal2, sigma = self.__sigma_scaling)
        
        globaltrans = dm.deformationmodules.Translations(dm.manifold.Landmarks(2,1,gd = torch.tensor([0.,0.]).view(-1)), sigma=400, coeff=5.)
        trans = dm.deformationmodules.Translations(dm.manifold.Landmarks(2, len(self.__source[0]), gd=self.__source[0].view(-1).requires_grad_()), sigma=0.2, coeff=10.)

        comp1 = dm.deformationmodules.CompoundModule([silent, scal1, scal2, globaltrans, trans])
        
        self.__modules = [comp1]
                  
    def __call__(self):
        dty = torch.float64
        with open('../data/nuts/nutsdata.pickle', 'rb') as f:
            lines, sigv, sig = pickle.load(f)
            
        source = torch.tensor(lines[0][::2], requires_grad=True, dtype=dty)[1:]
        target = torch.tensor(lines[1][::2]  , requires_grad=True, dtype=dty)[1:]
        num_target = 1

        pts_source = source.detach().numpy()
        pts_target = target.detach().numpy()
        
        self.__source = [source]
        self.__target = [target]
        
        self.build_modules()
        
class Nut_translated(RegistrationData):
    def __init__(self, source_transvec=[0.,0.], target_transvec=[0.,0.], reflect_target=1.):
        super().__init__()
        self.__source = None
        self.__target = None
        self.__modules = None
        
        self.__reflect = reflect_target
        self.__source_transvec = source_transvec
        self.__target_transvec = target_transvec
        
    @property
    def source(self):
        return self.__source
    
    @property
    def target(self):
        return self.__target
    
    @property
    def modules(self):
        return self.__modules    
        
    def __call__(self):
        Nut1 = Nut()
        Nut1()

        self.__source = [Nut1.source[0] + torch.ones(len(Nut1.source[0]),1) * torch.tensor(self.__source_transvec).view(-1,2)]
        #self.__target = [Nut1.target[0] + torch.ones(len(Nut1.target[0]),1) * torch.tensor(self.__target_transvec).view(-1,2)]     
        self.__target = [Nut1.target[0]* torch.tensor([self.__reflect,1.]) + torch.ones(len(Nut1.target[0]),1) * torch.tensor(self.__target_transvec).view(-1,2)]
        
        self.build_modules()
        
    def build_modules(self):
        dty = torch.float64
        
        gd_scal1 = torch.tensor([-1., 0.], requires_grad=True, dtype=dty) + torch.tensor(self.__source_transvec).view(-1)
        gd_scal2 = torch.tensor([1., 0.], requires_grad=True, dtype=dty)  + torch.tensor(self.__source_transvec).view(-1)
        gd_trans = torch.tensor([0., 0.], requires_grad=True, dtype=dty) + torch.tensor(self.__source_transvec).view(-1)
        gd0 = [self.__source[0].clone().view(-1).requires_grad_(), gd_scal1, gd_scal2, gd_trans, self.__source[0].clone().view(-1).requires_grad_()]
    
        mod_list = []
        for gd in [gd0]:
            
            nb_pts = gd[0].view(-1,2).shape[0]
            man_silent = dm.manifold.Landmarks(2, nb_pts, gd=gd[0])
            man_scal1 = dm.manifold.Landmarks(2,1,gd=gd[1])
            man_scal2 = dm.manifold.Landmarks(2,1,gd=gd[2])

            silent = dm.deformationmodules.SilentPoints(man_silent)

            scal1 = dm.deformationmodules.ConstrainedTranslations_Scaling(man_scal1, sigma = 1.)
            scal2 = dm.deformationmodules.ConstrainedTranslations_Scaling(man_scal2, sigma = 1.)

            globaltrans = dm.deformationmodules.Translations(dm.manifold.Landmarks(2,1,gd=gd[3]), sigma=400, coeff=5.)
            trans = dm.deformationmodules.Translations(dm.manifold.Landmarks(2, len(self.__source[0]), gd=gd[4]), sigma=0.2, coeff=10.)

            comp = dm.deformationmodules.CompoundModule([silent, scal1, scal2, globaltrans, trans])

            mod_list.append(comp)

        self.__modules = mod_list
        
        
