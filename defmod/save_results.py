from defmod.shooting import shoot_euler, shoot_euler_silent
from defmod.multimodule_usefulfunctions import point_labels, gridpoints, plot_grid, plot_MultiGrid
import matplotlib.pyplot as plt
import torch
import defmod as dm
from defmod.manifold import Landmarks
from defmod.deformationmodules import CompoundModule, SilentPoints
from defmod.multishape import MultiShapeModule


class Save_Results:

    def __init__(self, H, source, target, it=10, figsize=(5,5), dpi=100):
        """ Input: Hamiltonian
                    target: Target shape 
                    it: number of iterations for shooting
                    
            """
        self.__H = H
        self.__target = target
        self.__source = source
        self.__init_cotan = H.module.manifold.cotan
        self.__init_gd = H.module.manifold.gd
        
        self.__it = it
        self.__states, self.__controls = shoot_euler(H, it)     
        self.__H.module.manifold.fill_gd(self.__init_gd)
        self.__H.module.manifold.fill_cotan(self.__init_cotan)
        
        self.__gridpoints = None
        self.__grid = None
        self.__grid_params = None
        
        self.__figsize = figsize
        self.__dpi = dpi
        
        super().__init__()
          
    
    def gridpoints(self, xlims, ylims, d):
        """ Builds a grid specified by x-and y-limits and distance between gridpoints
            returns: x, y  (meshgrid); gridpoints (shape (-1,2)) """
        xmin, xmax = xlims
        ymin, ymax = ylims
        dx, dy = d
        nx = int((xmax-xmin)/dx)
        ny = int((ymax-ymin)/dy)
        x,y,gridpts = gridpoints(xmin, xmax, ymin, ymax, dx, dy)
        
        self.__grid_params = (xlims, ylims, d, [nx, ny])
        self.__gridpoints = gridpts
        self.__grid = [x,y]
        return x,y, gridpts
        
    
    def fig_states(self, show=False):
        """ Plots a separate figure with source, target and state for each state during shooting """
        fig_list = []
        
        for s in self.__states:  
            if isinstance(s.gd[0], torch.Tensor):
                s = [s]
            fig_shooting = plt.figure(figsize = self.__figsize, dpi=self.__dpi)
            for i in range(len(list(s))):
                plt.scatter(s[i][0].gd.view(-1,2)[:, 0].detach().numpy(), s[i][0].gd.view(-1,2)[:, 1].detach().numpy(), c='r')

            for i in range(len(self.__source)):
                plt.plot(self.__target[i][:, 0].detach().numpy(), self.__target[i][:, 1].detach().numpy(), 'xk')
                plt.plot(self.__source[i][:, 0].detach().numpy(), self.__source[i][:, 1].detach().numpy(), '.k')
            
            fig_list.append(fig_shooting)
        
        if show == True:
            plt.axis('equal')
            plt.show()
        return fig_list
                      
            
    def save_all(self):
        self.__gridpoints(xlim, ylim, d)
        
        
        
class Save_Results_MultiShape(Save_Results):
    def __init__(self, H, source, target, it=10, figsize=(5,5), dpi=100):
        self.__H = H
        self.__target = target
        self.__source = source
        self.__init_cotan = H.module.manifold.cotan
        self.__init_gd = H.module.manifold.gd
        
        self.__it = it
        self.__states, self.__controls = shoot_euler(H, it)     
        self.__H.module.manifold.fill_gd(self.__init_gd)
        self.__H.module.manifold.fill_cotan(self.__init_cotan)
        
        self.__gridpoints = None
        self.__grid = None
        self.__grid_params = None
        
        self.__figsize = figsize
        self.__dpi = dpi
        super().__init__(H, source, target, it=10, figsize=(5,5), dpi=100)
    
    
    def gridpoints(self, xlims, ylims, d):
        """ Builds a grid specified by x-and y-limits and distance between gridpoints
            returns: x, y  (meshgrid); gridpoints (shape (-1,2)) """
        xmin, xmax = xlims
        ymin, ymax = ylims
        dx, dy = d
        nx = int((xmax-xmin)/dx)
        ny = int((ymax-ymin)/dy)
        x,y,gridpts = gridpoints(xmin, xmax, ymin, ymax, dx, dy)
        
        self.__grid_params = (xlims, ylims, d, [nx, ny])
        self.__gridpoints = gridpts
        self.__grid = [x,y]
        return x,y, gridpts
    
    def shoot_grid(self, gridpoints):
        self.__H.module.manifold.fill_gd(self.__init_gd)
        self.__H.module.manifold.fill_cotan(self.__init_cotan)
        grid_states, grid_controls, grid_intermediate = dm.shooting.shoot_euler_silent(self.__H, gridpoints, self.__it)

        grid_final = grid_intermediate[-1]
        return grid_final, grid_intermediate
     
    def fig_grids(self, show=False):
        """ Plots final deformed grid for each submodule """
        # Works for two shapes in a background
        # TO DO: Make it general for more submodules
        assert isinstance(self.__H.module, MultiShapeModule)
        assert isinstance(self.__gridpoints, torch.Tensor)
        grid_final,_ = self.shoot_grid(self.__gridpoints)
        
        xlim, ylim, _, n = self.__grid_params
        nx, ny = n
        
        x1, y1 = dm.usefulfunctions.vec2grid(grid_final[0], nx,ny)
        x2, y2 = dm.usefulfunctions.vec2grid(grid_final[1], nx,ny)
        x3, y3 = dm.usefulfunctions.vec2grid(grid_final[2], nx,ny)
        x,y = self.__grid

        fig_grid1 = plot_grid(x1.detach().numpy(), y1.detach().numpy(), color = 'blue', xlim=xlim, ylim=ylim, figsize=self.__figsize, dpi=self.__dpi)
        fig_grid2 = plot_grid(x2.detach().numpy(), y2.detach().numpy(), color = 'blue', xlim=xlim, ylim=ylim, figsize=self.__figsize, dpi=self.__dpi)
        fig_grid_bg = plot_grid( x3.detach().numpy(), y3.detach().numpy(), color = 'blue', xlim=xlim, ylim=ylim, figsize=self.__figsize, dpi=self.__dpi)
        
        if show == True:
            plt.show()
            
        label = point_labels(self.__source, self.__gridpoints).view(nx, ny)

        plt.figure()
        plot_MultiGrid([[x1,y1], [x2,y2], [x3,y3]], [x, y], xlim=xlim, ylim=ylim,label=label)
        
        if show == True:
            plt.show()
            
    
class Save_Results_SingleShape(Save_Results):
    def __init__(self, H, source, target, it=10, figsize=(5,5), dpi=100):
        self.__H = H
        self.__target = target
        self.__source = source
        self.__init_cotan = H.module.manifold.cotan
        self.__init_gd = H.module.manifold.gd
        
        self.__it = it
        self.__states, self.__controls = shoot_euler(H, it)     
        self.__H.module.manifold.fill_gd(self.__init_gd)
        self.__H.module.manifold.fill_cotan(self.__init_cotan)
        
        self.__gridpoints = None
        self.__grid = None
        self.__grid_params = None
        
        self.__figsize = figsize
        self.__dpi = dpi
        super().__init__(H, source, target, it=10, figsize=(5,5), dpi=100)
        super().__init__(H, source, target, it=10, figsize=(5,5), dpi=100)
    
    def gridpoints(self, xlims, ylims, d):
        """ Builds a grid specified by x-and y-limits and distance between gridpoints
            returns: x, y  (meshgrid); gridpoints (shape (-1,2)) """
        xmin, xmax = xlims
        ymin, ymax = ylims
        dx, dy = d
        nx = int((xmax-xmin)/dx)
        ny = int((ymax-ymin)/dy)
        x,y,gridpts = gridpoints(xmin, xmax, ymin, ymax, dx, dy)
        
        self.__grid_params = (xlims, ylims, d, [nx, ny])
        self.__gridpoints = gridpts
        self.__grid = [x,y]
        return x,y, gridpts
        
    def shoot_grid(self, gridpoints): 
        nb_pts, dim = gridpoints.shape

        silent_grid = SilentPoints(dm.manifold.Landmarks(dim, nb_pts, gd=self.__gridpoints.view(-1).requires_grad_() ))
        comp = CompoundModule([silent_grid, self.__H.module.copy()])
        H = dm.hamiltonian.Hamiltonian(comp)
        H.geodesic_controls()
        grid_states, _ = shoot_euler(H, self.__it)
        grid_final = grid_states[-1].gd
        return grid_final, grid_states
    
    def fig_states(self, show=False):
        """ Plots a separate figure with source, target and state for each state during shooting """
        fig_list = []
        
        for s in self.__states: 
            
            fig_shooting = plt.figure(figsize = self.__figsize, dpi=self.__dpi)
            for i in range(len(list(s))):
                plt.scatter(s[i].gd.view(-1,2)[:, 0].detach().numpy(), s[i].gd.view(-1,2)[:, 1].detach().numpy(), c='r')

            for i in range(len(self.__source)):
                plt.plot(self.__target[i][:, 0].detach().numpy(), self.__target[i][:, 1].detach().numpy(), 'xk')
                plt.plot(self.__source[i][:, 0].detach().numpy(), self.__source[i][:, 1].detach().numpy(), '.k')
            
            fig_list.append(fig_shooting)
        
        if show == True:
            plt.axis('equal')
            plt.show()
        return fig_list
                 
    
    def fig_grid(self, show=False):
        """ plots deformed grid for compound module """
        assert isinstance(self.__H.module, CompoundModule)
        assert isinstance(self.__gridpoints, torch.Tensor)
        nb_pts, dim = self.__gridpoints.shape
        grid_final, _ = self.shoot_grid(self.__gridpoints)
                
        xlim, ylim, _, n = self.__grid_params
        nx, ny = n        
        x, y = dm.usefulfunctions.vec2grid(grid_final[0].view(-1,dim), nx,ny)
        
        fig_grid = plot_grid(x.detach().numpy(), y.detach().numpy(), color = 'blue', xlim=xlim, ylim=ylim, figsize=self.__figsize, dpi=self.__dpi)
        plt.scatter(self.__states[-1][1].gd.view(-1,2)[:, 0].detach().numpy(), self.__states[-1][1].gd.view(-1,2)[:, 1].detach().numpy(), c='r')

        if show == True:
            plt.axis('equal')
            plt.show()
                