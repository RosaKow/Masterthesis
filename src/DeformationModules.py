import torch
import numpy as np

import torch.nn as nn
from   torch.nn import Parameter
from torch.autograd import grad

import useful_fun as fun


# GD, Cont and Mom are vectors
class Module(nn.Module) :
    "Abstract Module."
    def __init__(self) :
        super(Module, self).__init__()

    def __call__(self, GD, Cont, Points) :
        "Applies the generated vector field on given points."
        raise NotImplementedError
    
    def action(self, GD, Mod, GD_mod, Cont_mod) :
        "Applies the vector field generated by Mod on GD"
        raise NotImplementedError
    
    def cost(self, GD, Cont) :
        "Returns the cost."
        raise NotImplementedError
    
class TranslationsIdenticalCost(Module):
    "Module generating sum of translations."
    def __init__(self, sigma, d, nb_pts):
        super(Module, self).__init__()
        self.scale = sigma
        self.dim = d
        self.nb_pts = nb_pts
        self.dimGD = d*nb_pts
        self.dimCont = d*nb_pts

    def __call__(self, GD, Cont, Points) :
        "Applies the generated vector field on given points."
        
        cov_mat = fun.K_xy(Points, GD.view([-1, self.dim]), self.scale)
        
        return cov_mat @ (Cont.view([-1, self.dim]))
    
    def action(self, GD, Mod, GD_mod, Cont_mod) :
        "Applies the vector field generated by Mod on GD"
        return Mod(GD_mod, Cont_mod, GD.view([-1, self.dim])).view(-1)
    
    def cost(self, GD, Cont) :
        "Returns the cost."
        #Cont_arr = Cont.view([-1, self.dim])
        #GD_arr = GD.view([-1, self.dim])
        #cov_mat = K_xy(GD_arr, GD_arr, self.scale)
        #return torch.tensor([Cont_arr[:,i] @ cov_mat @ Cont_arr[:,i] for i in range(self.dim)]).sum()
        return 0.5*fun.scal(Cont, Cont)

    
    def compute_geodesic_control(self, delta, GD):
        """ computes geodesic control from \delta \in H^\ast """
        return delta
    
    def cost_inv(self, GD, Cont) :
        #GD_arr = GD.view([-1, self.dim])
        #cov_mat = K_xy(GD_arr, GD_arr, self.scale)
        #Cont_arr = Cont.view([-1, self.dim])
        ##X = torch.stack([torch.gesv(Cont_arr[:,i], cov_mat) for i in range(self.dim)]).squeeze()
        #a, _ = torch.gesv(Cont_arr[:,0], cov_mat)
        #b, _ = torch.gesv(Cont_arr[:,1], cov_mat)
        #X = torch.stack([a, b]).view([2, -1])
        #Y=torch.t(X).contiguous()
        #return Y.view(-1)
        return Cont
        

        
class Translations(Module):
    "Module generating sum of translations."
    def __init__(self, sigma, d, nb_pts):
        super(Module, self).__init__()
        self.scale = sigma
        self.dim = d
        self.nb_pts = nb_pts
        self.dimGD = d*nb_pts
        self.dimCont = d*nb_pts

    def __call__(self, GD, Cont, Points) :
        "Applies the generated vector field on given points."
        
        cov_mat = fun.K_xy(Points, GD.view([-1, self.dim]), self.scale)
        
        return cov_mat @ (Cont.view([-1, self.dim]))
    
    def action(self, GD, Mod, GD_mod, Cont_mod) :
        "Applies the vector field generated by Mod on GD"
        return Mod(GD_mod, Cont_mod, GD.view([-1, self.dim])).view(-1)
    
    def cost(self, GD, Cont) :
        "Returns the cost."
        Cont_arr = Cont.view([-1, self.dim])
        GD_arr = GD.view([-1, self.dim])
        cov_mat = K_xy(GD_arr, GD_arr, self.scale)
        return 0.5*torch.tensor([Cont_arr[:,i] @ cov_mat @ Cont_arr[:,i] for i in range(self.dim)]).sum()
        
    
    def compute_geodesic_control(self, delta, GD):
        """ computes geodesic control from \delta \in H^\ast """
        raise NotImplementedError
    
    def cost_inv(self, GD, Cont) :
        #GD_arr = GD.view([-1, self.dim])
        #cov_mat = K_xy(GD_arr, GD_arr, self.scale)
        #Cont_arr = Cont.view([-1, self.dim])
        ##X = torch.stack([torch.gesv(Cont_arr[:,i], cov_mat) for i in range(self.dim)]).squeeze()
        #a, _ = torch.gesv(Cont_arr[:,0], cov_mat)
        #b, _ = torch.gesv(Cont_arr[:,1], cov_mat)
        #X = torch.stack([a, b]).view([2, -1])
        #Y=torch.t(X).contiguous()
        #return Y.view(-1)
        raise NotImplementedError
        






class SilentPoints(Module):
    "Module generating sum of translations."
    def __init__(self, d, nb_pts):
        super(Module, self).__init__()
        self.dim = d
        self.nb_pts = nb_pts
        self.dimGD = d*nb_pts
        self.dimCont = 0

    def __call__(self, GD, Cont, Points) :
        "Applies the generated vector field on given points."
        
        return torch.zeros_like(Points)
    
    def action(self, GD, Mod, GD_mod, Cont_mod) :
        "Applies the vector field generated by Mod on GD"
        return Mod(GD_mod, Cont_mod, GD.view([-1, d])).view(-1)
    
    def cost(self, GD, Cont) :
        "Returns the cost."
        return torch.tensor(0.)

         
    def compute_geodesic_control(self, delta, GD):
        """ computes geodesic control from \delta \in H^\ast """
        raise torch.tensor([])   

        
    def cost_inv(self, GD, Cont) :
        " Apply the inverse of K such that cost = Cont.view(-1) @ K Cont.view(-1) to Cont"
        
        return torch.tensor([])
        
        
            
        
        
    
class TranslationBased(Module):
    "Module generating sum of translations."
    def __init__(self, sigma, d, nb_pts, f_list, g_list, nb_fun):
        "f_list is a list of point-generators and g_list is a list of vector-generators (len (f_list) = len(g_list) = nb_fun) "
        
        super(Module, self).__init__()
        self.scale = sigma
        self.dim = d
        self.nb_pts = nb_pts
        self.dimGD = d*nb_pts
        self.dimCont = nb_fun
        self.f = f_list
        self.g = g_list

    def __call__(self, GD, Cont, Points) :
        "Applies the generated vector field on given points."
        
        speed = torch.zeros_like(Points)
        for i in range(self.dimCont):
            supp_pts = self.f[i](GD)
            supp_vec = self.g[i](GD)
            cov_mat = K_xy(Points, supp_pts.view([-1, self.dim]), self.scale)
            speed = speed + Cont[i] * (cov_mat @  (supp_vec.view([-1, self.dim])))
        
        return speed
    
    def action(self, GD, Mod, GD_mod, Cont_mod) :
        "Applies the vector field generated by Mod on GD"
        return Mod(GD_mod, Cont_mod, GD.view([-1, d])).view(-1)
    
    def cost(self, GD, Cont) :
        "Returns the cost."
        #cov_mat = K_xy(GD, GD, self.scale)
        #return torch.tensor([Cont[:,i] @ cov_mat @ Cont[:,i] for i in range(self.dim)]).sum()
        return scal(Cont, Cont)

    def cost_inv(self, GD, Cont) :
        "inverse of K such that cost = Cont.view(-1) @ K Cont.view(-1)"
        return Cont
        #return torch.eye(self.dimCont)
        

class Compound(Module):
    "Combination of modules"
    def __init__(self, Mod_list):
        super(Module, self).__init__()
        self.Mod_list = Mod_list
        self.Nb_mod = len(Mod_list)
        self.dimGD = sum([mo.dimGD for mo in Mod_list])
        self.dimCont = sum([mo.dimCont for mo in Mod_list])
        self.indiceGD = [0]
        self.indiceGD.extend(np.cumsum([mo.dimGD for mo in Mod_list]))
        self.indiceCont = [0]
        self.indiceCont.extend(np.cumsum([mo.dimCont for mo in Mod_list]))
        
        
    def __call__(self, GD, Cont, Points) :
        list_app = [self.Mod_list[i](GD[self.indiceGD[i]:self.indiceGD[i+1]], Cont[self.indiceCont[i]:self.indiceCont[i+1]], Points).unsqueeze(0) for i in range(self.Nb_mod)]
        return torch.sum(torch.cat(list_app), 0)
        
    
    def action(self, GD, Mod, GD_mod, Cont_mod) :
        "Applies the vector field generated by Mod on GD"
        list_app = [self.Mod_list[i].action(GD[self.indiceGD[i]:self.indiceGD[i+1]], Mod, GD_mod, Cont_mod) for i in range(self.Nb_mod)]
        return torch.cat(list_app)
    
    def cost(self, GD, Cont) :
        "Returns the cost."
        #cov_mat = K_xy(GD, GD, self.scale)
        #return torch.tensor([Cont[:,i] @ cov_mat @ Cont[:,i] for i in range(self.dim)]).sum()
        list_cost = [self.Mod_list[i].cost(GD[self.indiceGD[i]:self.indiceGD[i+1]], Cont[self.indiceCont[i]:self.indiceCont[i+1]]).unsqueeze(0) for i in range(self.Nb_mod)]
        return torch.sum(torch.cat(list_cost), 0)

         
    def compute_geodesic_control(self, delta, GD):
        """ computes geodesic control from \delta \in H^\ast """
        
        list_cont = [self.Mod_list[i].compute_geodesic_control(delta[self.indiceCont[i]:self.indiceCont[i+1]], GD[self.indiceGD[i]:self.indiceGD[i+1]]) for i in range(self.Nb_mod)]
        #print(list_cont)
        return torch.cat(list_cont)

        
    def cost_inv(self, GD, Cont):
        X = torch.zeros_like(Cont)
        for i in range(self.Nb_mod):
            indmin=self.indiceCont[i]
            indmax=self.indiceCont[i+1]
            indminGD=self.indiceGD[i]
            indmaxGD=self.indiceGD[i+1]
            GD_i = GD[indminGD:indmaxGD].view(-1)
            X[indmin:indmax] = self.Mod_list[i].cost_inv(GD_i, Cont[indmin:indmax])
        return X