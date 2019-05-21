import torch
import numpy as np

from .multimodule_usefulfunctions import gdlist2tensor, gdlist_reshape


class Constraints():
    ''' '''
    def __init__(self):
        super().__init__()
        
        
class Identity(Constraints):
    ''' '''
    def __init__(self):
        """ manifold must be a compound manifold of m+1 manifolds, the last one being the compound of the 1 to m first manifolds (background) 
        gd are assumed to be flattened"""
        super().__init__()
       
    def constraintsmatrix(self, modules):
        """ Matrix that corresponds to the function g in C = g xi """
        n = modules.manifold.numel_gd
        G = torch.eye(n)
        for i in range(len(modules.module_list)-2):
            ni = modules.manifold.numel_gd
            G = torch.cat([torch.cat([G, torch.zeros(n, ni)], 1), torch.cat([torch.zeros(ni, n), torch.eye(ni)], 1)], 0)
            n = n + ni
        G = torch.cat( [G, -torch.eye(n)], 1)   
        return G
    
    def call_by_matmul(self, modules):
        fields = modules.field_generator()
        action = gdlist2tensor([ f(man) for f,man in zip(fields, gdlist_reshape(modules.manifold.gd,[-1,modules.manifold.dim]))]).view(-1,1)
        return  torch.mm(self.constraintsmatrix(modules), action)
                
    def __call__(self, modules):
        ''' applies identity constraints on generated velocity field'''
        constr = torch.tensor([])
        fields = modules.field_generator()
        field_bg = fields[-1]
        
        for i in range(len(modules.module_list) -1):
            gd_bg = modules.manifold.manifold_list[-1].manifold_list[i].gd
            constr = torch.cat([constr, fields[i](modules.manifold.gd[i].view(-1,modules.manifold.dim)) 
                                - field_bg(gd_bg.view(-1,modules.manifold.dim))], 0)
            
        return constr

    
class Null(Constraints):
    """ applying no constraints, setting constraints value to zero """
    def __init__(self):
        super().__init__()
        
    def constraintsmatrix(self, modules):
        N = sum([gd.view(-1).shape[0] for gd in modules.manifold.gd[:-1]])
        return torch.zeros(N, 2*N)
    
    def __call__(self, modules):
        N = sum([gd.view(-1).shape[0] for gd in modules.manifold.gd[:-1]])
        return torch.zeros(N).view(-1, modules.manifold.dim)