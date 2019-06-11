import torch
import numpy as np


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
        n = modules.module_list[0].manifold.numel_gd
        G = torch.eye(n)
        for i in range(len(modules.module_list)-2):
            ni = modules.module_list[i+1].manifold.numel_gd
            G = torch.cat([torch.cat([G, torch.zeros(n, ni)], 1), torch.cat([torch.zeros(ni, n), torch.eye(ni)], 1)], 0)
            n = n + ni
        G = torch.cat( [G, -torch.eye(n)], 1)   
        return G
    
    def call_by_matmul(self, modules):
        fields = modules.field_generator().fieldlist
                
        action = torch.cat([*[ torch.cat(f(p)) for f,p in zip(fields[:-1], modules.manifold.gd_points()[:-1])],
                  *[fields[-1](torch.cat(modules.manifold.manifold_list[-1].unroll_gd_points()))]]).view(-1,1)
        
        return  torch.mm(self.constraintsmatrix(modules), action)
                
    def __call__(self, modules):
        ''' applies identity constraints on generated velocity field'''
        constr = torch.tensor([])
        fields = modules.field_generator().fieldlist
        field_bg = fields[-1]
                
        for i in range(len(modules.module_list) -1):
            gd_bg = modules.manifold.manifold_list[-1].gd_points()[i]
            constr = torch.cat([constr, torch.cat(fields[i](modules.manifold.gd_points()[i])) 
                                - field_bg(gd_bg)], 0)
            
        return constr

class Identity_Silent(Constraints):
    ''' '''
    def __init__(self):
        """ manifold must be a compound manifold of m+1 manifolds, the last one being the compound of the 1 to m first manifolds (background) 
        gd are assumed to be flattened"""
        super().__init__()
       
    def constraintsmatrix(self, modules):
        """ Matrix that corresponds to the function g in C = g xi """
        n = modules.module_list[0].manifold.manifold_list[0].numel_gd
        G = torch.eye(n)
        for i in range(len(modules.module_list)-2):
            ni = modules.module_list[i+1].manifold.manifold_list[0].numel_gd
            G = torch.cat([torch.cat([G, torch.zeros(n, ni)], 1), torch.cat([torch.zeros(ni, n), torch.eye(ni)], 1)], 0)
            n = n + ni
        G = torch.cat( [G, -torch.eye(n)], 1)   
        return G
    
    def call_by_matmul(self, modules):
        fields = modules.field_generator().fieldlist
        
        action = torch.cat([*[ f(man.manifold_list[0].gd.view(-1,2)) for f,man in zip(fields[:-1], modules.manifold.manifold_list[:-1])],
                  *[fields[-1](torch.cat(modules.manifold.manifold_list[-1].gd_points()))]]).view(-1,1)
        
        return  torch.mm(self.constraintsmatrix(modules), action)
                
    def __call__(self, modules):
        ''' applies identity constraints on generated velocity field'''
        constr = torch.tensor([])
        fields = modules.field_generator().fieldlist
        field_bg = fields[-1]
                            
        for i in range(len(modules.module_list) -1):
            gd_bg = modules.manifold.manifold_list[-1].gd_points()[i]
            # field i is applied to the silent points that correspond to the ith boundary 
            constr = torch.cat([constr, fields[i](modules.module_list[i].module_list[0].manifold.gd.view(-1,2))
                                - field_bg(gd_bg)], 0)        
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