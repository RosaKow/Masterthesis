import torch
import numpy as np
import defmod as dm
import multimodule_usefulfunctions as mm


class Constraints():
    ''' '''
    def __init__(self):
        super().__init__()
        
        
class Identity(Constraints):
    ''' '''
    def __init__(self, modules):
        # input: compound manifold, list of fields
        """ manifold must be a compound manifold of m+1 manifolds, the last one being the compound of the 1 to m first manifolds (background) 
        gd are assumed to be flattened"""
        super().__init__()
        self.__modules = modules
        self.__manifold = modules.manifold
        self.__gd = self.__manifold.gd
        self.__nb = len(self.__modules.module_list) -1
        self.__dim = self.__manifold.dim
        

    def constraintsmatrix(self):
        """ Matrix that corresponds to the function g in C = g xi """
        n = self.__gd[0].shape[0]
        G = torch.eye(n)
        for i in range(self.__nb-1):
            ni = self.__gd[i+1].shape[0]
            G = torch.cat([torch.cat([G, torch.zeros(n, ni)], 1), torch.cat([torch.zeros(ni, n), torch.eye(ni)], 1)], 0)
            n = n + ni
        G = torch.cat( [G, -torch.eye(n)], 1)   
        return G
    
    def call_by_matmul(self):
        #fields = [zeta(c) for zeta, c in zip(self.__fieldgenerators, controls)]
        fields = self.__modules.field_generator()
        action = mm.gdlist2tensor([ f(man) for f,man in zip(fields, mm.gdlist_reshape(self.__gd,[-1,self.__manifold.dim]))]).view(-1,1)
        return  torch.mm(self.constraintsmatrix(), action)
                
    def __call__(self):
        ''' applies identity constraints on generated velocity field'''
        constr = torch.tensor([])
        fields = self.__modules.field_generator()
        field_bg = fields[-1]
        #fields = [zeta(c) for zeta, c in zip(self.__fieldgenerators, controls)]
        # is this way to compute maybe faster than the matrixmultiplication? 
        for i in range(self.__nb):
            gd_bg = self.__manifold.manifold_list[-1].manifold_list[i].gd
            constr = torch.cat([constr, fields[i](self.__manifold.gd[i].view(-1,self.__dim)) - field_bg(gd_bg.view(-1,self.__dim))], 0)
            #print('field module:', self.__fields[i](self.__manifold.gd[i].view(-1,self.__dim)))
            #print('field background:', field_bg(gd_bg.view(-1,self.__dim)))
        return constr
