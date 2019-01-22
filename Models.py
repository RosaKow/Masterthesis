import torch
import torch.nn as nn
import torch.optim
import numpy as np
import itertools

import DeformationModules
import Hamiltonian
import Shooting
import UsefulFunctions as fun


def fidelity(a, b) :
    "Energy Distance between two sampled probability measures."
    x_i, a_i = a
    y_j, b_j = b
    K_xx = -fun.distances(x_i, x_i)
    K_xy = -fun.distances(x_i, y_j)
    K_yy = -fun.distances(y_j, y_j)
    cost = .5*fun.scal(a_i, torch.mm(K_xx, a_i.view(-1, 1))) - fun.scal(a_i, torch.mm(K_xy, b_j.view(-1, 1))) + .5*fun.scal(b_j, torch.mm(K_yy, b_j.view(-1, 1)))
    return cost


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, it=10):
        raise NotImplementedError

    def getVars(self):
        raise NotImplementedError

    def getModuleCompound(self):
        raise NotImplementedError
    
    def cost(self, target, l=1.):
        GD, MOM = self.getVars()
        attach = l*fidelity(self(), target)
        deformationCost = self.getModuleCompound().cost(GD, self.H.Cont_geo(GD, MOM))
        return attach, deformationCost

    def fit(self, target, lr=1e-3, l=1., maxiter=100, tol=1e-7, logInterval=10):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.05)
        self.nit = -1
        self.breakloop = False
        costs = []

        def closure():
            self.nit += 1
            optimizer.zero_grad()
            attach, deformationCost = self.cost(target, 50.)
            cost = attach + deformationCost

            if(self.nit%logInterval == 0):
                print("It: %d, deformation cost: %.6f, attach: %.6f. Total cost: %.6f" % (self.nit, deformationCost.detach().numpy(), attach.detach().numpy(), cost.detach().numpy()))

            costs.append(cost.item())

            cost.backward()

            if(len(costs) > 1 and abs(costs[-1] - costs[-2]) < tol) or self.nit >= maxiter:
                self.breakloop = True

            return cost

        for i in range(0, maxiter):
            optimizer.step(closure)

            if(self.breakloop):
                break

        print("End of the optimisation process")
        return costs


class ModelTranslationModuleRegistration(Model):
    def __init__(self, sigma, dim, source, translationGD, fixedTranslationPoints = True):
        super().__init__()
        self.dim = dim
        self.data = source[0].requires_grad_()
        self.alpha = source[1]
        self.sigma = sigma

        self.silentModule = DeformationModules.SilentPoints(self.dim, self.data.shape[0])
        self.translationModule = DeformationModules.Translations(self.sigma, self.dim, translationGD.view(-1, self.dim).shape[0])
        self.compound = DeformationModules.Compound([self.silentModule, self.translationModule])
        self.H = Hamiltonian.Hamilt(self.compound)

        self.translationMOM = torch.nn.Parameter(torch.zeros_like(translationGD).view(-1))
        self.silentMOM = torch.nn.Parameter(torch.zeros_like(self.data).view(-1))

        self.translationGD = None
        if(fixedTranslationPoints):
            self.translationGD = translationGD
        else:
            self.translationGD = torch.nn.Parameter(translationGD)

    def getVars(self):
        GD, MOM = torch.cat((self.data.view(-1), self.translationGD), 0), torch.cat((self.silentMOM, self.translationMOM), 0)
        return GD, MOM

    def shoot(self):
        GD, MOM = self.getVars()
        return Shooting.shoot(self.modules, GD, MOM, self.H, 10)

    def getModuleCompound(self):
        return self.compound
            
    def __call__(self, it = 10):
        GD_In, MOM_In = self.getVars()
        GD_Out, MOM_Out = Shooting.shoot(self.modules, GD_In.requires_grad_(), MOM_In.requires_grad_(), self.H, it)
        return GD_Out[0:self.data.shape[0]*self.dim].view(-1, self.dim), self.alpha
    

class ModelCompoundRegistration(Model):
    def __init__(self, dim, source, moduleList, GDList):
        super(Model, self).__init__()
        self.dim = dim
        self.data = source[0]
        self.alpha = source[1]
        self.compound = DeformationModules.Compound([mod for sublist in [[DeformationModules.SilentPoints(self.dim, source[0].shape[0])], moduleList] for mod in sublist])
        print(self.compound.Mod_list)
        
        self.GDList = torch.nn.ParameterList()
        self.MOMList = torch.nn.ParameterList()
        
        for i in range(0, len(moduleList)):
            self.GDList.append(torch.nn.Parameter(GDList[i]))
            self.MOMList.append(torch.nn.Parameter(torch.zeros_like(GDList[i])))
        
    def getVars(self):
        GD = self.data.view(-1)        
        GD = torch.cat([GD, *self.GDList])
        MOM = self.data.view(-1)
        MOM = torch.cat([MOM, *self.MOMList])
        return GD, MOM

    def getModuleCompound(self):
        return self.compound

    def __call__(self, it = 10):
        GD_In, MOM_In = self.getVars()
        GD_Out, MOM_Out = Shooting.shoot(self.compound, GD_In, MOM_In, self.H, it)
        #return GD_Out[0:
        

