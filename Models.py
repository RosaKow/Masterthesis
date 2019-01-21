import torch
import torch.nn as nn
import torch.optim
import numpy as np

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

    def H_r(self, GD, MOM):
        return self.H(GD, MOM, self.H.Cont_geo(GD, MOM))

    def getVars(self):
        raise NotImplementedError

    def getModuleCompound(self):
        raise NotImplementedError
    
    def cost(self, target):
        GD, MOM = self.getVars()
        return fidelity(self(), target) + self.getModuleCompound().cost(GD, self.H.Cont_geo(GD, MOM))

    def fit(self, target, maxiter=100):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.0001, momentum=0.05)
        #self.fit.nit = 0
        #self.fit.breakloop = False
        costs = []

        def closure():
            #self.fit.nit += 1
            optimizer.zero_grad()
            cost = self.cost(target)
            cost.backward()
            costs.append(cost.item())
            return cost

        for i in range(0, maxiter):
            optimizer.step(closure)

            if i%1 == 0:
                print("It:", i, ", cost:", costs[-1])

        return costs


class ModelTranslationModuleRegistration(Model):
    def __init__(self, sigma, dim, source, translationGD, fixedTranslationPoints = True):
        super().__init__()
        self.dim = dim
        self.data = source[0]
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
        return torch.cat((self.data.view(-1), self.translationGD), 0), torch.cat((self.silentMOM, self.translationMOM), 0)

    def getModuleCompound(self):
        return self.compound
            
    def __call__(self, it = 10):
        GD_In, MOM_In = self.getVars()
        GD_Out, MOM_Out = Shooting.shoot(self.modules, GD_In, MOM_In, self.H_r, it)
        return GD_Out[0:self.data.shape[0]*self.dim].view(-1, self.dim), self.alpha
    

