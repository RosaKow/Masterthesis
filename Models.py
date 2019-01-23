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

    def getVarTensor(self):
        raise NotImplementedError

    def getVarList(self):
        raise NotImplementedError

    def getModuleCompound(self):
        raise NotImplementedError

    def shootTensor(self):
        raise NotImplementedError

    def shootList(self):
        raise NotImplementedError
    
    def cost(self, target, l=1.):
        GD, MOM = self.getVarTensor()
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
            attach, deformationCost = self.cost(target, l)
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
        self.data = source[0].clone()
        self.alpha = source[1].clone()
        self.sigma = sigma

        self.silentModule = DeformationModules.SilentPoints(self.dim, self.data.shape[0])
        self.translationModule = DeformationModules.Translations(self.sigma, self.dim, translationGD.view(-1, self.dim).shape[0])
        self.compound = DeformationModules.Compound([self.silentModule, self.translationModule])
        self.H = Hamiltonian.Hamilt(self.compound)

        self.translationMOM = torch.nn.Parameter(torch.zeros_like(translationGD).view(-1))
        self.silentMOM = torch.nn.Parameter(torch.zeros_like(self.data).view(-1))

        self.translationGD = None
        if(fixedTranslationPoints):
            self.translationGD = translationGD.clone().requires_grad_()
        else:
            self.translationGD = torch.nn.Parameter(translationGD.clone())

    def getVarTensor(self):
        """Returns the variables from the problem (GDs and MOMs) as tensors."""
        return torch.cat((self.data.view(-1), self.translationGD), 0), torch.cat((self.silentMOM, self.translationMOM), 0)

    def getVarList(self):
        """Returns the variables from the problem (GDs and MOMs) as lists."""
        return [self.data.view(-1), self.translationGD], [self.silentMOM, self.translationMOM]

    def shootTensor(self, it=10):
        """Solves the shooting equations and returns the result as tensors."""
        GD, MOM = self.getVarTensor()
        return Shooting.shoot(self.compound, GD, MOM, self.H, it)

    def shootList(self, it=10):
        """Solves the shooting equations and returns the result as lists."""
        GD, MOM = self.getVarTensor()
        GD, MOM = Shooting.shoot(self.compound, GD, MOM, self.H, it)
        return [GD.view(-1, self.dim)[0:self.data.shape[0]], GD.view(-1, self.dim)[self.data.shape[0]:-1]], [MOM.view(-1, self.dim)[0:self.data.shape[0]], MOM.view(-1, self.dim)[self.data.shape[0], -1]]

    def getModuleCompound(self):
        return self.compound

    def __call__(self, it=10):
        """Returns the projected data by the deformation modules."""
        GD_In, MOM_In = self.getVarTensor()
        GD_Out, MOM_Out = Shooting.shoot(self.modules, GD_In, MOM_In, self.H, it)
        return GD_Out[0:self.data.shape[0]*self.dim].view(-1, self.dim), self.alpha
    

class ModelCompoundRegistration(Model):
    def __init__(self, dim, source, moduleList, GDList, fixed):
        super(Model, self).__init__()
        self.dim = dim
        self.data = source[0].clone()
        self.alpha = source[1].clone()

        self.MOMData = torch.nn.Parameter(torch.zeros_like(self.data).view(-1))

        self.GDParams, self.MOMParams = torch.nn.ParameterList(), torch.nn.ParameterList()
        self.GDFixed = []
        self.ModList = [DeformationModules.SilentPoints(self.dim, self.data.shape[0])]

        for i in range(len(moduleList)):
            if(fixed[i]):
                self.GDFixed.append(GDList[i].clone().requires_grad_())
                self.ModList.append(moduleList[i])

        for i in range(len(moduleList)):
            if(not fixed[i]):
                self.GDParams.append(torch.nn.Parameter(GDList[i].clone()))
                self.MOMParams.append(torch.nn.Parameter(torch.zeros_like(GDList[i])))
                self.ModList.append(moduleList[i])

        self.compound = DeformationModules.Compound(self.ModList)
        self.H = Hamiltonian.Hamilt(self.compound)

    def getVarTensor(self):
        GDList = [self.data.view(-1), *self.GDFixed, *self.GDParams]
        MOMList = [self.MOMData, torch.zeros(sum(a.shape[0] for a in self.GDFixed)), *self.MOMParams]
        return torch.cat(GDList), torch.cat(MOMList)

    def getVarList(self):
        return [self.data.view(-1), *self.GDFixed, *self.GDParams], [self.MOMData, torch.zeros(sum(a.shape[0] for a in self.GDFixed)), *self.MOMParams]

    def shootTensor(self, it=10):
        GD_In, MOM_In = self.getVarTensor()
        return Shooting.shoot(self.compound, GD_In, MOM_In, self.H, it)

    def shootList(self, it=10):
        """Solves the shooting equations and returns the result as lists."""
        GD, MOM = self.getVarTensor()
        GD, MOM = Shooting.shoot(self.compound, GD, MOM, self.H, it)
        GDList = []
        MOMList = []
        for i in range(self.compound.Nb_mod):
            GDList.append(GD[self.compound.indiceGeoDesc[i]:self.compound.indiceGeoDesc[i+1]])
            MOMList.append(MOM[self.compound.indiceGeoDesc[i]:self.compound.indiceGeoDesc[i+1]])
        return GDList, MOMList

    def getModuleCompound(self):
        return self.compound

    def __call__(self, it = 10):
        GD_In, MOM_In = self.getVarTensor()
        GD_Out, MOM_Out = Shooting.shoot(self.compound, GD_In, MOM_In, self.H, it)
        return GD_Out.view(-1, self.dim)[0:self.data.shape[0]], self.alpha
        

