import torch
import scipy.optimize

class Optimizer:
    def __init__(self):
        super().__init__()

class BFGS(Optimizer):
    
    def __init__(self, EnergyFunctional, X, disp=True):
        ''' Optimizes the Energyfunctional using Scipy BFGS Algorithm'''
        super().__init__()
        self.__energyfun = EnergyFunctional
        self.__mom0 = X[1].detach().numpy()
        self.__gd0 = X[0]
        self.__disp = disp
        self.__iter_states = []
        
    def iter_states(self, x):
        self.__iter_states.append(x)
    
    
    def fun(self, P):
        Y = torch.tensor(P).requires_grad_()
        return self.__energyfun.energy_tensor(self.__gd0, Y).detach().numpy()

    def jac(self, P):
        Y = torch.tensor(P).requires_grad_()
        return self.__energyfun.gradE_autograd(self.__gd0, Y).detach().numpy()

    def __call__(self, maxiter, gtol=1e-03, eps=1e-08):
        res = scipy.optimize.minimize(self.fun, self.__mom0,
                              method='BFGS',
                              jac=self.jac,
                              bounds=None,
                              tol=None,
                              callback=self.iter_states,
                              options={
                                  'gtol': gtol,
                                  'eps': eps,
                                  'maxiter': maxiter,
                                  'disp' : self.__disp
                              })
        return torch.tensor(res.x), self.__iter_states
    
    
    
class Newton(Optimizer):
    
    def __init__(self, EnergyFunctional, X, disp=True):
        super().__init__()
        self.__energyfun = EnergyFunctional
        self.__mom0 = X[1].detach().numpy()
        self.__gd0 = X[0]
        self.__disp = disp
        self.__iter_states = []
        
    def iter_states(self, x):
        self.__iter_states.append(x)
    
    
    def fun(self, P):
        Y = torch.tensor(P).requires_grad_()
        return self.__energyfun.energy_tensor(self.__gd0, Y).detach().numpy()

    def jac(self, P):
        Y = torch.tensor(P).requires_grad_()
        return self.__energyfun.gradE_autograd(self.__gd0, Y).detach().numpy()

    def __call__(self, maxiter, gtol=1e-03, eps=1e-08):
        res = scipy.optimize.minimize(self.fun, self.__mom0,
                              method='Newton-CG',
                              jac=self.jac,
                              bounds=None,
                              tol=None,
                              callback=self.iter_states,
                              options={
                                  'xtol': gtol,
                                  'eps': eps,
                                  'maxiter': maxiter,
                                  'disp' : self.__disp
                              })
        return torch.tensor(res.x), self.__iter_states

    
  
    
class CG(Optimizer):
    
    def __init__(self, EnergyFunctional, X, disp=True):
        super().__init__()
        self.__energyfun = EnergyFunctional
        self.__mom0 = X[1].detach().numpy()
        self.__gd0 = X[0]
        self.__disp = disp
        self.__iter_states = []
        
    def iter_states(self, x):
        self.__iter_states.append(x)
    
    
    def fun(self, P):
        Y = torch.tensor(P).requires_grad_()
        return self.__energyfun.energy_tensor(self.__gd0, Y).detach().numpy()

    def jac(self, P):
        Y = torch.tensor(P).requires_grad_()
        return self.__energyfun.gradE_autograd(self.__gd0, Y).detach().numpy()

    def __call__(self, maxiter, gtol=1e-03, eps=1e-08):
        res = scipy.optimize.minimize(self.fun, self.__mom0,
                              method='CG',
                              jac=self.jac,
                              bounds=None,
                              tol=None,
                              callback=self.iter_states,
                              options={
                                  'xtol': gtol,
                                  'eps': eps,
                                  'maxiter': maxiter,
                                  'disp' : self.__disp
                              })
        return torch.tensor(res.x), self.__iter_states
