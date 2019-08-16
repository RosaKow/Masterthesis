import torch
import scipy.optimize



class BFGS():
    
    def __init__(self, EnergyFunctional, X, disp=True):
        ''' Optimizes the Energyfunctional using Scipy BFGS Algorithm'''
        super().__init__()
        self.__energyfun = EnergyFunctional
        self.__mom0 = X[1].detach().numpy()
        self.__gd0 = X[0]
        self.__disp = disp
    
    
    
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
                              callback=None,
                              options={
                                  'gtol': gtol,
                                  'eps': eps,
                                  'maxiter': maxiter,
                                  'disp' : self.__disp
                              })
        return torch.tensor(res.x)