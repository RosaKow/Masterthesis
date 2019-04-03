import torch
import multimodule_usefulfunctions as mm 
import numpy as np


def armijo(E, gradE, energy, X):
    """ Armijo Stepsize Calculation 
    Args: E: functionvalue at current x
        gradE: gradient at current x
    Returns: alpha: stepsize
    """
    alpha = 1
    p = 0.8
    c1=10**-4
    norm = torch.norm
    while energy(X[0],X[1] - alpha*gradE[0]) > E - c1*alpha*norm(gradE[0]):
        alpha *= p
    return alpha

def gill_murray_wright(E, Enew, gradE,X, Xnew, k, kmax = 1000):
    """ Convergence Criteria of Gill, Murray, Wright """
    tau = 10**-8
    eps = 10**-8
    norm = torch.norm
    cond0 = E < Enew
    cond1 = E - Enew < tau*(1 + E)
    cond2 = norm(Xnew[1] - X[1])<np.sqrt(tau)*(1 + (norm(X[1]))**(1/2))
    cond3 = norm(gradE) < tau**(1/3)*(1 + E)
    cond4 = norm(gradE)<eps
    cond5 = k > kmax
    if (cond1 and cond2 and cond3) or cond4 or cond5 or cond0:
        print(cond1, cond2, cond3)
        print (cond4)
        print(cond5)
        return True
    return False

def gradientdescent(energy, energygradient, X):
    
    [gd, mom] = X
    k = 0
    convergence = False
    
    while convergence == False:
        gradE = energygradient(gd, mom)    
        E = energy(gd, mom)
        alpha = 0.01 #armijo(E, gradE, energy, [gd, mom])
        #Xnew = [X[0], X[1] + alpha*gradE]
        momnew = mom + alpha*gradE[0]
        print(" iter : {}  ,total energy: {}".format(k, E))
        convergence = gill_murray_wright(E, energy(gd, momnew), gradE[0], [gd, mom], [gd, momnew], k)
        print('convergence', convergence)
        mom = momnew
        k+=1
    print(" iter : {}  ,total energy: {}".format(k, energy(gd, mom)))
    return gd, mom


############################################

class EnergyFunctional():
    def __init__(self, module_list, h, Constr, target, sigma, dim=2, gamma=1):
        super().__init__()
        self.module_list = module_list
        self.h = h
        self.Constr = Constr
        self.target = target
        self.sigma = sigma
        self.dim = dim
        self.nb_pts = [len(target[0]), len(target[1])]
        self.gamma = gamma
        
    def attach(self, gd1, target):
        d = 0
        for i in range(len(self.module_list)):
            d = d + torch.dist(gd1[i], target[i])
        return d
        
    def cost(self,gd, control):   
        cost = 0
        for i in range(len(self.module_list)):
            cost = cost + self.module_list[i].cost(gd[i], control[i])
        return cost
    
    def shoot(self, gd_list0, mom_list0):
        return mm.shootMultishape(gd_list0, mom_list0, self.h, self.Constr, self.sigma, self.dim, n=10)
    
    def energy_tensor(self, gd0, mom0):
        ''' Energy functional for tensor input
            (to compute the automatic gradient the input is needed as tensor, not as list) '''
        
        gd0_list = self.tensor2list(gd0)
        mom0_list = self.tensor2list(mom0)

        gd_t,_,controls_t = self.shoot(gd0_list, mom0_list)
        
        geodesicControls0 = controls_t[-1]
        gd1 = gd_t[-1]
        
        z = [mm.computeCenter(gd0_list[0]), mm.computeCenter(gd0_list[1]), gd0_list[2]]
        
        cost = self.cost(z, geodesicControls0)
        attach = self.attach(gd1, self.target)
        print('cost:', cost)
        print('attach:', attach)
        print('gamma:', self.gamma)
        print('total:', self.gamma*cost + attach)
        
        return self.gamma*cost + attach
        
    
    def energy(self, gd0, mom0):
        gd_t,_,controls_t = self.shoot(gd0, mom0)
        
        geodesicControls0 = controls_t[0]
        gd1 = gd_t[-1]
        
        z = [mm.computeCenter(gd0[0]), mm.computeCenter(gd0[1]), gd0[2]]
        
        cost = self.cost(z, geodesicControls0)
        attach = self.attach(gd1, self.target)
        print('cost:', cost)
        print('attach:', attach)
        
        return self.gamma*cost + attach
    
    def gradE(self, gd0_tensor, mom0_tensor):
        return torch.autograd.grad(self.energy_tensor(gd0_tensor, mom0_tensor), mom0_tensor)
    
    def tensor2list(self, x):
        x_list = [x[0:self.nb_pts[0]], x[self.nb_pts[0]:self.nb_pts[0]+self.nb_pts[1]], x[self.nb_pts[1]+self.nb_pts[0]:]]
        return x_list
    