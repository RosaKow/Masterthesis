import torch
import multimodule_usefulfunctions as mm 
import numpy as np
from defmod.shooting import shoot_euler, shoot_euler_source
from defmod.attachement import L2NormAttachement_multi, L2NormAttachement


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

def gill_murray_wright(E, Enew, gradE,X, Xnew, k, kmax = 50):
    """ Convergence Criteria of Gill, Murray, Wright """
    tau = 10**-9
    eps = 10**-8
    norm = torch.norm
    cond0 = E < Enew
    cond1 = E - Enew < tau*(1 + E)
    cond2 = norm(Xnew[1] - X[1])<np.sqrt(tau)*(1 + (norm(X[1]))**(1/2))
    cond3 = norm(gradE) < tau**(1/3)*(1 + E)
    cond4 = norm(gradE)<eps
    cond5 = k >= kmax
    if (cond1 and cond2 and cond3) or cond4 or cond5 or cond0:
        print('Condition 0:', cond0)
        print('Condition 1, 2, 3:',cond1, cond2, cond3)
        print ('Contition 4:', cond4)
        print('Condition 5:', cond5)
        return cond0, cond1
    return cond0, cond1

def gradientdescent( EnergyFunctional , X):
    
    energy = EnergyFunctional.energy_tensor
    energygradient = EnergyFunctional.gradE_autograd
    
    [gd, mom] = X
    k = 0
    convergence = False
    alpha = 0.1
    
    Enew = energy(gd, mom)
    E = Enew
    print(" iter : {}  ,total energy: {}".format(k, Enew.detach().numpy()))
    c=0

    
    while convergence == False:
        gradE = energygradient(gd, mom)    
        
        
       # alpha = armijo(E, gradE, energy, X) 
        
                        
        momnew = mom - alpha*gradE
        Enew = energy(gd, momnew)
        cond0, cond1 = gill_murray_wright(E, Enew, gradE, [gd, mom], [gd, momnew], k)
        
        if cond0 == True:
            alpha = alpha*0.5
            c = c+1
        elif cond1 == True:
            alpha = alpha*1.5
            E = Enew
            mom = momnew.detach().requires_grad_()
            c = c+1
        else:
            alpha = alpha*1.2
            E = Enew
            mom = momnew.detach().requires_grad_()
            c = 0
        print('c', c)
    
        print(" iter : {}  ,total energy: {}".format(k, Enew.detach().numpy()))
        if c == 5 or k == 50:
            convergence = True
        k+=1
        
    print(" iter : {}  ,total energy: {}".format(k, Enew))
    return gd, mom


############################################

class EnergyFunctional():
    def __init__(self, modules, h, Constr, source, target, dim=2, gamma=1):
        super().__init__()
        self.__modules = modules
        self.h = h
        self.Constr = Constr
        self.source = source
        self.target = target
        self.dim = dim
        self.nb_pts = [modules.module_list[0].manifold.nb_pts, modules.module_list[1].manifold.nb_pts]
        self.gamma = gamma
        
    @property
    def modules(self):
        return self.__modules
        
    def attach(self):
        return sum([L2NormAttachement()( self.modules.module_list[i][0].manifold.gd, self.target[i]) 
                    #+ L2NormAttachement()( self.modules.module_list[-1], self.target[i])
                    for i in range(len(self.target))])
        
    def cost(self):   
        return self.modules.cost()
      
    def shoot(self):       
        intermediate_states, intermediate_controls = shoot_euler_source(self.h, self.source, it=10)
        return intermediate_states, intermediate_controls    
    
    
    def energy_tensor(self, gd0, mom0):
        ''' Energy functional for tensor input
            (to compute the automatic gradient the input is needed as tensor, not as list) '''
        
        self.h.module.manifold.fill_gd(gd0)
        self.h.module.manifold.fill_cotan(mom0)
        self.h.geodesic_controls()
        self.shoot()
                        
        cost = self.cost()
        attach = self.attach()
        print('cost:', self.gamma * cost.detach().numpy(), 'attach:', attach.detach().numpy())
        
        return self.gamma*cost + attach
        
    
    def energy(self, gd0, mom0):
        gd_t,_,controls_t = self.shoot(gd0, mom0)
        
        geodesicControls0 = controls_t[0]
        gd1 = gd_t[-1]
        
        z = [mm.computeCenter(gd0[0]), mm.computeCenter(gd0[1]), gd0[2]]
        
        cost = self.cost(z, geodesicControls0)
        attach = self.attach(gd1, self.target)
        
        return self.gamma*cost + attach
    
    def gradE(self, gd0_tensor, mom0_tensor):
        E = self.energy_tensor(gd0_tensor, mom0_tensor)
        E.backward(create_graph = True)
        grad = mom0_tensor.grad
        #mom0_tensor.grad.data.zero_()         <- doesn't seem to be needed
        return grad
    
    def gradE_autograd(self, gd0_tensor, mom0_tensor):
        grad = torch.autograd.grad(self.energy_tensor(gd0_tensor, mom0_tensor), mom0_tensor)[0]
        return grad
    
    def tensor2list(self, x):
        x_list = [x[0:self.nb_pts[0]*self.dim], x[self.nb_pts[0]*self.dim:(self.nb_pts[0]+self.nb_pts[1])*self.dim], [x[(self.nb_pts[1]+self.nb_pts[0])*self.dim:(self.nb_pts[1]+2*self.nb_pts[0])*self.dim], x[(self.nb_pts[1]+2*self.nb_pts[0])*self.dim:]]]
        return x_list
    
    def list2tensor(self, x):
        a = [*[a for a in x[:-1]], *[a for a in x[-1]]]
        return torch.cat(a,0).requires_grad_().view(-1).double()
    