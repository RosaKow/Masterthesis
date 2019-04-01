def armijo(E, gradE):
    """ Armijo Stepsize Calculation 
    Args: E: functionvalue at current x
        gradE: gradient at current x
    Returns: alpha: stepsize
    """
    alpha = 1
    p = 0.8
    c1=10**-4
    norm = odl.solvers.L2NormSquared(X[1].space)
    while energy([X[0],X[1] - alpha*gradE]) > E - c1*alpha*norm(gradE):
        alpha *= p
    return alpha

def gill_murray_wright(E, Enew, gradE,X, Xnew, k, kmax = 1000):
    """ Convergence Criteria of Gill, Murray, Wright """
    tau = 10**-8
    eps = 10**-8
    norm = odl.solvers.L2NormSquared(X[1].space)
    cond0 = E < Enew
    cond1 = E - Enew < tau*(1 + E)
    cond2 = norm(Xnew[1] - X[1])<np.sqrt(tau)*(1 + (norm(X[1]))**(1/2))
    cond3 = norm(gradE) < tau**(1/3)*(1 + E)
    cond4 = norm(gradE)<eps
    cond5 = k > kmax
    if (cond1 and cond2 and cond3) or cond4 or cond5 or cond0:
        return True
    return False

def gradientdescent(X):
    
    k = 0
    convergence = False
    
    while convergence == False:
        gradE = energygradient(X)    
        E = energy(X)
        alpha = 0.00001 #armijo(E, gradE)
        Xnew = [X[0], X[1] + alpha*gradE]
        print(" iter : {}  ,  attachment term : {}".format(k,E))
        convergence = gill_murray_wright(E, energy(Xnew), gradE, X, Xnew, k)
        X = Xnew
        k+=1
    return X


############################################

class EnergyFunctional():
    def __init__(module, h, Constr, target):
        self.module = module
        
    def attach(self, gd1, target):
        return torch.dist(gd1, target)
        
    def cost(self,gd, control):   
        return module.cost(gd, control)
    
    def shoot(gd_list0, mom_list0):
        return shootMultishape(gd_list0, mom_list0, h, Constr, sigma, dim, n=10)
    
    def energy(self, gd0, mom0, gamma=1):
        gd_t,~,controls_t = shoot(gd0, mom0)
        
        geodesicControls0 = controls_t[0]
        gd1 = gd_t[-1]
        
        return cost(gd0, geodesicControls0) + gamma*attach(gd1, target)
    
    