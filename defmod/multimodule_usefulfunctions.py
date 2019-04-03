import torch
import math
import numpy as np
import defmod as dm

pi = math.pi

def PointsOnCircle(origin, r,n):
    return [(origin[0] + math.cos(2*pi/n*x)*r,origin[1] + math.sin(2*pi/n*x)*r) for x in range(0,n)]

def multipleCircles(origin, r, n):
    circle1 = PointsOnCircle(origin[0], r[0], n[0])
    circle2 = PointsOnCircle(origin[1], r[1], n[1])
    return [torch.tensor(circle1, requires_grad=True), torch.tensor(circle2, requires_grad=True)]

def computeCenter(gd):
    return torch.mean(gd,0)

def kronecker_I2(K):
    N = K.shape
    tmp = K.view(-1,1).repeat(1,2).view(N[0],2*N[1]).transpose(1,0).contiguous().view(-1,1).repeat(1,2).view(-1,N[0]*2).transpose(1,0)
    Ktilde = torch.mul(tmp, torch.eye(2).repeat(N))
    return Ktilde

def pointInCircles(x,y , z, r):
    label = torch.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            
            if (torch.norm(z[0] - torch.tensor([x[i][j], y[i][j]])) < r[0]):
                label[i][j] = 1
            elif (torch.norm(z[1] - torch.tensor([x[i][j], y[i][j]])) < r[1]):
                label[i][j] = 2
            else:
                label[i][j] = 3
    return label
            

import matplotlib.pyplot as plt
def plotControl(controls_list, z):
    X = []
    Y = []
    controlX = []
    controlY = []
    for i in range(len(z)):
        X.append(z[i][0].detach().numpy())
        Y.append(z[i][1].detach().numpy())
        controlX.append(controls_list[i][:,0].detach().numpy())
        controlY.append(controls_list[i][:,1].detach().numpy())
    plt.quiver( X, Y, controlX, controlY)
    return
        

def plot_grid(gridx,gridy, **kwargs):
    fig1 = plt.figure()  
    ax = fig1.add_subplot(1, 1, 1) 
    for i in range(gridx.shape[0]):
        ax.plot(gridx[i,:], gridy[i,:], **kwargs)
    for i in range(gridx.shape[1]):
        ax.plot(gridx[:,i], gridy[:,i], **kwargs)
                
        
        
def largeDeformation(module, gd_t, controls_t, points):
    """ compute large deformation of points"""
    phi = points
    N = len(gd_t)-1
    
    for t in range(N):
        phi = phi + 1/N * module(gd_t[t], controls_t[t], phi)
        
    return phi

        
def plot_MultiGrid(phi, grid, label):
    '''input: grid = [gridx, gridy]
              phi = [phi1, phi2, phi_background] 
                      final deformation of gridpoints (as a grid)
              label: tensor with labels for each grid point '''
    x = np.zeros(grid[0].shape)
    y = np.zeros(grid[1].shape)
    
    for i in range(grid[0].shape[0]):
        for j in range(grid[0].shape[0]):
            x[i,j] = phi[label[i,j].numpy().astype(int)-1][0][i,j]
            y[i,j] = phi[label[i,j].numpy().astype(int)-1][1][i,j]
            
    plot_grid(x, y, color='blue')
    return x,y



##############################################################

def shootMultishape(gd_list0, mom_list0, h, Constr, sigma, dim, n=10):
    step = 1. / n   
    
    gd = [gd_list0]
    mom = [mom_list0]
    controls = []
    
    gd_list = gd_list0.copy()
    mom_list = mom_list0.copy()

    
    nb_points = [len(gd_list[0]), len(gd_list[1])]

    for i in range(n):  
        z = [computeCenter(gd[-1][0]), computeCenter(gd[-1][1])]
        
        l_list = lambda_qp(gd[-1], mom[-1], sigma, z, dim)
        controls_list = h_qp(gd[-1], mom[-1], l_list, sigma, z, dim)

        [d_gd1, d_gd2, d_gd3, d_mom1, d_mom2, d_mom3] = torch.autograd.grad(h(gd[-1], mom[-1], controls_list, l_list, Constr),[*gd[-1], *mom[-1]], create_graph=True)  # differentiate wrt [gd_list, mom_list] or [gd[-1], mom[-1]]?
        
        gd_list[0] = gd_list[0] + step*d_mom1
        gd_list[1] = gd_list[1] + step*d_mom2
        gd_list[2] = gd_list[2] + step*d_mom3
        mom_list[0] = mom_list[0] - step*d_gd1
        mom_list[1] = mom_list[1] - step*d_gd2
        mom_list[2] = mom_list[2] - step*d_gd3
        
        gd.append([gd_list[0], gd_list[1], gd_list[2]])
        mom.append([mom_list[0], mom_list[1], mom_list[2]])
        controls.append(controls_list)
    controls.append(controls_list)
               
    return gd, mom, controls


##############################################################################    
from kernels import K_xx, K_xy

def h_qp(gd, mom, l, sigma, z, dim):
    sigma1 = sigma[0]
    sigma2 = sigma[1]
    sigma3 = sigma[2]
    z1 = z[0].view(-1, dim).float()
    z2 = z[1].view(-1, dim).float() 

    gd1 = gd[0].view(-1, dim).float()
    K1 = K_xy(z1, gd1, sigma1)

    hqp1 = torch.mm(K1, mom[0] - l[0])
    
    gd2 = gd[1].view(-1, dim).float()
    K2 = K_xy(z2, gd2, sigma2)
   
    hqp2 = torch.mm(K2, mom[1] - l[1])
    
    gd3 = gd[2].view(-1, dim).float()
    K3 = K_xy(gd3, gd3, sigma3)
    
    hqp3 = mom[2]  + torch.cat([l[0], l[1]])
    
    return [hqp1, hqp2, hqp3]


def lambda_qp(gd, mom, sigma, z, dim):
    
    sigma1 = sigma[0]
    sigma2 = sigma[1]
    sigma3 = sigma[2]
    
    nb_points = [len(gd[0]), len(gd[1])]
    
    mom31 = mom[2][0:len(gd[0])]
    mom32 = mom[2][len(gd[0]):len(mom[2])]

    z1 = z[0].view(-1, dim).float()
    gd1 = gd[0].view(-1, dim).float()
    K1 = K_xy(z1, gd1, sigma1)
    
    K11 = torch.mm(torch.transpose(K1,0,1), K1)
    
    z2 = z[1].view(-1, dim).float() 
    gd2 = gd[1].view(-1, dim).float()
    K2 = K_xy(z2, gd2, sigma2)
    
    K22 = torch.mm(torch.transpose(K2,0,1), K2)
   
    gd3 = gd[2].view(-1, dim).float()
    K3 = K_xy(gd3, gd3, sigma3)
    K31 = K3[0:nb_points[0],:]
    K32 = K3[nb_points[0]:,:]
    
    # X = C zeta Z^-1 zeta^ast C^ast
    X = torch.cat([torch.cat([K11, torch.zeros(nb_points[0],nb_points[1])],1), torch.cat([torch.zeros(nb_points[1], nb_points[0]), K22],1)],0) + K3
    
    # A = C zeta Z^-1 zeta^ast xi^ast
    A = torch.cat([torch.cat([K11, torch.zeros(nb_points[0], nb_points[1]), -K31], 1),  torch.cat([torch.zeros(nb_points[1], nb_points[0]), K22, -K32], 1)],0)
    
    lambdaqp, _ = torch.gesv(torch.mm(A,torch.cat([mom[0], mom[1], mom[2]])),X)
    lambdaqp1 = lambdaqp[0:nb_points[0],:]
    lambdaqp2 = lambdaqp[nb_points[0]:,:]

    
    return [lambdaqp1, lambdaqp2]