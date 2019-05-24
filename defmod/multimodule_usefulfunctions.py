import torch
import math
import numpy as np
import defmod as dm

pi = math.pi

def CirclePoints(origin, r,n):
    points = []
    for x in range (0,n):
        points.append([origin[0] + math.cos(2*pi/n*x)*r,origin[1] + math.sin(2*pi/n*x)*r])
    return torch.tensor(points, requires_grad=True)

def EllipsePoints(origin, a, b, n):
    points = []
    for x in range(0,n):
        points.append([origin[0] + math.cos(2*pi*x/n)*a, origin[1] + math.sin(2*pi*x/n)*b])
    return torch.tensor(points, requires_grad=True)

def multipleCircles(origin, radius, numberPoints):
    circles = []
    for o,r,n in zip(origin, radius, numberPoints):
        circles.append(CirclePoints(o, r, n))
    return circles

def pointInCircle(points, z, r):
    """ returns a list a True / False values for the points belonging to the defined region or not"""
    label = []
    for i in range(len(points)):     
        if (torch.norm(z - points[i,:]) < r):
            label.append(True)
        else:
            label.append(False)
    return label

def list_points_in_region(points, points_in_region):
    """ returns a list of all points in points that belong to the region defined by the points_in_region_function"""
    tmp = np.array(points_in_region(points)).astype(int)
    nb_pts = points.shape[0]
    dim = points.shape[1]
    region_list = points_in_region(points)
    points_list = []
    for i in range(len(points)):
        if region_list[i] == True:
            points_list.append(points[i,:])
    return points_list
    
####################################################
# Plotting
    
def computeCenter(gd):
    if len(gd.shape) == 1:
            gd = gd.unsqueeze(0)
    return torch.mean(gd,0)
        
def largeDeformation(modules, states, controls, points):
    """ compute large deformation of points"""
    phi = [points, points, points]
    N = len(states)-1
    
    for t in range(N):
        modules.manifold.fill(states[t])
        modules.fill_controls(controls[t])
        for i in range(len(modules.module_list)):
            phi[i] = phi[i] + 1/N * modules.module_list[i](phi[i])
            
        
    return phi

import matplotlib.pyplot as plt
def plot_grid(gridx,gridy, **kwargs):
    fig1 = plt.figure()  
    ax = fig1.add_subplot(1, 1, 1) 
    for i in range(gridx.shape[0]):
        ax.plot(gridx[i,:], gridy[i,:], **kwargs)
    for i in range(gridx.shape[1]):
        ax.plot(gridx[:,i], gridy[:,i], **kwargs)
        
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
        

########################################################
    
def pointslist_reshape(pointslist, n):
    l = []
    for x in pointslist:
        if isinstance(x, list): 
            l.append(pointslist_reshape(x,n))
        else:
            l.append(x.view(n))
    return l

def gdlist_reshape(pointslist, n):
    l = [*pointslist_reshape(pointslist[:-1], n), pointslist_reshape(pointslist[-1],n)]
    return l

def gdtensor2list(gdtensor, nb_pts):
    gdlist = []
    gdlist_bg = []
    j = 0
    n = sum(nb_pts)
    for i in range(len(nb_pts)):
        gdlist.append(gdtensor[j:j+nb_pts[i]])
        gdlist_bg.append(gdtensor[n+j:n+j+nb_pts[i]])
        j = j + nb_pts[i]
    return [*gdlist, gdlist_bg]

def gdlist2tensor(gdlist):
    return torch.cat([*gdlist[:-1],*gdlist[-1]],0)
                           
def kronecker_I2(K):
    N = K.shape
    tmp = K.view(-1,1).repeat(1,2).view(N[0],2*N[1]).transpose(1,0).contiguous().view(-1,1).repeat(1,2).view(-1,N[0]*2).transpose(1,0)
    Ktilde = torch.mul(tmp, torch.eye(2).repeat(N))
    return Ktilde

def boollist2tensor(boollist):
    return torch.tensor(np.asarray(boollist).astype(np.uint8))