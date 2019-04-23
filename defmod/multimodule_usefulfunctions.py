import torch
import math
import numpy as np
import defmod as dm

pi = math.pi

def CirclePoints(origin, r,n):
    points = []
    for x in range (0,n):
        points.append([origin[0] + math.cos(2*pi/n*x)*r,origin[1] + math.sin(2*pi/n*x)*r])
    return torch.tensor(points)

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
    
    
def pointslist_reshape(pointslist, n):
    return [p.view(n) for p in pointslist]

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