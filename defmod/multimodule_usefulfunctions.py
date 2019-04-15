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
    label = []
    for i in range(len(points)):     
        if (torch.norm(z - points[i,:]) < r):
            label.append(True)
        else:
            label.append(False)
    return label


def boollist2tensor(boollist):
    return torch.tensor(np.asarray(boollist).astype(np.uint8))