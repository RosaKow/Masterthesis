import torch
import math
import numpy as np
import defmod as dm

pi = math.pi

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