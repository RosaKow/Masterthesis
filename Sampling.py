import numpy as np
import torch
import matplotlib.image
import UsefulFunctions

import matplotlib.pyplot as plt

def loadGreyscaleImage(filename):
    """Load grescale image from disk as an array of normalized float values."""
    img = matplotlib.image.imread(filename)
    if(img.ndim == 2):
        return 1. - img
    elif(img.ndim ==3):
        return 1. - img[:,:,0]
    else:
        raise NotImplementedError

"""
def sampleFromGreyscale(filename, threshold=1e-3, centered=False):
    "Sample from a greyscale image with specified threshold"
    img = loadGreyscaleImage(filename)

    ind = (img[::-1] > threshold).nonzero()
    print(img[np.array(ind).T].shape)

    x = np.array([ind])
    x = np.ascontiguousarray(x.T.reshape(-1, 2))
    x = x / img.shape
    if(centered):
        x = x - np.mean(x, axis=0)
    
    alpha = img[ind[0], ind[1]]
    #alpha = alpha / alpha.sum()
    
    return torch.tensor(x, dtype=torch.get_default_dtype()), torch.tensor(alpha, dtype=torch.get_default_dtype())
"""

def sampleFromGreyscale(filename, threshold=1e-2, centered=False, normaliseWeights=True):
    img = loadGreyscaleImage(filename)

    length = np.sum(img >= threshold)
    x = np.zeros([length, 2])
    alpha = np.zeros([length])

    totalweight = np.sum(x)
    count = 0
    for j in range(0, img.shape[1]):
        for i in range(0, img.shape[0]):
            if(img[img.shape[0] - i - 1, j] < threshold):
                continue

            x[count, 0] = i/img.shape[0]
            x[count, 1] = j/img.shape[1]
            #alpha[count] = img[img.shape[0] - i - 1, j]
            alpha[count] = img[img.shape[0] - i - 1, j]

            count = count + 1

    if(centered):
        x = x - np.mean(x, axis=0)

    if(normaliseWeights):
        alpha = alpha/np.sum(alpha)
    
    return torch.tensor(x, dtype=torch.get_default_dtype()), torch.tensor(alpha, dtype=torch.get_default_dtype())


def gaussianKernel(x, sigma):
    return 1/np.sqrt(2*np.pi)/sigma*np.exp(-0.5*x**2/sigma**2)


def kernelSmoother(x, data, kernel=gaussianKernel, sigma=0.1):
    """Apply a kernel smoother on the values at x."""
    y = torch.zeros(x.shape[0])
    for i in range(0, x.shape[0]):
        y[i] = np.sum(kernel(np.linalg.norm((x[i,:].expand_as(data[0]) - data[0]).numpy(), axis=1), sigma)*data[1].numpy())

    return y


def sampleImageFromPoints(data, res, kernel=gaussianKernel, sigma=0.1, normalize=True, aabb=None):
    """Sample an image from a list of smoothened points."""
    if(aabb is None):
        aabb = UsefulFunctions.AABB()
        aabb.buildFromPoints(data[0])
    x, y = torch.meshgrid([torch.arange(aabb.minx-sigma, aabb.maxx+sigma, step=(aabb.maxx-aabb.minx+2.*sigma)/res[0]), torch.arange(aabb.miny-sigma, aabb.maxy+sigma, step=(aabb.maxy-aabb.miny+2.*sigma)/res[1])])
    vec = UsefulFunctions.grid2vec(x, y)
    pixels = kernelSmoother(vec, data, sigma=sigma)
    if(normalize):
        pixels = pixels/torch.sum(pixels)

    return pixels.view(res[0], res[1])


def sampleFromShape2D(nbPts, shape, K):
    """Simple rejection sampling algorithm with shape a list of points defining the shape, K a kernel. Returns nbPts from the distribution."""
    minx, miny, maxx, maxy
    for i in range(nbPts):
        pass
