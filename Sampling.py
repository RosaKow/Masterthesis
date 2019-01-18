import numpy as np
import torch
import matplotlib.image

def loadGreyscaleImage(filename):
    """Load grescale image from disk as an array of normalized float values."""
    img = matplotlib.image.imread(filename)
    return 1. - img[:,:,0].astype(float)

def sampleFromGreyscale(filename, threshold=1e-3):
    """Sample from a greyscale image with specified threshold"""
    img = loadGreyscaleImage(filename)

    ind = (img[::-1] > threshold).nonzero()

    x = np.array([ind])
    x = np.ascontiguousarray(x.T.reshape(-1, 2))
    x = x / img.shape
    
    alpha = img[ind[0], ind[1]]
    alpha = alpha / alpha.sum()
    
    return torch.tensor(x, dtype=torch.float32), torch.tensor(alpha, dtype=torch.float32)

