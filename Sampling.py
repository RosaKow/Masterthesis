import numpy as np
import torch
import matplotlib.image

def loadGreyscaleImage(filename):
    """Load grescale image from disk as an array of normalized float values."""
    img = matplotlib.image.imread(filename)
    return 1. - img[:,:,0].astype(np.float64)

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
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if(img[j, i] < threshold):
                continue
            
            x[count, 1] = i/img.shape[1]
            x[count, 0] = j/img.shape[0]
            alpha[count] = img[j, i]

            count = count + 1

    if(centered):
        x = x - np.mean(x, axis=0)

    if(normaliseWeights):
        alpha = alpha/np.sum(alpha)
    
    return torch.tensor(x, dtype=torch.get_default_dtype()), torch.tensor(alpha, dtype=torch.get_default_dtype())
    
