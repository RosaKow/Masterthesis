import numpy as np
import torch
import matplotlib.image
import matplotlib.pyplot as plt


def load_greyscale_image(filename):
    """Load grescale image from disk as an array of normalised float values."""
    image = matplotlib.image.imread(filename)
    if(image.ndim == 2):
        return torch.tensor(1. - image)
    elif(image.ndim ==3):
        return torch.tensor(1. - image[:,:,0])
    else:
        raise NotImplementedError


def sample_from_greyscale(image, threshold, centered=False, normalise_weights=False):
    length = torch.sum(image >= threshold)
    points = torch.zeros([length, 2])
    alpha = torch.zeros([length])

    total_weight = torch.sum(points)
    count = 0

    # TODO: write a better (i.e. non looping) way of doing this
    for j in range(0, image.shape[1]):
        for i in range(0, image.shape[0]):
            if(image[image.shape[0] - i - 1, j] < threshold):
                continue

            points[count, 0] = i/image.shape[0]
            points[count, 1] = j/image.shape[1]
            alpha[count] = image[image.shape[0] - i - 1, j]

            count = count + 1

    if(centered):
        points = points - torch.mean(points, dim=0)

    if(normalise_weights):
        alpha = alpha/torch.sum(alpha)

    return points, alpha


def load_and_sample_greyscale(filename, threshold=0., centered=False, normalise_weights=True): 
    image = load_greyscale_image(filename)

    return sample_from_greyscale(image, threshold, centered, normalise_weights)

