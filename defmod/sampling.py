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


def sample_from_greyscale(image, threshold, centered=False, normalise_weights=False, normalise_position=True):
    length = torch.sum(image >= threshold)
    points = torch.zeros([length, 2])
    alpha = torch.zeros([length])

    total_weight = torch.sum(points)
    count = 0
    
    width_weight = 1.
    height_weight = 1.
    
    if(normalise_position):
        width_weight = 1./image.shape[0]
        height_weight = 1./image.shape[1]

    # TODO: write a better (i.e. non looping) way of doing this
    for j in range(0, image.shape[1]):
        for i in range(0, image.shape[0]):
            if(image[image.shape[0] - i - 1, j] < threshold):
                continue

            points[count, 0] = i*width_weight
            points[count, 1] = j*height_weight
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


def sample_from_points(points, frame_res):
    u, v = points[0][:, 0], points[0][:, 1]

    u1 = torch.floor(u).long()
    v1 = torch.floor(v).long()

    u1 = torch.clamp(u1, 0, frame_res[0] - 1)
    v1 = torch.clamp(v1, 0, frame_res[1] - 1)
    u2 = torch.clamp(u1 + 1, 0, frame_res[0] - 1)
    v2 = torch.clamp(v1 + 1, 0, frame_res[1] - 1)

    fu = u - u1.type(torch.get_default_dtype())
    fv = v - v1.type(torch.get_default_dtype())
    gu = (u1 + 1).type(torch.get_default_dtype()) - u
    gv = (v1 + 1).type(torch.get_default_dtype()) - v

    intensities = points[1].view(frame_res)
    img_out = (intensities[u1, v1] * gu * gv +
               intensities[u1, v2] * gu * fv +
               intensities[u2, v1] * fu * gv +
               intensities[u2, v2] * fu * fv).view(frame_res)

    return img_out

