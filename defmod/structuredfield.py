import torch
import numpy as np

from .kernels import gauss_kernel, rel_differences, K_xy

class StructuredField:
    def __init__(self, support, moments):
        self.fill(support, moments)

    @property
    def support(self):
        return self.__support

    @property
    def moments(self):
        return self.__moments

    def fill(self, support, moments):
        self.__support = support
        self.__moments = moments

    def __call__(self, points, k=0):
        raise NotImplementedError


class StructuredField_Null(StructuredField):
    def __init__(self):
        super().__init__(None, None)

    def __call__(self, points, k=0):
        return torch.zeros([points.shape[0]] + [2]*(k+1))


class StructuredField_0(StructuredField):
    def __init__(self, support, moments, sigma):
        super().__init__(support, moments)
        self.__sigma = sigma

    @property
    def sigma(self):
        return self.__sigma

    def __call__(self, points, k=0):
        ker_vec = gauss_kernel(rel_differences(points, self.support), k, self.__sigma)
        ker_vec = ker_vec.reshape((points.shape[0], self.support.shape[0]) + tuple(ker_vec.shape[1:]))
        return torch.tensordot(torch.transpose(torch.tensordot(torch.eye(2), ker_vec, dims=0), 0, 2), self.moments, dims=([2, 3], [1, 0]))


class StructuredField_p(StructuredField):
    def __init__(self, support, moments, sigma):
        super().__init__(support, moments)
        self.__sigma = sigma

    @property
    def sigma(self):
        return self.__sigma

    def __call__(self, points, k=0):
        P = (self.moments + torch.transpose(self.moments, 1, 2))/2
        ker_vec = -gauss_kernel(rel_differences(points, self.support), k + 1, self.__sigma)
        ker_vec = ker_vec.reshape((points.shape[0], self.support.shape[0]) + tuple(ker_vec.shape[1:]))
        return torch.tensordot(torch.transpose(torch.tensordot(torch.eye(2), ker_vec, dims=0), 0, 2), P, dims=([2, 3, 4], [1, 0, 2]))


class StructuredField_m(StructuredField):
    def __init__(self, support, moments, sigma):
        super().__init__(support, moments)
        self.__sigma = sigma

    @property
    def sigma(self):
        return self.__sigma

    def __call__(self, points, k=0):
        P = (self.moments - torch.transpose(self.moments, 1, 2))/2
        ker_vec = -gauss_kernel(rel_differences(points, self.support), k + 1, self.__sigma)
        ker_vec = ker_vec.reshape((points.shape[0], self.support.shape[0]) + tuple(ker_vec.shape[1:]))
        return torch.tensordot(torch.transpose(torch.tensordot(torch.eye(2), ker_vec, dims=0), 0, 2), P, dims=([2, 3, 4], [1, 0, 2]))


class CompoundStructuredField(StructuredField):
    def __init__(self, fields):
        super().__init__(None, None) # TODO: find a nice way to handle this
        self.__fields = fields

    @property
    def fields(self):
        return self.__fields

    @property
    def nb_field(self):
        return len(self.__fields)

    def __getitem__(self, index):
        return self.__fields

    def __call__(self, points, k=0):
        if type(points)==list:
            return [sum([field(p, k) for field in self.__fields]) for p in points]
        else:
            return sum([field(points, k) for field in self.__fields])

    
class StructuredField_multi(StructuredField):
    def __init__(self, fields, points_in_region):
        # TODO: assert that regions are not overlapping
        super().__init__(None, None)
        self.__fields = fields
        self.__nb_fields = len(fields)
        self.__points_in_region = points_in_region
        
    @property
    def fields(self):
        return self.__fields

    @property
    def nb_field(self):
        return len(self.__fields)

    def __getitem__(self, index):
        return self.__fields

    def __call__(self, points, k=0):
        multifield = torch.zeros(points.shape)
        for i in range(len(self.__points_in_region)):
            label = torch.tensor(np.array(self.__points_in_region[i](points)).astype(int))
            field = self.__fields[i](points)
            field = torch.mul(field, torch.ger(label,torch.ones(points.shape[1]).long()).float())
            multifield = multifield + field
        return multifield

