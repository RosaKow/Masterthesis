import copy
import itertools
from collections import Iterable

import torch
import numpy as np

from .structuredfield import StructuredField_0, StructuredField_m, CompoundStructuredField, StructuredField_Null


class Manifold:
    def __init__(self):
        super().__init__()

    # We would idealy use deepcopy but only graph leaves Tensors supports it right now
    def copy(self):
        raise NotImplementedError

    @property
    def numel_gd(self):
        raise NotImplementedError

    def action(self, module):
        raise NotImplementedError


class Landmarks(Manifold):
    def __init__(self, dim, nb_pts, gd=None, tan=None, cotan=None):
        assert (gd is None) or (gd.shape[0] == dim*nb_pts)
        assert (tan is None) or (tan.shape[0] == dim*nb_pts)
        assert (cotan is None) or (cotan.shape[0] == dim*nb_pts)
        super().__init__()
        
        self.__nb_pts = nb_pts
        self.__dim = dim
        self.__numel_gd = nb_pts*dim

        self.__gd = torch.zeros(nb_pts, dim, requires_grad=True).view(-1)
        if isinstance(gd, torch.Tensor):
            self.fill_gd(gd.requires_grad_(), copy=False)

        self.__tan = torch.zeros(nb_pts, dim, requires_grad=True).view(-1)
        if isinstance(tan, torch.Tensor):
            self.fill_tan(tan.requires_grad_(), copy=False)

        self.__cotan = torch.zeros(nb_pts, dim, requires_grad=True).view(-1)
        if isinstance(cotan, torch.Tensor):
            self.fill_cotan(cotan.requires_grad_(), copy=False)

    def copy(self, retain_grad=False):
        out = Landmarks(self.__dim, self.__nb_pts)
        out.fill(self, copy=True, retain_grad=retain_grad)
        return out

    @property
    def nb_pts(self):
        return self.__nb_pts

    @property
    def dim(self):
        return self.__dim

    @property
    def numel_gd(self):
        return self.__numel_gd

    @property
    def len_gd(self):
        return 1

    @property
    def dim_gd(self):
        return (self.__numel_gd,)

    def unroll_gd(self):
        return [self.__gd]

    def unroll_tan(self):
        return [self.__tan]
    
    def gd_points(self):
        return self.gd.view(-1, self.dim)
    
    def unroll_gd_points(self):
        return [self.gd_points()]

    def unroll_cotan(self):
        return [self.__cotan]

    def roll_gd(self, l):
        return l.copy().pop(0)

    def roll_tan(self, l):
        return l.copy().pop(0)

    def roll_cotan(self, l):
        return l.copy().pop(0)

    def __get_gd(self):
        return self.__gd

    def __get_tan(self):
        return self.__tan
 
    def __get_cotan(self):
        return self.__cotan

    def fill(self, manifold, copy=False, retain_grad=False):
        assert isinstance(manifold, Landmarks)
        self.fill_gd(manifold.gd, copy=copy, retain_grad=retain_grad)
        self.fill_tan(manifold.tan, copy=copy, retain_grad=retain_grad)
        self.fill_cotan(manifold.cotan, copy=copy, retain_grad=retain_grad)

    def fill_gd(self, gd, copy=False, retain_grad=False):
        assert gd.shape[0] == self.__numel_gd
        if copy:
            self.__gd = gd.detach().clone().requires_grad_()
            if retain_grad:
                self.__gd = gd.clone().requires_grad_()
        else:
            self.__gd = gd.requires_grad_()

    def fill_tan(self, tan, copy=False, retain_grad=False):
        assert tan.shape[0] == self.__numel_gd
        if copy:
            self.__tan = tan.detach().clone().requires_grad_()
            if retain_grad:
                self.__tan = tan.clone().requires_grad_()
        else:
            self.__tan = tan.requires_grad_()

    def fill_cotan(self, cotan, copy=False, retain_grad=False):
        assert cotan.shape[0] == self.__numel_gd
        if copy:
            self.__cotan = cotan.detach().clone().requires_grad_()
            if retain_grad:
                self.__cotan = cotan.clone().requires_grad_()
        else:
            self.__cotan = cotan.requires_grad_()

    gd = property(__get_gd, fill_gd)
    tan = property(__get_tan, fill_tan)
    cotan = property(__get_cotan, fill_cotan)

    def muladd_gd(self, gd, scale):
        if isinstance(gd, torch.Tensor):
            self.__gd = self.__gd + scale*gd
        else:
            self.__gd = self.__gd + scale*gd.__gd

    def muladd_tan(self, tan, scale):
        if isinstance(tan, torch.Tensor):
            self.__tan = self.__tan + scale*tan
        else:
            self.__tan = self.__tan + scale*tan.__tan

    def muladd_cotan(self, cotan, scale):
        if isinstance(cotan, torch.Tensor):
            self.__cotan = self.__cotan + scale*cotan
        else:
            self.__cotan = self.__cotan + scale*cotan.__cotan

    def negate_gd(self):
        self.__gd = -self.__gd

    def negate_tan(self):
        self.__tan = -self.__tan

    def negate_cotan(self):
        self.__cotan = -self.__cotan

    def inner_prod_field(self, field):
        man = self.action(field)
        return torch.dot(self.__cotan.view(-1), man.tan.view(-1))

    def action(self, field) :
        """Applies the vector field generated by the module on the landmark."""
        tan = field(self.__gd.view(-1, self.__dim)).view(-1)
        return Landmarks(self.__dim, self.__nb_pts, gd=self.__gd, tan=tan)

    def cot_to_vs(self, sigma):
        return StructuredField_0(self.__gd.view(-1, self.__dim),
                                 self.__cotan.view(-1, self.__dim), sigma)


class Stiefel(Manifold):
    def __init__(self, dim, nb_pts, gd=None, tan=None, cotan=None):
        assert (gd is None) or (isinstance(gd, Iterable) and (len(gd) == 2))
        assert (tan is None) or (isinstance(tan, Iterable) and (len(tan) == 2))
        assert (cotan is None) or (isinstance(cotan, Iterable) and (len(cotan) == 2))
        assert (gd is None) or ((gd[0].shape[0] == dim*nb_pts) and (gd[1].shape[0] == dim*dim*nb_pts))
        assert (tan is None) or ((tan[0].shape[0] == dim*nb_pts) and (tan[1].shape[0] == dim*dim*nb_pts))
        assert (cotan is None) or ((cotan[0].shape[0] == dim*nb_pts) and (cotan[1].shape[0] == dim*dim*nb_pts))

        self.__dim = dim
        self.__nb_pts = nb_pts

        self.__point_shape = torch.Size([self.__nb_pts, self.__dim])
        self.__mat_shape = torch.Size([self.__nb_pts, self.__dim, self.__dim])

        self.__numel_gd_points = self.__nb_pts * self.__dim
        self.__numel_gd_mat = self.__nb_pts * self.__dim * self.__dim
        self.__numel_gd = self.__numel_gd_points + self.__numel_gd_mat

        if gd is not None:
            self.fill_gd(gd, copy=False)
        else:
            self.__gd = (torch.zeros(self.__numel_gd_points, requires_grad=True),
                         torch.zeros(self.__numel_gd_mat, requires_grad=True))

        if tan is not None:
            self.fill_tan(tan, copy=False)
        else:
            self.__tan = (torch.zeros(self.__numel_gd_points, requires_grad=True),
                          torch.zeros(self.__numel_gd_mat, requires_grad=True))

        if cotan is not None:
            self.fill_cotan(cotan, copy=False)
        else:
            self.__cotan = (torch.zeros(self.__numel_gd_points, requires_grad=True),
                            torch.zeros(self.__numel_gd_mat, requires_grad=True))

    def copy(self):
        out = Stiefel(self.__dim, self.__nb_pts)
        out.fill(self, copy=True)
        return out

    @property
    def nb_pts(self):
        return self.__nb_pts

    @property
    def dim(self):
        return self.__dim

    @property
    def numel_gd(self):
        return self.__numel_gd

    @property
    def len_gd(self):
        return 2

    @property
    def dim_gd(self):
        return (self.__numel_gd_points, self.__numel_gd_mat)

    def unroll_gd(self):
        return [self.__gd[0], self.__gd[1]]

    def unroll_tan(self):
        return [self.__tan[0], self.__tan[1]]

    def unroll_cotan(self):
        return [self.__cotan[0], self.__cotan[1]]

    def roll_gd(self, l):
        return [l.pop(0), l.pop(0)]

    def roll_tan(self, l):
        return [l.pop(0), l.pop(0)]

    def roll_cotan(self, l):
        return [l.pop(0), l.pop(0)]

    def __get_gd(self):
        return self.__gd

    def __get_tan(self):
        return self.__tan

    def __get_cotan(self):
        return self.__cotan

    def fill(self, manifold, copy=False,retain_grad=False):
        assert isinstance(manifold, Stiefel)
        self.fill_gd(manifold.gd, copy=copy, retain_grad=retain_grad)
        self.fill_tan(manifold.tan, copy=copy, retain_grad=retain_grad)
        self.fill_cotan(manifold.cotan, copy=copy, retain_grad=retain_grad)

    def fill_gd(self, gd, copy=False):
        assert isinstance(gd, Iterable) and (len(gd) == 2) and (gd[0].numel() == self.__numel_gd_points) and (gd[1].numel() == self.__numel_gd_mat)
        if not copy:
            self.__gd = gd
        else:
            self.__gd = (gd[0].detach().clone().requires_grad_(),
                         gd[1].detach().clone().requires_grad_())


    def fill_tan(self, tan, copy=False):
        assert isinstance(tan, Iterable) and (len(tan) == 2) and (tan[0].numel() == self.__numel_gd_points) and (tan[1].numel() == self.__numel_gd_mat)
        if not copy:
            self.__tan = tan
        else:
            self.__tan = (tan[0].detach().clone().requires_grad_(),
                          tan[1].detach().clone().requires_grad_())


    def fill_cotan(self, cotan, copy=False):
        assert isinstance(cotan, Iterable) and (len(cotan) == 2) and (cotan[0].numel() == self.__numel_gd_points) and (cotan[1].numel() == self.__numel_gd_mat)
        if not copy:
            self.__cotan = cotan
        else:
            self.__cotan = (cotan[0].detach().clone().requires_grad_(),
                            cotan[1].detach().clone().requires_grad_())


    gd = property(__get_gd, fill_gd)
    tan = property(__get_tan, fill_tan)
    cotan = property(__get_cotan, fill_cotan)

    def muladd_gd(self, gd, scale):
        self.__gd = (self.__gd[0] + scale*gd[0], self.__gd[1] + scale*gd[1])

    def muladd_tan(self, tan, scale):
        self.__tan = (self.__tan[0] + scale*tan[0], self.__tan[1] + scale*tan[1])

    def muladd_cotan(self, cotan, scale):
        self.__cotan = (self.__cotan[0] + scale*cotan[0], self.__cotan[1] + scale*cotan[1])

    def negate_gd(self):
        self.__gd = (-self.__gd[0], -self.__gd[1])

    def negate_tan(self):
        self.__tan = (-self.__tan[0], -self.__tan[1])

    def negate_cotan(self):
        self.__cotan = (-self.__cotan[0], -self.__cotan[1])

    def cot_to_vs(self, sigma):
        v0 = StructuredField_0(self.__gd[0].view(-1, self.__dim), self.__cotan[0].view(-1, self.__dim), sigma)
        # TODO: Remove this loop
        #R = torch.stack([torch.mm(self.__cotan[1].view(-1, self.__dim, self.__dim)[i], self.__gd[1].view(-1, self.__dim, self.__dim)[i].t()) for i in range(self.nb_pts)])
        R = torch.einsum('nik, njk->nij', self.__cotan[1].view(-1, self.__dim, self.__dim), self.__gd[1].view(-1, self.__dim, self.__dim))
        vm = StructuredField_m(self.__gd[0].view(-1, self.__dim), R, sigma)

        return CompoundStructuredField([v0, vm])

    def inner_prod_field(self, field):
        man = self.action(field)

        # TODO: Remove this loop
        # out = torch.dot(self.cotan[0].view(-1), man.tan[0].view(-1))
        # return out + sum([torch.tensordot(self.cotan[1].view(-1, self.__dim, self.__dim)[i], man.tan[1].view(-1, self.__dim, self.__dim)[i]) for i in range(self.__mat_shape[0])])
        return torch.dot(self.cotan[0].view(-1), man.tan[0].view(-1)) + torch.einsum('nij, nij->', self.cotan[1].view(-1, self.__dim, self.__dim), man.tan[1].view(-1, self.__dim, self.__dim))

    def action(self, field):
        """Applies the vector field generated by the module on the landmark."""
        vx = field(self.__gd[0].view(-1, self.__dim))
        d_vx = field(self.__gd[0].view(-1, self.__dim), k=1)

        S = 0.5 * (d_vx - torch.transpose(d_vx, 1, 2))
        # TODO: Remove this loop
        #vr = torch.stack([torch.mm(S[i], self.__gd[1].view(-1, self.__dim, self.__dim)[i]) for i in range(self.__nb_pts)])
        vr = torch.einsum('nik, nkj->nij', S, self.__gd[1].view(-1, self.__dim, self.__dim))

        return Stiefel(self.__dim, self.__nb_pts, gd=self.__gd, tan=(vx.view(-1), vr.view(-1)))


class CompoundManifold(Manifold):
    def __init__(self, manifold_list):
        super().__init__()
        self.__manifold_list = manifold_list
        self.__dim = self.__manifold_list[0].dim
        self.__nb_pts = sum([m.nb_pts for m in self.__manifold_list])
        self.__numel_gd = sum([m.numel_gd for m in self.__manifold_list])
        self.__len_gd = sum([m.len_gd for m in self.__manifold_list])
        self.__dim_gd = tuple(sum((m.dim_gd for m in self.__manifold_list), ()))

    def copy(self, retain_grad=False):
        manifold_list = []
        for i in range(len(self.__manifold_list)):
            manifold_list.append(self.manifold_list[i].copy(retain_grad))
        #manifold_list = [m.copy() for m in self.__manifold_list]
        return CompoundManifold(manifold_list)

    @property
    def manifold_list(self):
        return self.__manifold_list

    @property
    def nb_manifold(self):
        return len(self.__manifold_list)

    def __getitem__(self, index):
        return self.__manifold_list[index]

    @property
    def dim(self):
        return self.__dim

    @property
    def nb_pts(self):
        return self.__nb_pts

    @property
    def numel_gd(self):
        return self.__numel_gd

    @property
    def len_gd(self):
        return self.__len_gd

    @property
    def dim_gd(self):
        return self.__dim_gd

    def unroll_gd(self):
        """Returns a flattened list of all gd tensors."""
        l = []
        for man in self.__manifold_list:
            l.extend(man.unroll_gd())
        return l

    def unroll_cotan(self):
        l = []
        for man in self.__manifold_list:
            l.extend(man.unroll_cotan())
        return l
    
    def unroll_tan(self):
        l = []
        for man in self.__manifold_list:
            l.extend(man.unroll_tan())
        return l

    def roll_gd(self, l):
        """Unflattens the list into one suitable for fill_gd() or all *_gd() numerical operations."""
        out = []
        for man in self.__manifold_list:
            out.append(man.roll_gd(l))
        return out
    
    def gd_points(self):
        return [man.gd_points() for man in self.manifold_list]
    
    def unroll_gd_points(self):
        l = []
        for man in self.__manifold_list:
            l.extend(man.unroll_gd_points())
        return l

    def roll_cotan(self, l):
        out = []
        for man in self.__manifold_list:
            out.append(man.roll_cotan(l))
        return out

    def __get_gd(self):
        return [m.gd for m in self.__manifold_list]

    def __get_tan(self):
        return [m.tan for m in self.__manifold_list]

    def __get_cotan(self):
        return [m.cotan for m in self.__manifold_list]

    def fill(self, manifold, copy=False, retain_grad=False):
        self.fill_gd(manifold.gd, copy=copy, retain_grad=retain_grad)
        self.fill_tan(manifold.tan, copy=copy, retain_grad=retain_grad)
        self.fill_cotan(manifold.cotan, copy=copy, retain_grad=retain_grad)

    def fill_gd(self, gd, copy=False, retain_grad=False):
        if isinstance(gd, torch.DoubleTensor):
            l = []
            n = 0
            for i in range(self.nb_manifold):
                m = self.manifold_list[i].numel_gd
                l.append(gd[n:n+m])
                n = n + m
            gd = l
        for i in range(len(self.__manifold_list)):
            self.__manifold_list[i].fill_gd(gd[i], copy=copy, retain_grad=retain_grad)

    def fill_tan(self, tan, copy=False, retain_grad=False):
        for i in range(len(self.__manifold_list)):
            self.__manifold_list[i].fill_tan(tan[i], copy=copy, retain_grad=retain_grad)

    def fill_cotan(self, cotan, copy=False, retain_grad=False):
        if isinstance(cotan, torch.DoubleTensor):
            l = []
            n = 0
            for i in range(self.nb_manifold):
                m = self.manifold_list[i].numel_gd
                l.append(cotan[n:n+m])
                n = n + m
            cotan = l
        for i in range(len(self.__manifold_list)):
            self.__manifold_list[i].fill_cotan(cotan[i], copy=copy, retain_grad=retain_grad)
           

    gd = property(__get_gd, fill_gd)
    tan = property(__get_tan, fill_tan)
    cotan = property(__get_cotan, fill_cotan)

    def muladd_gd(self, gd, scale):
        for i in range(len(self.__manifold_list)):
            self.__manifold_list[i].muladd_gd(gd[i], scale)

    def muladd_tan(self, tan, scale):
        for i in range(len(self.__manifold_list)):
            self.__manifold_list[i].muladd_tan(tan[i], scale)

    def muladd_cotan(self, cotan, scale):
        for i in range(len(self.__manifold_list)):
            self.__manifold_list[i].muladd_cotan(cotan[i], scale)

    def negate_gd(self):
        for m in self.__manifold_list:
            m.negate_gd()

    def negate_tan(self):
        for m in self.__manifold_list:
            m.negate_tan()

    def negate_cotan(self):
        for m in self.__manifold_list:
            m.negate_cotan()

    def cot_to_vs(self, sigma):
        return CompoundStructuredField([m.cot_to_vs(sigma) for m in self.__manifold_list])

    def inner_prod_field(self, field):
        return sum([m.inner_prod_field(field) for m in self.__manifold_list])


    def action(self, field):
        actions = []
        for m in self.__manifold_list:
            actions.append(m.action(field))

        return CompoundManifold(actions)

