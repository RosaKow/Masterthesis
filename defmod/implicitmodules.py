import torch

from .deformationmodules import DeformationModule
from .kernels import K_xx, K_xy

class ImplicitModule0(DeformationModule):
    def __init__(self, dim, nb_pts, sigma, nu):
        self.__dim = dim
        self.__nb_pts = nb_pts
        self.__sigma = sigma
        self.__nu = nu

    @property
    def dim(self):
        return self.__dim

    @property
    def nb_pts(self):
        return self.__nb_pts

    @property
    def sigma(self):
        return self.__sigma

    @property
    def nu(self):
        return self.__nu

    def __call__(self, gd, controls, points) :
        """Applies the generated vector field on given points."""
        cov_mat = K_xy(points, gd.view(-1, self.__dim), self.__sigma)
        return torch.mm(cov_mat, controls.view(-1, self.__dim))

    def action(self, gd, module, gd_module, controls_module) :
        """Applies the vector field generated by Mod on GeoDesc."""
        return module(gd_module, controls_module, gd.view(-1, self.__dim)).view(-1)

    def cost(self, gd, controls):
        K_q = K_xx(gd.view(-1, self.__dim), self.__sigma)
        m = torch.mm(controls.view(self.__dim, -1), K_q + self.__nu*torch.eye(self.__nb_pts))
        return 0.5*torch.dot(m.view(-1), controls.view(-1))

    def compute_geodesic_control(self, delta, gd):
        """Computes geodesic control from \delta \in H^\ast."""
        K_q = K_xx(gd.view(-1, self.__dim), self.__sigma)
        controls, _ = torch.gesv(delta.view(-1, self.__dim), K_q)
        return controls.contiguous().view(-1)

