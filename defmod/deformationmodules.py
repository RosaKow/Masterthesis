import torch
import numpy as np

from .kernels import gauss_kernel, K_xx, K_xy


class DeformationModule(torch.nn.Module):
    """Abstract module."""
    def __init__(self):
        super().__init__()

    def __call__(self, gd, controls, points):
        """Applies the generated vector field on given points."""
        raise NotImplementedError

    def action(self, gd, module, gd_module, controls_module):
        """Applies the vector field generated by Mod on GD"""
        raise NotImplementedError

    def cost(self, gd, controls):
        """Returns the cost."""
        raise NotImplementedError


class Translations(DeformationModule):
    """Module generating sum of translations."""
    def __init__(self, dim, nb_pts, sigma):
        super().__init__()
        self.__sigma = sigma
        self.__dim = dim
        self.__nb_pts = nb_pts
        self.__dim_gd = dim*nb_pts
        self.__dim_controls = dim*nb_pts
        self.__K = gauss_kernel(self.__sigma)

    @property
    def sigma(self):
        return self.__sigma

    @property
    def dim(self):
        return self.__dim

    @property
    def nb_pts(self):
        return self.__nb_pts

    @property
    def dim_gd(self):
        return self.__dim_gd

    @property
    def dim_controls(self):
        return self.__dim_controls

    def __call__(self, gd, controls, points) :
        """Applies the generated vector field on given points."""
        cov_mat = K_xy(points, gd.view(-1, self.__dim), self.__sigma)
        #return torch.mm(cov_mat, controls.view(-1, self.__dim))
        return self.__K(points, gd.view(-1, self.__dim), controls.view(-1, self.__dim))

    def action(self, gd, module, gd_module, controls_module) :
        """Applies the vector field generated by Mod on GeoDesc."""
        return module(gd_module, controls_module, gd.view(-1, self.__dim)).view(-1)

    def cost(self, gd, controls) :
        """Returns the cost."""
        controls_array = controls.view(-1, self.__dim)
        gd_array = gd.view(-1, self.__dim)
        cov_mat = K_xy(gd_array, gd_array, self.__sigma)
        #m = torch.mm(cov_mat, controls_array)
        m = self.__K(gd_array, gd_array, controls_array)
        return 0.5*torch.dot(m.view(-1), controls.view(-1))

    def compute_geodesic_control(self, delta, gd):
        """Computes geodesic control from \delta \in H^\ast."""
        K_q = K_xx(gd.view(-1, self.__dim), self.__sigma)
        controls, _ = torch.gesv(delta.view(-1, self.__dim), K_q)
        return controls.contiguous().view(-1)

    def cot_to_vs(self, gd, mom, sigma, points, j=0):
        if(j is not 0):
            raise NotImplemented("Translations.cot_to_vs() to order", j, "is not implemented!")

        K = K_xy(points, gd.view(-1, self.dim), sigma)

        return torch.mm(K, mom.view(-1, self.dim))

    def apply_adjoint(self, gd, module, gd_module, mom_module):
        return module.cot_to_vs(gd_module, mom_module, self.sigma, gd.view(-1, self.dim)).view(-1)


class SilentPoints(DeformationModule):
    """Module handling silent points."""
    def __init__(self, dim, nb_pts):
        super().__init__()
        self.__dim = dim
        self.__nb_pts = nb_pts
        self.__dim_gd = dim*nb_pts
        self.__dim_controls = 0

    @property
    def dim(self):
        return self.__dim

    @property
    def nb_pts(self):
        return self.__nb_pts

    @property
    def dim_gd(self):
        return self.__dim_gd

    @property
    def dim_controls(self):
        return self.__dim_controls

    def __call__(self, gd, controls, points):
        """Applies the generated vector field on given points."""
        return torch.zeros_like(points)

    def action(self, gd, module, gd_module, controls_module):
        """Applies the vector field generated by Mod on GeoDesc."""
        return module(gd_module, controls_module, gd.view(-1, self.__dim)).view(-1)

    def cost(self, gd, controls):
        """Returns the cost."""
        return torch.tensor(0.)

    def compute_geodesic_control(self, delta, gd):
        """Computes geodesic control from \delta \in H^\ast."""
        return torch.tensor([])

    def cot_to_vs(self, gd, mom, sigma, points, j=0):
        if(j is not 0):
            raise NotImplemented("Translations.cot_to_vs() to order", j, "is not implemented!")

        K = K_xy(points, gd.view(-1, self.dim), sigma)

        return torch.mm(K, mom.view(-1, self.dim))

    def apply_adjoint(self, gd, module, gd_module, mom_module):
        return torch.tensor([])


class Compound(DeformationModule):
    """Combination of modules."""
    def __init__(self, module_list):
        super().__init__()
        self.__module_list = list(module_list)
        self.__nb_module = len(module_list)
        self.__dim_gd = sum([mod.dim_gd for mod in module_list])
        self.__dim_controls = sum([mod.dim_controls for mod in module_list])
        self.__indice_gd = [0]
        self.__indice_gd.extend(np.cumsum([mod.dim_gd for mod in module_list]))
        self.__indice_controls = [0]
        self.__indice_controls.extend(np.cumsum([mod.dim_controls for mod in module_list]))
        self.__nb_pts = sum(mod.nb_pts for mod in module_list)

    @property
    def module_list(self):
        return self.__module_list

    @property
    def nb_module(self):
        return self.__nb_module

    @property
    def dim_gd(self):
        return self.__dim_gd

    @property
    def dim_controls(self):
        return self.__dim_controls

    @property
    def indice_gd(self):
        return self.__indice_gd

    @property
    def indice_controls(self):
        return self.__indice_controls

    @property
    def nb_pts(self):
        return self.__nb_pts

    def __call__(self, gd, controls, points) :
        app_list = []
        for i in range(self.__nb_module):
            app_list.append(self.__module_list[i](
                gd[self.__indice_gd[i]:self.__indice_gd[i+1]],
                controls[self.__indice_controls[i]:self.__indice_controls[i+1]],
                points
            ).unsqueeze(0))

        return torch.sum(torch.cat(app_list), 0)

    def action(self, gd, module, gd_module, controls_module) :
        """Applies the vector field generated by Mod on GeoDesc."""
        app_list = []
        for i in range(self.__nb_module):
            app_list.append(self.__module_list[i].action(
                gd[self.indice_gd[i]:self.indice_gd[i+1]], module,
                gd_module, controls_module)
            )

        return torch.cat(app_list)

    def cost(self, gd, controls) :
        """Returns the cost."""
        cost_list = []
        for i in range(self.__nb_module):
            cost_list.append(self.__module_list[i].cost(
                gd[self.__indice_gd[i]:self.__indice_gd[i+1]],
                controls[self.__indice_controls[i]:self.__indice_controls[i+1]]
            ).unsqueeze(0))
                             
        return torch.sum(torch.cat(cost_list), 0)

    def compute_geodesic_control(self, delta, GeoDesc):
        """Computes geodesic control from \delta \in H^\ast."""
        controls_list = []
        for i in range(self.__nb_module):
            controls_list.append(self.__module_list[i].compute_geodesic_control(
                delta[self.__indice_controls[i]:self.__indice_controls[i+1]],
                GeoDesc[self.indice_gd[i]:self.indice_gd[i+1]])
            )

        return torch.cat(controls_list)

    def cot_to_vs(self, gd, mom, sigma, points, j=0):
        if(j is not 0):
            raise NotImplemented("Compound.cot_to_vs() to order", j, "is not implemented!")

        out = torch.zeros_like(points)

        for i in range(0, self.__nb_module):
            out += self.__module_list[i].cot_to_vs(
                gd[self.__indice_gd[i]:self.__indice_gd[i+1]],
                mom[self.__indice_gd[i]:self.__indice_gd[i+1]],
                sigma, points)

        return out

    def apply_adjoint(self, gd, module, gd_module, mom_module):
        out = torch.zeros_like(gd)

        for i in range(0, self.__nb_module):
            out[self.__indice_controls[i]:self.__indice_controls[i+1]] = \
                self.__module_list[i].apply_adjoint(gd[self.__indice_gd[i]:self.__indice_gd[i+1]],
                                                    module, gd_module, mom_module)

        return out

