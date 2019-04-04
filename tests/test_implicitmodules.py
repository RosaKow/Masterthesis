import unittest

import torch

import defmod as dm

torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(1337)

class TestImplicitModule1_2D(unittest.TestCase):
    def setUp(self):
        self.nb_pts = 7
        self.gd = (torch.rand(self.nb_pts, 2).view(-1), torch.rand(self.nb_pts, 2, 2).view(-1))
        self.tan = (torch.rand(self.nb_pts, 2).view(-1), torch.rand(self.nb_pts, 2, 2).view(-1))
        self.cotan = (torch.rand(self.nb_pts, 2).view(-1), torch.rand(self.nb_pts, 2, 2).view(-1))
        self.stiefel = dm.manifold.Stiefel(2, self.nb_pts, gd=self.gd, tan=self.tan, cotan=self.cotan)
        self.dim_controls = 3
        self.controls = torch.rand(self.dim_controls)
        self.C = torch.rand(self.nb_pts, 2, self.dim_controls)

        self.implicit = dm.implicitmodules.ImplicitModule1(self.stiefel, self.C, self.controls, 0.5, 0.05)

    # def test_compute_aqh(self):
    #     aqh = self.implicit._ImplicitModule1__compute_aqh()

    #     self.assertIsInstance(aqh, torch.Tensor)
    #     self.assertEqual(aqh.shape, torch.Size([self.nb_pts, 2, 2]))
    
