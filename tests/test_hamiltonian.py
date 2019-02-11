import unittest

import torch

import defmod as dm

torch.set_default_tensor_type(torch.DoubleTensor)

class TestHamiltonian(unittest.TestCase):
    def setUp(self):
        self.nb_pts = 10
        self.sigma = 0.5
        self.trans = dm.deformationmodules.Translations(2, self.nb_pts, self.sigma)

        self.gd = torch.rand(self.nb_pts, 2)
        self.mom = torch.rand_like(self.gd)
        self.controls = torch.rand_like(self.gd)

        self.h = dm.hamiltonian.Hamiltonian(self.trans)

    def test_good_init(self):        
        self.assertEqual(torch.all(torch.eq(self.h.init_controls,
                                            torch.zeros(self.trans.dim_controls))), True)
        self.assertIsInstance(self.h.def_module, dm.deformationmodules.DeformationModule)

    def test_apply_mom(self):
        self.assertIsInstance(self.h.apply_mom(self.gd, self.mom, self.controls), torch.Tensor)
        self.assertEqual(self.h.apply_mom(self.gd, self.mom, self.controls).shape, torch.Size([]))

    def test_call(self):
        self.assertIsInstance(self.h(self.gd, self.mom, self.controls), torch.Tensor)
        self.assertEqual(self.h(self.gd, self.mom, self.controls).shape, torch.Size([]))

    def test_gradcheck_call(self):
        self.gd.requires_grad_()
        self.mom.requires_grad_()
        self.controls.requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(
            self.h.__call__, (self.gd, self.mom, self.controls), raise_exception=False))

    def test_gradcheck_apply_mom(self):
        self.gd.requires_grad_()
        self.mom.requires_grad_()
        self.controls.requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(
            self.h.apply_mom, (self.gd, self.mom, self.controls), raise_exception=False))

    def test_gradcheck_geodesic_controls(self):
        self.gd.requires_grad_()
        self.mom.requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(self.h.geodesic_controls, (self.gd, self.mom),
                                                raise_exception=False))

