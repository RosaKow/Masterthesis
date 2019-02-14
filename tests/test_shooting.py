import unittest

import torch

import defmod as dm

from defmod.shooting import shoot

torch.set_default_tensor_type(torch.DoubleTensor)

class TestShooting(unittest.TestCase):
    def setUp(self):
        self.m = 4
        self.trans = dm.deformationmodules.Translations(2, self.m, 0.5)
        self.h = dm.hamiltonian.Hamiltonian(self.trans)
        self.gd = torch.rand(self.m, 2).view(-1)
        self.mom = torch.rand(self.m, 2).view(-1)

    def test_shooting(self):
        gd_out, mom_out = shoot(self.gd, self.mom, self.h)

        self.assertIsInstance(gd_out, torch.Tensor)
        self.assertIsInstance(mom_out, torch.Tensor)

        self.assertEqual(self.gd.shape, gd_out.shape)
        self.assertEqual(self.mom.shape, mom_out.shape)

    def test_shooting_zero(self):
        mom = torch.zeros_like(self.gd, requires_grad=True).view(-1)
        gd_out, mom_out = shoot(self.gd, mom, self.h)

        self.assertTrue(torch.allclose(self.gd, gd_out))
        self.assertTrue(torch.allclose(mom, mom_out))

    def test_shooting_rand(self):
        gd_out, mom_out = shoot(self.gd, self.mom, self.h)

        self.assertFalse(torch.allclose(self.gd, gd_out))
        self.assertFalse(torch.allclose(self.mom, mom_out))

    def test_gradcheck_shoot(self):
        self.gd.requires_grad_()
        self.mom.requires_grad_()

        # We multiply GD by 100. as it seems gradcheck is very sensitive to
        # badly conditioned problems
        self.assertTrue(torch.autograd.gradcheck(shoot,
            (100.*self.gd, self.mom, self.h), raise_exception=False))


