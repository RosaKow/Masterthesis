import unittest

import torch

import defmod as dm

it_count = 10

class TestShooting(unittest.TestCase):
    def setUp(self):
        self.m = 10
        self.trans = dm.deformationmodules.Translations(2, self.m, 0.5)
        self.h = dm.hamiltonian.Hamiltonian(self.trans)

    def test_shooting(self):
        gd = torch.rand(self.m, 2, requires_grad=True)
        mom = torch.rand(self.m, 2, requires_grad=True)
        gd_out, mom_out = dm.shooting.shoot(gd, mom, self.h, it_count)

        self.assertIsInstance(gd_out, torch.Tensor)
        self.assertIsInstance(mom_out, torch.Tensor)

        self.assertEqual(gd.shape, gd_out.shape)
        self.assertEqual(mom.shape, mom_out.shape)

    def test_shooting_zero(self):
        gd = torch.rand(self.m, 2, requires_grad=True)
        mom = torch.zeros_like(gd, requires_grad=True)
        gd_out, mom_out = dm.shooting.shoot(gd, mom, self.h, it_count)

        self.assertEqual(torch.all(torch.eq(gd, gd_out)), True)
        self.assertEqual(torch.all(torch.eq(mom, mom_out)), True)


