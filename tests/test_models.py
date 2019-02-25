import unittest

import torch

import defmod as dm


class TestModelCompound(unittest.TestCase):
    def setUp(self):
        self.sigma = 1.

        self.k0 = 5
        self.translation_gd0 = torch.rand(self.k0, 2).view(-1)
        self.translation0 = dm.deformationmodules.Translations(2, self.k0, self.sigma)

        self.k1 = 10
        self.translation_gd1 = torch.rand(self.k1, 2).view(-1)
        self.translation1 = dm.deformationmodules.Translations(2, self.k1, self.sigma)

        self.model = dm.models.ModelCompound(
            2, [self.translation0, self.translation1],
            [self.translation_gd0, self.translation_gd1], [True, False])

    def test_get_var(self):
        gd0, mom0 = self.model.get_var_tensor()
        self.assertEqual(len(gd0.shape), 1)
        self.assertEqual(len(mom0.shape), 1)
        self.assertEqual(gd0.shape[0], 2*self.k0 + 2*self.k1)
        self.assertEqual(mom0.shape[0], 2*self.k0 + 2*self.k1)

        gd1, mom1 = self.model.get_var_list()
        self.assertEqual(len(gd1[0].shape), 1)
        self.assertEqual(len(gd1[1].shape), 1)
        self.assertEqual(len(mom1[0].shape), 1)
        self.assertEqual(len(mom1[1].shape), 1)
        self.assertEqual(gd1[0].shape[0], 2*self.k0)
        self.assertEqual(gd1[1].shape[0], 2*self.k1)
        self.assertEqual(mom1[0].shape[0], 2*self.k0)
        self.assertEqual(mom1[1].shape[0], 2*self.k1)

    def test_shoot_tensor(self):
        gd0, mom0 = self.model.get_var_tensor()
        gd0_out, mom0_out = self.model.shoot_tensor()
        self.assertEqual(gd0_out[-1].shape, gd0.shape)
        self.assertEqual(mom0_out[-1].shape, mom0.shape)

    def test_shoot_list(self):
        gd1, mom1 = self.model.get_var_list()
        gd1_out, mom1_out = self.model.shoot_list()
        self.assertEqual(gd1_out[-1][0].shape, gd1[0].shape)
        self.assertEqual(gd1_out[-1][1].shape, gd1[1].shape)
        self.assertEqual(mom1_out[-1][0].shape, mom1[0].shape)
        self.assertEqual(mom1_out[-1][1].shape, mom1[1].shape)

    def test_compute_deformation_grid(self):
        pass


class TestModelCompoundWithPointsRegistration(unittest.TestCase):
    def setUp(self):
        self.n = 10
        self.source = torch.rand(self.n, 2), torch.rand(self.n)
        self.sigma = 1.

        self.m = 15
        self.target = torch.rand(self.m, 2), torch.rand(self.m)

        self.k0 = 15
        self.translation_gd0 = torch.rand(self.k0, 2).view(-1)
        self.translation0 = dm.deformationmodules.Translations(2, self.k0, self.sigma)

        self.k1 = 20
        self.translation_gd1 = torch.rand(self.k1, 2).view(-1)
        self.translation1 = dm.deformationmodules.Translations(2, self.k1, self.sigma)

        self.model = dm.models.ModelCompoundWithPointsRegistration(
            2, self.source, [self.translation0, self.translation1],
            [self.translation_gd0, self.translation_gd1], [True, False])

    def test_fidelity(self):
        fidelity = self.model.fidelity(self.target)
        self.assertIsInstance(fidelity, torch.Tensor)
        self.assertEqual(fidelity.shape, torch.Size([]))

    def test_get_var(self):
        gd0, mom0 = self.model.get_var_tensor()
        self.assertEqual(len(gd0.shape), 1)
        self.assertEqual(len(mom0.shape), 1)
        self.assertEqual(gd0.shape[0], 2*self.n + 2*self.k0 + 2*self.k1)
        self.assertEqual(mom0.shape[0], 2*self.n + 2*self.k0 + 2*self.k1)

        gd1, mom1 = self.model.get_var_list()
        self.assertEqual(len(gd1[0].shape), 1)
        self.assertEqual(len(gd1[1].shape), 1)
        self.assertEqual(len(mom1[0].shape), 1)
        self.assertEqual(len(mom1[1].shape), 1)
        self.assertEqual(gd1[0].shape[0], 2*self.n)
        self.assertEqual(gd1[1].shape[0], 2*self.k0)
        self.assertEqual(gd1[2].shape[0], 2*self.k1)
        self.assertEqual(mom1[0].shape[0], 2*self.n)
        self.assertEqual(mom1[1].shape[0], 2*self.k0)
        self.assertEqual(mom1[2].shape[0], 2*self.k1)

    def test_shoot_tensor(self):
        gd0, mom0 = self.model.get_var_tensor()
        gd0_out, mom0_out = self.model.shoot_tensor()
        self.assertEqual(gd0_out[-1].shape, gd0.shape)
        self.assertEqual(mom0_out[-1].shape, mom0.shape)

    def test_shoot_list(self):
        gd1, mom1 = self.model.get_var_list()
        gd1_out, mom1_out = self.model.shoot_list()
        self.assertEqual(gd1_out[-1][0].shape, gd1[0].shape)
        self.assertEqual(gd1_out[-1][1].shape, gd1[1].shape)
        self.assertEqual(mom1_out[-1][0].shape, mom1[0].shape)
        self.assertEqual(mom1_out[-1][1].shape, mom1[1].shape)

