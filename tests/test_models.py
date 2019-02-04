import unittest

import torch

import defmod as dm


class TestModelTranslationModuleRegistration2D(unittest.TestCase):
    def setUp(self):
        self.n = 10
        self.source = torch.rand(self.n, 2), torch.rand(self.n)
        self.sigma = 1.

        self.m = 15
        self.target = torch.rand(self.m, 2), torch.rand(self.m)
        
        self.k = 5
        self.translation_gd = torch.rand(self.k, 2).view(-1)

    def test_call(self):
        model = dm.models.ModelTranslationModuleRegistration(2, self.source, self.sigma,
                                                             self.translation_gd, True)

        out = model()
        self.assertIsInstance(out, tuple)
        self.assertIsInstance(out[0], torch.Tensor)
        self.assertIsInstance(out[1], torch.Tensor)
        
        # Since there is no training, the output should be equal to the input.
        self.assertTrue(torch.all(torch.eq(out[0], self.source[0])))
        self.assertTrue(torch.all(torch.eq(out[1], self.source[1])))

    def test_fidelity(self):
        model = dm.models.ModelTranslationModuleRegistration(2, self.source, self.sigma,
                                                             self.translation_gd, True)
        fidelity = model.fidelity(self.target)
        self.assertIsInstance(fidelity, torch.Tensor)
        self.assertEqual(fidelity.shape, torch.Size([]))

    def test_get_var(self):
        model = dm.models.ModelTranslationModuleRegistration(2, self.source, self.sigma,
                                                             self.translation_gd, True)
        gd0, mom0 = model.get_var_tensor()
        self.assertEqual(len(gd0.shape), 1)
        self.assertEqual(len(mom0.shape), 1)
        self.assertEqual(gd0.shape[0], 2*self.n + 2*self.k)
        self.assertEqual(mom0.shape[0], 2*self.n + 2*self.k)

        gd1, mom1 = model.get_var_list()
        self.assertEqual(len(gd1[0].shape), 1)
        self.assertEqual(len(gd1[1].shape), 1)
        self.assertEqual(len(mom1[0].shape), 1)
        self.assertEqual(len(mom1[1].shape), 1)
        self.assertEqual(gd1[0].shape[0], 2*self.n)
        self.assertEqual(gd1[1].shape[0], 2*self.k)
        self.assertEqual(mom1[0].shape[0], 2*self.n)
        self.assertEqual(mom1[1].shape[0], 2*self.k)

    def test_shoot_tensor(self):
        model = dm.models.ModelTranslationModuleRegistration(2, self.source, self.sigma,
                                                             self.translation_gd, True)
        gd0, mom0 = model.get_var_tensor()
        gd0_out, mom0_out = model.shoot_tensor(it=5)
        self.assertEqual(gd0_out.shape, gd0.shape)
        self.assertEqual(mom0_out.shape, mom0.shape)

    def test_shoot_list(self):
        model = dm.models.ModelTranslationModuleRegistration(2, self.source, self.sigma,
                                                             self.translation_gd, True)
        
        gd1, mom1 = model.get_var_list()
        gd1_out, mom1_out = model.shoot_list(it=5)
        self.assertEqual(gd1_out[0].shape, gd1[0].shape)
        self.assertEqual(gd1_out[1].shape, gd1[1].shape)
        self.assertEqual(mom1_out[0].shape, mom1[0].shape)
        self.assertEqual(mom1_out[1].shape, mom1[1].shape)


class TestModelCompoundRegistration2D(unittest.TestCase):
    def setUp(self):
        self.n = 10
        self.source = torch.rand(self.n, 2), torch.rand(self.n)

        self.m = 15
        self.target = torch.rand(self.m, 2), torch.rand(self.m)
        
        self.k0 = 5
        self.sigma = 1.
        self.translation_gd0 = torch.rand(self.k0, 2).view(-1)
        self.translation0 = dm.deformationmodules.Translations(2, self.k0, self.sigma)

        self.k1 = 10
        self.translation_gd1 = torch.rand(self.k1, 2).view(-1)
        self.translation1 = dm.deformationmodules.Translations(2, self.k1, self.sigma)

    def test_call(self):
        model = dm.models.ModelCompoundRegistration(2, self.source,
                                                    [self.translation0, self.translation1],
                                                    [self.translation_gd0, self.translation_gd1],
                                                    [True, False])

        out = model()
        self.assertIsInstance(out, tuple)
        self.assertIsInstance(out[0], torch.Tensor)
        self.assertIsInstance(out[1], torch.Tensor)
        
        # Since there is no training, the output should be equal to the input.
        self.assertTrue(torch.all(torch.eq(out[0], self.source[0])))
        self.assertTrue(torch.all(torch.eq(out[1], self.source[1])))

    def test_fidelity(self):
        model = dm.models.ModelCompoundRegistration(2, self.source,
                                                    [self.translation0, self.translation1],
                                                    [self.translation_gd0, self.translation_gd1],
                                                    [True, False])

        fidelity = model.fidelity(self.target)
        self.assertIsInstance(fidelity, torch.Tensor)
        self.assertEqual(fidelity.shape, torch.Size([]))

    def test_get_var(self):
        model = dm.models.ModelCompoundRegistration(2, self.source,
                                                    [self.translation0, self.translation1],
                                                    [self.translation_gd0, self.translation_gd1],
                                                    [True, False])

        gd0, mom0 = model.get_var_tensor()
        self.assertEqual(len(gd0.shape), 1)
        self.assertEqual(len(mom0.shape), 1)
        self.assertEqual(gd0.shape[0], 2*self.n + 2*self.k0 + 2*self.k1)
        self.assertEqual(mom0.shape[0], 2*self.n + 2*self.k0 + 2*self.k1)

        gd1, mom1 = model.get_var_list()
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
        model = dm.models.ModelCompoundRegistration(2, self.source,
                                                    [self.translation0, self.translation1],
                                                    [self.translation_gd0, self.translation_gd1],
                                                    [True, False])
        gd0, mom0 = model.get_var_tensor()
        gd0_out, mom0_out = model.shoot_tensor(it=5)
        self.assertEqual(gd0_out.shape, gd0.shape)
        self.assertEqual(mom0_out.shape, mom0.shape)

    def test_shoot_list(self):
        model = dm.models.ModelCompoundRegistration(2, self.source,
                                                    [self.translation0, self.translation1],
                                                    [self.translation_gd0, self.translation_gd1],
                                                    [True, False])
        gd1, mom1 = model.get_var_list()
        gd1_out, mom1_out = model.shoot_list(it=5)
        self.assertEqual(gd1_out[0].shape, gd1[0].shape)
        self.assertEqual(gd1_out[1].shape, gd1[1].shape)
        self.assertEqual(gd1_out[2].shape, gd1[2].shape)
        self.assertEqual(mom1_out[0].shape, mom1[0].shape)
        self.assertEqual(mom1_out[1].shape, mom1[1].shape)
        self.assertEqual(mom1_out[2].shape, mom1[2].shape)

