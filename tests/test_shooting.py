import unittest

import torch

import defmod as dm

torch.set_default_tensor_type(torch.DoubleTensor)

class TestShooting(unittest.TestCase):
    def setUp(self):
        self.it = 10
        self.m = 4
        self.gd = torch.rand(self.m, 2, requires_grad=True).view(-1)
        self.mom = torch.rand(self.m, 2, requires_grad=True).view(-1)
        self.landmarks = dm.manifold.Landmarks(2, self.m, gd=self.gd, cotan=self.mom)
        self.trans = dm.deformationmodules.Translations(self.landmarks, 0.5)
        self.h = dm.hamiltonian.Hamiltonian(self.trans)

    def test_shooting(self):
        intermediates = dm.shooting.shoot(self.h, it=self.it)

        self.assertIsInstance(self.h.module.manifold.gd, torch.Tensor)
        self.assertIsInstance(self.h.module.manifold.cotan, torch.Tensor)

        self.assertEqual(self.h.module.manifold.gd.shape, self.gd.shape)
        self.assertEqual(self.h.module.manifold.cotan.shape, self.mom.shape)

        self.assertEqual(len(intermediates), self.it+1)

    def test_shooting_zero(self):
        mom = torch.zeros_like(self.mom, requires_grad=True)
        self.h.module.manifold.fill_cotan(mom)
        dm.shooting.shoot(self.h, it=self.it)

        self.assertTrue(torch.allclose(self.h.module.manifold.gd, self.gd))
        self.assertTrue(torch.allclose(self.h.module.manifold.cotan, mom))

    def test_shooting_rand(self):
        dm.shooting.shoot(self.h, it=self.it)

        self.assertFalse(torch.allclose(self.h.module.manifold.gd, self.gd))
        self.assertFalse(torch.allclose(self.h.module.manifold.cotan, self.mom))

    def test_gradcheck_shoot(self):
        def shoot(gd, mom):
            self.h.module.manifold.fill_gd(gd)
            self.h.module.manifold.fill_cotan(mom)

            dm.shooting.shoot(self.h, it=self.it)

            return self.h.module.manifold.gd, self.h.module.manifold.cotan

        self.gd.requires_grad_()
        self.mom.requires_grad_()

        # We multiply GD by 400. as it seems gradcheck is very sensitive to
        # badly conditioned problems
        # TODO: be sure it is because of that
        self.assertTrue(torch.autograd.gradcheck(shoot, (100.*self.gd, self.mom), raise_exception=True))

