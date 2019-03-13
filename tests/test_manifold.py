import unittest

import torch
from torch.autograd import gradcheck

import defmod as dm

torch.set_default_tensor_type(torch.DoubleTensor)

class TestLandmarks(unittest.TestCase):
    def setUp(self):
        self.nb_pts = 10
        self.gd = torch.rand(self.nb_pts, 2, requires_grad=True).view(-1)
        self.tan = torch.rand(self.nb_pts, 2, requires_grad=True).view(-1)
        self.cotan = torch.rand(self.nb_pts, 2, requires_grad=True).view(-1)

    def test_constructor(self):
        landmarks = dm.manifold.Landmarks(2, self.nb_pts, gd=self.gd, tan=self.tan, cotan=self.cotan)

        self.assertEqual(landmarks.nb_pts, self.nb_pts)
        self.assertEqual(landmarks.dim, 2)
        self.assertEqual(landmarks.dim_gd, 2*self.nb_pts)
        self.assertTrue(torch.allclose(landmarks.gd, self.gd))
        self.assertTrue(torch.allclose(landmarks.tan, self.tan))
        self.assertTrue(torch.allclose(landmarks.cotan, self.cotan))

    def test_fill(self):
        landmarks = dm.manifold.Landmarks(2, self.nb_pts)

        landmarks.fill_gd(self.gd)
        landmarks.fill_tan(self.tan)
        landmarks.fill_cotan(self.cotan)

        self.assertTrue(torch.allclose(landmarks.gd, self.gd))
        self.assertTrue(torch.allclose(landmarks.tan, self.tan))
        self.assertTrue(torch.allclose(landmarks.cotan, self.cotan))

    def test_assign(self):
        landmarks = dm.manifold.Landmarks(2, self.nb_pts)

        landmarks.gd = self.gd
        landmarks.tan = self.tan
        landmarks.cotan = self.cotan

        self.assertTrue(torch.allclose(landmarks.gd, self.gd))
        self.assertTrue(torch.allclose(landmarks.tan, self.tan))
        self.assertTrue(torch.allclose(landmarks.cotan, self.cotan))

    def test_muladd(self):
        landmarks = dm.manifold.Landmarks(2, self.nb_pts, gd=self.gd, tan=self.tan, cotan=self.cotan)

        scale = 1.5
        d_gd = torch.rand(self.nb_pts, 2, requires_grad=True).view(-1)
        d_tan = torch.rand(self.nb_pts, 2, requires_grad=True).view(-1)
        d_cotan = torch.rand(self.nb_pts, 2, requires_grad=True).view(-1)

        landmarks.muladd_gd(d_gd, scale)
        landmarks.muladd_tan(d_tan, scale)
        landmarks.muladd_cotan(d_cotan, scale)

        self.assertTrue(torch.allclose(landmarks.gd, self.gd+scale*d_gd))
        self.assertTrue(torch.allclose(landmarks.tan, self.tan+scale*d_tan))
        self.assertTrue(torch.allclose(landmarks.cotan, self.cotan+scale*d_cotan))

    def test_action(self):
        landmarks = dm.manifold.Landmarks(2, self.nb_pts, gd=self.gd, tan=self.tan, cotan=self.cotan)

        nb_pts_mod = 15
        landmarks_mod = dm.manifold.Landmarks(2, nb_pts_mod, gd=torch.rand(nb_pts_mod, 2).view(-1))
        trans = dm.deformationmodules.Translations(landmarks_mod, 1.5)
        trans.fill_controls(torch.rand_like(landmarks_mod.gd))

        man = landmarks.action(trans)

        self.assertIsInstance(man, dm.manifold.Landmarks)
        self.assertEqual(man.gd.shape[0], 2*self.nb_pts)

    def test_inner_prod_module(self):
        landmarks = dm.manifold.Landmarks(2, self.nb_pts, gd=self.gd, tan=self.tan, cotan=self.cotan)

        nb_pts_mod = 15
        landmarks_mod = dm.manifold.Landmarks(2, nb_pts_mod, gd=torch.rand(nb_pts_mod, 2).view(-1))
        trans = dm.deformationmodules.Translations(landmarks_mod, 1.5)
        trans.fill_controls(torch.rand_like(landmarks_mod.gd))

        inner_prod = landmarks.inner_prod_module(trans)

        self.assertIsInstance(inner_prod, torch.Tensor)
        self.assertEqual(inner_prod.shape, torch.Size([]))        

    def test_gradcheck_fill(self):
        def fill_gd(gd):
            landmarks.fill_gd(gd)
            return landmarks.gd
 
        def fill_tan(tan):
            landmarks.fill_tan(tan)
            return landmarks.tan

        def fill_cotan(cotan):
            landmarks.fill_cotan(cotan)
            return landmarks.cotan

        self.gd.requires_grad_()
        self.tan.requires_grad_()
        self.cotan.requires_grad_()

        landmarks = dm.manifold.Landmarks(2, self.nb_pts)

        self.assertTrue(gradcheck(fill_gd, (self.gd), raise_exception=False))
        self.assertTrue(gradcheck(fill_tan, (self.tan), raise_exception=False))
        self.assertTrue(gradcheck(fill_cotan, (self.cotan), raise_exception=False))

    def test_gradcheck_muladd(self):
        def muladd_gd(gd):
            landmarks.fill_gd(self.gd)
            landmarks.muladd_gd(gd, scale)
            return landmarks.gd

        def muladd_tan(tan):
            landmarks.fill_tan(self.tan)
            landmarks.muladd_tan(tan, scale)
            return landmarks.tan

        def muladd_cotan(cotan):
            landmarks.fill_cotan(self.cotan)
            landmarks.muladd_cotan(cotan, scale)
            return landmarks.cotan

        self.gd.requires_grad_()
        self.tan.requires_grad_()
        self.cotan.requires_grad_()

        landmarks = dm.manifold.Landmarks(2, self.nb_pts, gd=self.gd, tan=self.tan, cotan=self.cotan)
        scale = 2.

        gd_mul = torch.rand_like(self.gd, requires_grad=True)
        tan_mul = torch.rand_like(self.tan, requires_grad=True)
        cotan_mul = torch.rand_like(self.cotan, requires_grad=True)

        self.assertTrue(gradcheck(muladd_gd, (gd_mul), raise_exception=False))
        self.assertTrue(gradcheck(muladd_tan, (tan_mul), raise_exception=False))
        self.assertTrue(gradcheck(muladd_cotan, (cotan_mul), raise_exception=False))

    def test_gradcheck_action(self):
        def action(gd, controls):
            landmarks.fill_gd(gd)
            module = dm.deformationmodules.Translations(landmarks, 2.)
            module.fill_controls(controls)
            man = landmarks.action(module)
            return man.gd, man.tan, man.cotan

        self.gd.requires_grad_()
        controls = torch.rand_like(self.gd, requires_grad=True)
        landmarks = dm.manifold.Landmarks(2, self.nb_pts, gd=self.gd)

        self.assertTrue(gradcheck(action, (self.gd, controls), raise_exception=False))

    def test_gradcheck_inner_prod_module(self):
        def inner_prod_module(gd, controls):
            landmarks.fill_gd(gd)
            module = dm.deformationmodules.Translations(landmarks, 2.)
            module.fill_controls(controls)
            return landmarks.inner_prod_module(module)

        self.gd.requires_grad_()
        controls = torch.rand_like(self.gd, requires_grad=True)
        landmarks = dm.manifold.Landmarks(2, self.nb_pts, gd=self.gd)

        self.assertTrue(gradcheck(inner_prod_module, (self.gd, controls), raise_exception=False))


class TestCompoundManifold(unittest.TestCase):
    def setUp(self):
        self.nb_pts0 = 10
        self.nb_pts1 = 15
        self.gd0 = torch.rand(self.nb_pts0, 2, requires_grad=True).view(-1)
        self.tan0 = torch.rand(self.nb_pts0, 2, requires_grad=True).view(-1)
        self.cotan0 = torch.rand(self.nb_pts0, 2, requires_grad=True).view(-1)
        self.gd1 = torch.rand(self.nb_pts1, 2, requires_grad=True).view(-1)
        self.tan1 = torch.rand(self.nb_pts1, 2, requires_grad=True).view(-1)
        self.cotan1 = torch.rand(self.nb_pts1, 2, requires_grad=True).view(-1)
        self.landmarks0 = dm.manifold.Landmarks(2, self.nb_pts0, gd=self.gd0, tan=self.tan0, cotan=self.cotan0)
        self.landmarks1 = dm.manifold.Landmarks(2, self.nb_pts1, gd=self.gd1, tan=self.tan1, cotan=self.cotan1)
        self.compound = dm.manifold.CompoundManifold([self.landmarks0, self.landmarks1])

    def test_constructor(self):
        self.assertEqual(self.compound.nb_pts, self.nb_pts0+self.nb_pts1)
        self.assertEqual(self.compound.dim, 2)
        self.assertEqual(self.compound.dim_gd, 2*self.nb_pts0+2*self.nb_pts1)
        self.assertEqual(self.compound.nb_manifold, 2)
        self.assertTrue(torch.allclose(self.compound.gd, torch.cat([self.gd0, self.gd1])))
        self.assertTrue(torch.allclose(self.compound.tan, torch.cat([self.tan0, self.tan1])))
        self.assertTrue(torch.allclose(self.compound.cotan, torch.cat([self.cotan0, self.cotan1])))

    def test_fill(self):
        self.compound.fill_gd(torch.cat([self.gd0, self.gd1]))
        self.compound.fill_tan(torch.cat([self.tan0, self.tan1]))
        self.compound.fill_cotan(torch.cat([self.cotan0, self.cotan1]))

        self.assertTrue(torch.allclose(self.compound[0].gd, self.gd0))
        self.assertTrue(torch.allclose(self.compound[0].tan, self.tan0))
        self.assertTrue(torch.allclose(self.compound[0].cotan, self.cotan0))
        self.assertTrue(torch.allclose(self.compound[1].gd, self.gd1))
        self.assertTrue(torch.allclose(self.compound[1].tan, self.tan1))
        self.assertTrue(torch.allclose(self.compound[1].cotan, self.cotan1))

    def test_assign(self):
        self.compound.gd = torch.cat([self.gd0, self.gd1])
        self.compound.tan = torch.cat([self.tan0, self.tan1])
        self.compound.cotan = torch.cat([self.cotan0, self.cotan1])

        self.assertTrue(torch.allclose(self.compound[0].gd, self.gd0))
        self.assertTrue(torch.allclose(self.compound[0].tan, self.tan0))
        self.assertTrue(torch.allclose(self.compound[0].cotan, self.cotan0))
        self.assertTrue(torch.allclose(self.compound[1].gd, self.gd1))
        self.assertTrue(torch.allclose(self.compound[1].tan, self.tan1))
        self.assertTrue(torch.allclose(self.compound[1].cotan, self.cotan1))

    def test_muladd(self):
        scale = 1.5
        d_gd0 = torch.rand(self.nb_pts0, 2).view(-1)
        d_tan0 = torch.rand(self.nb_pts0, 2).view(-1)
        d_cotan0 = torch.rand(self.nb_pts0, 2).view(-1)
        d_gd1 = torch.rand(self.nb_pts1, 2).view(-1)
        d_tan1 = torch.rand(self.nb_pts1, 2).view(-1)
        d_cotan1 = torch.rand(self.nb_pts1, 2).view(-1)

        self.compound.muladd_gd(torch.cat([d_gd0, d_gd1]), scale)
        self.compound.muladd_tan(torch.cat([d_tan0, d_tan1]), scale)
        self.compound.muladd_cotan(torch.cat([d_cotan0, d_cotan1]), scale)

        self.assertTrue(torch.allclose(self.compound[0].gd, self.gd0+scale*d_gd0))
        self.assertTrue(torch.allclose(self.compound[0].tan, self.tan0+scale*d_tan0))
        self.assertTrue(torch.allclose(self.compound[0].cotan, self.cotan0+scale*d_cotan0))
        self.assertTrue(torch.allclose(self.compound[1].gd, self.gd1+scale*d_gd1))
        self.assertTrue(torch.allclose(self.compound[1].tan, self.tan1+scale*d_tan1))
        self.assertTrue(torch.allclose(self.compound[1].cotan, self.cotan1+scale*d_cotan1))


    def test_action(self):
        nb_pts_mod = 15
        landmarks_mod = dm.manifold.Landmarks(2, nb_pts_mod, gd=torch.rand(nb_pts_mod, 2).view(-1))
        trans = dm.deformationmodules.Translations(landmarks_mod, 1.5)

        man = self.compound.action(trans)

        self.assertIsInstance(man, dm.manifold.CompoundManifold)
        self.assertTrue(man.nb_manifold, self.compound.nb_manifold)
        self.assertEqual(man[0].gd.shape[0], 2*self.nb_pts0)
        self.assertEqual(man[1].gd.shape[0], 2*self.nb_pts1)

    # def test_gradcheck_fill(self):
    #     def fill_gd(gd):
    #         self.compound.fill_gd(gd)
    #         return self.compound.gd
 
    #     def fill_tan(tan):
    #         self.compound.fill_tan(tan)
    #         return self.compound.tan

    #     def fill_cotan(cotan):
    #         self.compound.fill_cotan(cotan)
    #         return self.compound.cotan

    #     gd = torch.cat([self.gd0, self.gd1]).requires_grad_()
    #     tan = torch.cat([self.tan0, self.tan1]).requires_grad_()
    #     cotan = torch.cat([self.cotan0, self.cotan1]).requires_grad_()

    #     self.assertTrue(gradcheck(fill_gd, (gd), raise_exception=False))
    #     self.assertTrue(gradcheck(fill_tan, (tan), raise_exception=False))
    #     self.assertTrue(gradcheck(fill_cotan, (cotan), raise_exception=False))

    # def test_gradcheck_muladd(self):
    #     def muladd_gd(gd_mul):
    #         self.compound.fill_gd(gd)
    #         self.compound.muladd_gd(gd_mul, scale)
    #         return self.compound.gd

    #     def muladd_tan(tan_mul):
    #         self.compound.fill_tan(tan)
    #         self.compound.muladd_tan(tan_mul, scale)
    #         return self.compound.tan

    #     def muladd_cotan(cotan_mul):
    #         self.compound.fill_cotan(cotan)
    #         self.compound.muladd_cotan(cotan_mul, scale)
    #         return self.compound.cotan

    #     scale = 2.
    #     gd = torch.cat([self.gd0, self.gd1]).requires_grad_()
    #     tan = torch.cat([self.tan0, self.tan1]).requires_grad_()
    #     cotan = torch.cat([self.cotan0, self.cotan1]).requires_grad_()

    #     gd_mul = torch.rand_like(gd, requires_grad=True)
    #     tan_mul = torch.rand_like(tan, requires_grad=True)
    #     cotan_mul = torch.rand_like(cotan, requires_grad=True)

    #     self.assertTrue(gradcheck(muladd_gd, (gd_mul), raise_exception=False))
    #     self.assertTrue(gradcheck(muladd_tan, (tan_mul), raise_exception=False))
    #     self.assertTrue(gradcheck(muladd_cotan, (cotan_mul), raise_exception=False))

    def test_gradcheck_action(self):
        def action(gd, controls):
            self.compound.fill_gd(gd)
            module = dm.deformationmodules.Translations(self.compound, 2.)
            module.fill_controls(controls)
            man = self.compound.action(module)
            return man.gd, man.tan, man.cotan

        gd = torch.cat([self.gd0, self.gd1]).requires_grad_()

        gd.requires_grad_()
        controls = torch.rand_like(gd, requires_grad=True)

        self.assertTrue(gradcheck(action, (gd, controls), raise_exception=False))

    # def test_gradcheck_inner_prod_module(self):
    #     def inner_prod_module(gd, controls):
    #         self.compound.fill_gd(gd)
    #         module = dm.deformationmodules.Translations(self.compound, 2.)
    #         module.fill_controls(controls)
    #         return self.compound.inner_prod_module(module)

    #     gd = torch.cat([self.gd0, self.gd1]).requires_grad_()
    #     gd.requires_grad_()
    #     controls = torch.rand_like(gd, requires_grad=True)

    #     self.assertTrue(gradcheck(inner_prod_module, (gd, controls), raise_exception=False))

