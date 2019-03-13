import unittest

import torch

import defmod as dm

torch.set_default_tensor_type(torch.DoubleTensor)

class TestHamiltonian(unittest.TestCase):
    def setUp(self):
        self.nb_pts = 10
        self.sigma = 0.5
        self.gd = 100.*torch.rand(self.nb_pts, 2).view(-1)
        self.mom = 100.*torch.rand_like(self.gd).view(-1)
        self.landmarks = dm.manifold.Landmarks(2, self.nb_pts, gd=self.gd, cotan=self.mom)
        self.controls = 100.*torch.rand_like(self.gd)

        self.trans = dm.deformationmodules.Translations(self.landmarks, self.sigma)
        self.trans.fill_controls(self.controls)

        self.h = dm.hamiltonian.Hamiltonian(self.trans)

    def test_good_init(self):
        self.assertIsInstance(self.h.module, dm.deformationmodules.DeformationModule)

    def test_apply_mom(self):
        self.assertIsInstance(self.h.apply_mom(), torch.Tensor)
        self.assertEqual(self.h.apply_mom().shape, torch.Size([]))

    def test_call(self):
        self.assertIsInstance(self.h(), torch.Tensor)
        self.assertEqual(self.h().shape, torch.Size([]))

    def test_geodesic_controls(self):
        self.h.geodesic_controls()
        self.assertIsInstance(self.h.module.controls, torch.Tensor)
        self.assertTrue(self.h.module.controls.shape, self.controls)

    def test_gradcheck_call(self):
        def call(gd, mom, controls):
            self.h.module.manifold.fill_gd(gd)
            self.h.module.manifold.fill_cotan(mom)
            self.h.module.fill_controls(controls)

            return self.h()

        self.gd.requires_grad_()
        self.mom.requires_grad_()
        self.controls.requires_grad_()
        
        self.assertTrue(torch.autograd.gradcheck(call, (self.gd, self.mom, self.controls), raise_exception=False))

    def test_gradcheck_apply_mom(self):
        def apply_mom(gd, mom, controls):

            self.h.module.manifold.fill_gd(gd)
            self.h.module.manifold.fill_cotan(mom)
            self.h.module.fill_controls(controls)

            return self.h.apply_mom()

        self.gd.requires_grad_()
        self.mom.requires_grad_()
        self.controls.requires_grad_()

        self.assertTrue(torch.autograd.gradcheck(apply_mom, (self.gd, self.mom, self.controls), raise_exception=False))

    def test_gradcheck_geodesic_controls(self):
        def geodesic_controls(gd, mom):
            self.h.module.manifold.fill_gd(gd)
            self.h.module.manifold.fill_cotan(mom)

            self.h.geodesic_controls()

            return self.h.module.controls

        self.gd.requires_grad_()
        self.mom.requires_grad_()

        self.assertTrue(torch.autograd.gradcheck(geodesic_controls, (self.gd, self.mom), raise_exception=False))


# This constitute more as an integration test than an unit test, but using Hamiltonian with
# compound modules need some attentions
class TestHamiltonianCompound(unittest.TestCase):
    def setUp(self):
        self.nb_pts_trans = 10
        self.nb_pts_silent = 15
        self.sigma = 0.5
        
        self.gd_trans = 100.*torch.rand(self.nb_pts_trans, 2).view(-1)
        self.mom_trans = 100.*torch.rand_like(self.gd_trans).view(-1)
        self.gd_silent = 100.*torch.rand(self.nb_pts_silent, 2).view(-1)
        self.mom_silent = 100.*torch.rand_like(self.gd_silent).view(-1)
        self.gd = torch.cat([self.gd_trans])
        self.mom = torch.cat([self.mom_trans])
        
        self.landmarks_trans = dm.manifold.Landmarks(2, self.nb_pts_trans, gd=self.gd_trans, cotan=self.mom_trans)
        self.landmarks_silent = dm.manifold.Landmarks(2, self.nb_pts_silent, gd=self.gd_silent, cotan=self.mom_silent)
        self.controls = 100.*torch.rand_like(self.gd_trans)

        self.trans = dm.deformationmodules.Translations(self.landmarks_trans, self.sigma)
        self.trans.fill_controls(self.controls)
        self.silent = dm.deformationmodules.SilentPoints(self.landmarks_silent)
        self.compound = dm.deformationmodules.CompoundModule([self.trans])

        self.h = dm.hamiltonian.Hamiltonian(self.compound)

    def test_good_init(self):
        self.assertIsInstance(self.h.module, dm.deformationmodules.DeformationModule)

    def test_apply_mom(self):
        self.assertIsInstance(self.h.apply_mom(), torch.Tensor)
        self.assertEqual(self.h.apply_mom().shape, torch.Size([]))

    def test_call(self):
        self.assertIsInstance(self.h(), torch.Tensor)
        self.assertEqual(self.h().shape, torch.Size([]))

    def test_geodesic_controls(self):
        self.gd_trans.requires_grad_()
        self.mom_trans.requires_grad_()
        self.h.geodesic_controls()
        self.assertIsInstance(self.h.module.controls, torch.Tensor)
        self.assertTrue(self.h.module.controls.shape, self.controls)

    def test_gradcheck_call(self):
        def call(gd, mom, controls):
            self.h.module.manifold.fill_gd(gd)
            self.h.module.manifold.fill_cotan(mom)
            self.h.module.fill_controls(controls)

            return self.h()

        self.gd.requires_grad_()
        self.mom.requires_grad_()
        self.controls.requires_grad_()
        
        self.assertTrue(torch.autograd.gradcheck(call, (self.gd, self.mom, self.controls), raise_exception=False))

    def test_gradcheck_apply_mom(self):
        def apply_mom(gd, mom, controls):
            self.h.module.manifold.fill_gd(gd)
            self.h.module.manifold.fill_cotan(mom)
            self.h.module.fill_controls(controls)

            return self.h.apply_mom()

        self.gd.requires_grad_()
        self.mom.requires_grad_()
        self.controls.requires_grad_()

        self.assertTrue(torch.autograd.gradcheck(apply_mom, (self.gd, self.mom, self.controls), raise_exception=False))

    # def test_gradcheck_geodesic_controls(self):
    #     def geodesic_controls(gd, mom):
    #         self.h.module.manifold.fill_gd(gd)
    #         self.h.module.manifold.fill_cotan(mom)

    #         self.h.geodesic_controls()

    #         return self.h.module.controls

    #     self.gd.requires_grad_()
    #     self.mom.requires_grad_()

    #     self.assertTrue(torch.autograd.gradcheck(geodesic_controls, (self.gd, self.mom), raise_exception=False))

