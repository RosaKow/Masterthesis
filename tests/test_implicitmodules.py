import unittest

import torch

import defmod as dm

torch.set_default_tensor_type(torch.DoubleTensor)

class TestImplicit02D(unittest.TestCase):
    def setUp(self):
        self.nb_pts = 10
        self.dim = 2
        self.sigma = 0.5
        self.nu = 0.2
        self.gd = torch.rand(self.nb_pts, self.dim).view(-1)
        self.mom = torch.rand(self.nb_pts, self.dim).view(-1)
        self.controls = torch.rand(self.nb_pts, self.dim).view(-1)
        self.implicit0 = dm.implicitmodules.ImplicitModule0(self.dim, self.nb_pts, self.sigma, self.nu)

    def test_call(self):
        points = torch.rand(100, self.dim)

        result = self.implicit0(self.gd, self.controls, points)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, points.shape)

        result = self.implicit0(self.gd, torch.zeros_like(self.controls), points)
        
        self.assertEqual(torch.all(torch.eq(result, torch.zeros_like(result))), True)

    def test_action(self):
        result1 = self.implicit0.action(self.gd, self.implicit0, self.gd, self.controls)
        result2 = self.implicit0(self.gd, self.controls, self.gd.view(-1, 2)).view(-1)

        self.assertEqual(torch.all(torch.eq(result1, result2)), True)

    def test_cost(self):
        cost = self.implicit0.cost(self.gd, self.controls)

        self.assertIsInstance(cost, torch.Tensor)
        self.assertEqual(cost.shape, torch.tensor(0.).shape)

        cost = self.implicit0.cost(self.gd, torch.zeros_like(self.controls))

        self.assertEqual(cost, torch.tensor([0.]))

    def test_compute_geodesic_control(self):
        delta = torch.rand_like(self.gd)

        result = self.implicit0.compute_geodesic_control(delta, self.gd)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(self.gd.shape, result.shape)

    def test_gradcheck_call(self):
        points = torch.rand(10, self.dim, requires_grad=True)
        self.gd.requires_grad_()
        self.controls.requires_grad_()

        self.assertTrue(torch.autograd.gradcheck(self.implicit0.__call__, (self.gd, self.controls, points), raise_exception=False))

    def test_gradcheck_cost(self):
        self.gd.requires_grad_()
        self.mom.requires_grad_()

        self.assertTrue(torch.autograd.gradcheck(self.implicit0.cost, (self.gd, self.mom), raise_exception=False))

    def test_gradcheck_action(self):
        self.gd.requires_grad_()
        self.mom.requires_grad_()
        self.controls.requires_grad_()
        self.gd_module = torch.rand_like(self.gd, requires_grad=True)

        self.assertTrue(torch.autograd.gradcheck(self.implicit0.action, (self.gd, self.implicit0, self.gd_module, self.controls), raise_exception=False))

    def test_gradcheck_compute_geodesic_control(self):
        delta = torch.rand_like(self.gd, requires_grad=True)

        self.assertTrue(torch.autograd.gradcheck(self.implicit0.compute_geodesic_control, (delta, self.gd), raise_exception=False))
