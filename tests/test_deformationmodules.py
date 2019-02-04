import unittest

import torch

import defmod as dm

class TestTranslations2D(unittest.TestCase):
    def setUp(self):
        self.nb_pts = 10
        self.dim = 2
        self.sigma = 0.5
        self.gd = torch.rand(self.nb_pts, self.dim).view(-1)
        self.controls = torch.rand(self.nb_pts, self.dim).view(-1)
        self.trans = dm.deformationmodules.Translations(self.dim, self.nb_pts, self.sigma)

    def test_call(self):
        points = torch.rand(100, self.dim)

        result = self.trans(self.gd, self.controls, points)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, points.shape)

        result = self.trans(self.gd, torch.zeros_like(self.controls), points)
        
        self.assertEqual(torch.all(torch.eq(result, torch.zeros_like(result))), True)

    def test_action(self):
        result1 = self.trans.action(self.gd, self.trans, self.gd, self.controls)
        result2 = self.trans(self.gd, self.controls, self.gd.view(-1, 2)).view(-1)

        self.assertEqual(torch.all(torch.eq(result1, result2)), True)

    def test_cost(self):
        cost = self.trans.cost(self.gd, self.controls)

        self.assertIsInstance(cost, torch.Tensor)
        self.assertEqual(cost.shape, torch.tensor(0.).shape)

        cost = self.trans.cost(self.gd, torch.zeros_like(self.controls))

        self.assertEqual(cost, torch.tensor([0.]))

    def test_compute_geodesic_control(self):
        delta = torch.rand_like(self.gd)

        result = self.trans.compute_geodesic_control(delta, self.gd)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(self.gd.shape, result.shape)


class TestSilentPoints2D(unittest.TestCase):
    def setUp(self):
        self.nb_pts = 10
        self.dim = 2
        self.gd = torch.rand(self.nb_pts, self.dim).view(-1)
        self.controls = torch.rand(self.nb_pts, self.dim).view(-1)
        self.silent_points = dm.deformationmodules.SilentPoints(self.dim, self.nb_pts)

    def test_call(self):
        points = torch.rand(100, self.dim)

        result = self.silent_points(self.gd, self.controls, points)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, points.shape)
        self.assertEqual(torch.all(torch.eq(result, torch.zeros_like(points))), True)

    def test_action(self):
        result1 = self.silent_points.action(self.gd, self.silent_points, self.gd, self.controls)
        result2 = self.silent_points(self.gd, self.controls, self.gd.view(-1, 2)).view(-1)

        self.assertIsInstance(result1, torch.Tensor)
        self.assertEqual(torch.all(torch.eq(result1, result2)), True)

    def test_cost(self):
        cost = self.silent_points.cost(self.gd, self.controls)

        self.assertIsInstance(cost, torch.Tensor)
        self.assertEqual(cost.shape, torch.tensor(0.).shape)
        self.assertEqual(cost, torch.tensor([0.]))

    def test_compute_geodesic_control(self):
        delta = torch.rand_like(self.gd)

        result = self.silent_points.compute_geodesic_control(delta, self.gd)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, torch.tensor([]).shape)


class CompoundTest2D(unittest.TestCase):
    def setUp(self):
        self.dim = 2
        self.sigma = 0.5
        self.nb_pts_silent = 10
        self.nb_pts_trans = 5
        self.nb_pts = self.nb_pts_silent + self.nb_pts_trans
        self.gd_trans = torch.rand(self.nb_pts_trans, self.dim).view(-1)
        self.gd_silent = torch.rand(self.nb_pts_silent, self.dim).view(-1)
        self.gd = torch.cat([self.gd_silent, self.gd_trans])

        self.trans = dm.deformationmodules.Translations(self.dim, self.nb_pts_trans, self.sigma)
        self.silent = dm.deformationmodules.SilentPoints(self.dim, self.nb_pts_silent)
        self.compound = dm.deformationmodules.Compound([self.silent, self.trans])

    def test_type(self):
        self.assertIsInstance(self.compound.module_list, list)
        self.assertIsInstance(self.compound.nb_module, int)
        self.assertIsInstance(self.compound.dim_gd, int)
        self.assertIsInstance(self.compound.dim_controls, int)
        self.assertIsInstance(self.compound.indice_gd, list)
        self.assertIsInstance(self.compound.indice_controls, list)
        self.assertIsInstance(self.compound.nb_pts, int)

    def test_compound(self):
        self.assertEqual(self.compound.nb_pts, self.nb_pts)
        self.assertEqual(self.compound.module_list, [self.silent, self.trans])
        self.assertEqual(self.compound.dim_gd, self.nb_pts_silent*2 + self.nb_pts_trans*2)
        self.assertEqual(self.compound.dim_controls, self.nb_pts_trans*2)
        self.assertEqual(self.compound.indice_controls, [0, 0, 10])
        self.assertEqual(self.compound.indice_gd, [0, 20, 30])

    def test_call(self):
        points = torch.rand(100, self.dim)
        controls = torch.rand_like(self.gd)

        result = self.compound(self.gd, controls, points)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, points.shape)

        result = self.compound(self.gd, torch.zeros_like(controls), points)
        
        self.assertEqual(torch.all(torch.eq(result, torch.zeros_like(points))), True)

    def test_action(self):
        controls = torch.rand_like(self.gd)

        result1 = self.compound.action(self.gd, self.compound, self.gd, controls)
        result2 = self.compound(self.gd, controls, self.gd.view(-1, 2)).view(-1)

        self.assertIsInstance(result1, torch.Tensor)
        self.assertEqual(torch.all(torch.eq(result1, result2)), True)

    def test_cost(self):
        controls = torch.rand_like(self.gd)

        cost = self.compound.cost(self.gd, controls)

        self.assertIsInstance(cost, torch.Tensor)
        self.assertEqual(cost.shape, torch.tensor(0.).shape)

    def test_compute_geodesic_control(self):
        delta = torch.rand_like(self.gd)

        result = self.compound.compute_geodesic_control(delta, self.gd)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, torch.Size([self.compound.dim_controls]))


