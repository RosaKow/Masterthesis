import unittest

import torch

import defmod as dm


class TestLandmarks(unittest.TestCase):
    def setUp(self):
        pass

    def test_default_constructor(self):
        points = torch.rand(10, 2)
        landmarks = dm.manifold.Landmarks(points)

        self.assertIsInstance(landmarks.gd, torch.Tensor)
        self.assertIsInstance(landmarks.mom, torch.Tensor)

        self.assertEqual(landmarks.dim_gd, points.view(-1).shape[0])
        self.assertEqual(landmarks.gd.shape[0], points.view(-1).shape[0])
        self.assertEqual(landmarks.mom.shape[0], points.view(-1).shape[0])
        self.assertTrue(torch.allclose(landmarks.gd.view(-1, 2), points))
        self.assertTrue(torch.allclose(landmarks.mom.view(-1, 2), torch.zeros_like(points)))

    def test_empty_constructor(self):
        nb_pts = 10
        landmarks = dm.manifold.Landmarks.build_empty(2, nb_pts)

        self.assertIsInstance(landmarks.gd, torch.Tensor)
        self.assertIsInstance(landmarks.mom, torch.Tensor)

        self.assertEqual(landmarks.dim_gd, 2*nb_pts)
        self.assertEqual(landmarks.gd.shape[0], 2*nb_pts)
        self.assertEqual(landmarks.mom.shape[0], 2*nb_pts)
        self.assertTrue(torch.allclose(landmarks.gd.view(-1, 2), torch.zeros(nb_pts, 2)))
        self.assertTrue(torch.allclose(landmarks.mom.view(-1, 2), torch.zeros(nb_pts, 2)))

    def test_fill(self):
        nb_pts = 10
        points = torch.rand(nb_pts, 2)
        landmarks = dm.manifold.Landmarks.build_empty(2, nb_pts)

        landmarks.fill_gd(points.view(-1))
        landmarks.fill_mom(points.view(-1))

        self.assertIsInstance(landmarks.gd, torch.Tensor)
        self.assertIsInstance(landmarks.mom, torch.Tensor)

        self.assertEqual(landmarks.dim_gd, points.view(-1).shape[0])
        self.assertEqual(landmarks.gd.shape[0], points.view(-1).shape[0])
        self.assertEqual(landmarks.mom.shape[0], points.view(-1).shape[0])
        self.assertTrue(torch.allclose(landmarks.gd.view(-1, 2), points))
        self.assertTrue(torch.allclose(landmarks.mom.view(-1, 2), points))

        landmarks.fill_gd(torch.zeros_like(points.view(-1)))
        landmarks.fill_mom(torch.zeros_like(points.view(-1)))

        self.assertTrue(torch.allclose(landmarks.gd.view(-1, 2), torch.zeros(nb_pts, 2)))
        self.assertTrue(torch.allclose(landmarks.mom.view(-1, 2), torch.zeros(nb_pts, 2)))

        landmarks.gd = points.view(-1)
        landmarks.mom = points.view(-1)

        self.assertIsInstance(landmarks.gd, torch.Tensor)
        self.assertIsInstance(landmarks.mom, torch.Tensor)

        self.assertEqual(landmarks.dim_gd, points.view(-1).shape[0])
        self.assertEqual(landmarks.gd.shape[0], points.view(-1).shape[0])
        self.assertEqual(landmarks.mom.shape[0], points.view(-1).shape[0])
        self.assertTrue(torch.allclose(landmarks.gd.view(-1, 2), points))
        self.assertTrue(torch.allclose(landmarks.mom.view(-1, 2), points))

    def test_muladd(self):
        nb_pts = 10
        scale = 0.1
        gd_0, mom_0 = torch.rand(nb_pts, 2), torch.rand(nb_pts, 2)
        gd_1, mom_1 = torch.rand(nb_pts, 2), torch.rand(nb_pts, 2)

        landmarks = dm.manifold.Landmarks.build_empty(2, nb_pts)
        landmarks.fill_gd(gd_0.view(-1))
        landmarks.fill_mom(mom_0.view(-1))

        landmarks.muladd(gd_1.view(-1), mom_1.view(-1), scale)

        self.assertTrue(torch.allclose(landmarks.gd.view(-1, 2), gd_0+scale*gd_1))
        self.assertTrue(torch.allclose(landmarks.mom.view(-1, 2), mom_0+scale*mom_1))

class TestCompoundManifold(unittest.TestCase):
    def setUp(self):
        self.nb_pts0 = 10
        self.nb_pts1 = 15
        self.nb_pts = self.nb_pts0+self.nb_pts1
        self.points0 = torch.rand(self.nb_pts0, 2)
        self.points1 = torch.rand(self.nb_pts1, 2)
        self.landmarks0 = dm.manifold.Landmarks(self.points0)
        self.landmarks1 = dm.manifold.Landmarks(self.points1)
        self.compound = dm.manifold.CompoundManifold([self.landmarks0, self.landmarks1])

    def test_default_constructor(self):
        self.assertIsInstance(self.compound[0], dm.manifold.Landmarks)
        self.assertIsInstance(self.compound[1], dm.manifold.Landmarks)
        self.assertEqual(self.compound[0], self.compound.manifold_list[0])
        self.assertEqual(self.compound[1], self.compound.manifold_list[1])
        
        self.assertIsInstance(self.compound.gd, torch.Tensor)
        self.assertIsInstance(self.compound.mom, torch.Tensor)

        self.assertEqual(self.compound.dim_gd, 2*self.nb_pts0+2*self.nb_pts1)
        self.assertEqual(self.compound.gd.shape[0], 2*self.nb_pts0+2*self.nb_pts1)
        self.assertEqual(self.compound.mom.shape[0], 2*self.nb_pts0+2*self.nb_pts1)
        self.assertTrue(torch.allclose(self.compound.gd,
                                       torch.cat([self.points0.view(-1), self.points1.view(-1)])))
        self.assertTrue(torch.allclose(self.compound.mom,
                                       torch.zeros(2*self.nb_pts0+2*self.nb_pts1)))
        self.assertTrue(torch.allclose(self.compound[0].gd.view(-1, 2), self.points0))
        self.assertTrue(torch.allclose(self.compound[0].mom.view(-1, 2), torch.zeros_like(self.points0)))
        self.assertTrue(torch.allclose(self.compound[1].gd.view(-1, 2), self.points1))
        self.assertTrue(torch.allclose(self.compound[1].mom.view(-1, 2), torch.zeros_like(self.points1)))

    def test_fill(self):
        self.compound.fill_gd(torch.zeros(2*self.nb_pts0+2*self.nb_pts1))
        self.compound.fill_mom(torch.zeros(2*self.nb_pts0+2*self.nb_pts1))

        self.assertTrue(torch.allclose(self.compound.gd.view(-1, 2), torch.zeros(self.nb_pts, 2)))
        self.assertTrue(torch.allclose(self.compound.mom.view(-1, 2), torch.zeros(self.nb_pts, 2)))

        self.compound.gd = torch.cat([self.points0.view(-1), self.points1.view(-1)])
        self.compound.mom = torch.cat([self.points0.view(-1), self.points1.view(-1)])

        self.assertIsInstance(self.compound.gd, torch.Tensor)
        self.assertIsInstance(self.compound.mom, torch.Tensor)

        self.assertEqual(self.compound.dim_gd, 2*self.nb_pts0+2*self.nb_pts1)
        self.assertEqual(self.compound.gd.shape[0], 2*self.nb_pts0+2*self.nb_pts1)
        self.assertEqual(self.compound.mom.shape[0], 2*self.nb_pts0+2*self.nb_pts1)
        self.assertTrue(torch.allclose(self.compound.gd.view(-1),
                                       torch.cat([self.points0.view(-1), self.points1.view(-1)])))
        self.assertTrue(torch.allclose(self.compound.mom.view(-1),
                                       torch.cat([self.points0.view(-1), self.points1.view(-1)])))

    def test_muladd(self):
        nb_pts = self.nb_pts0+self.nb_pts1
        scale = 0.1
        gd_0, mom_0 = torch.rand(nb_pts, 2), torch.rand(nb_pts, 2)
        gd_1, mom_1 = torch.rand(nb_pts, 2), torch.rand(nb_pts, 2)

        landmarks = dm.manifold.Landmarks.build_empty(2, nb_pts)
        landmarks.fill_gd(gd_0.view(-1))
        landmarks.fill_mom(mom_0.view(-1))

        landmarks.muladd(gd_1.view(-1), mom_1.view(-1), scale)

        self.assertTrue(torch.allclose(landmarks.gd.view(-1, 2), gd_0+scale*gd_1))
        self.assertTrue(torch.allclose(landmarks.mom.view(-1, 2), mom_0+scale*mom_1))

