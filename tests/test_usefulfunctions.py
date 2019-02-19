import unittest

import torch

import defmod as dm


class TestUsefulFunctions(unittest.TestCase):
    def test_aabb(self):
        points = torch.rand(10, 2)
        aabb = dm.usefulfunctions.AABB.build_from_points(points)

        self.assertEqual(aabb.xmin, torch.min(points[:, 0]))
        self.assertEqual(aabb.ymin, torch.min(points[:, 1]))
        self.assertEqual(aabb.xmax, torch.max(points[:, 0]))
        self.assertEqual(aabb.ymax, torch.max(points[:, 1]))

        self.assertEqual(aabb.width, torch.max(points[:, 0]) - torch.min(points[:, 0]))
        self.assertEqual(aabb.height, torch.max(points[:, 1]) - torch.min(points[:, 1]))

        self.assertIsInstance(aabb.get_list(), list)
        self.assertEqual(len(aabb.get_list()), 4)
        
        aabb_list = aabb.get_list()
        self.assertEqual(aabb_list[0], aabb.xmin)
        self.assertEqual(aabb_list[1], aabb.xmax)
        self.assertEqual(aabb_list[2], aabb.ymin)
        self.assertEqual(aabb_list[3], aabb.ymax)
        
        self.assertEqual(aabb[0], aabb.xmin)
        self.assertEqual(aabb[1], aabb.xmax)
        self.assertEqual(aabb[2], aabb.ymin)
        self.assertEqual(aabb[3], aabb.ymax)

    def test_aabb_is_inside(self):
        points = torch.tensor([[1., 0.],   # Inside
                               [2., 1.],   # Not inside
                               [0.5, 0.5], # Inside
                               [-1., 0.5], # Not inside
                               [0.5, -1]]) # Not inside

        aabb = dm.usefulfunctions.AABB(0., 1., 0., 1.)
        
        is_inside = aabb.is_inside(points)

        self.assertIsInstance(is_inside, torch.Tensor)
        self.assertEqual(is_inside.shape[0], points.shape[0])
        self.assertTrue(is_inside[0])
        self.assertFalse(is_inside[1])
        self.assertTrue(is_inside[2])
        self.assertFalse(is_inside[3])
        self.assertFalse(is_inside[4])

    def test_aabb_sample_random_point(self):
        points = torch.randn(4, 2)
        aabb = dm.usefulfunctions.AABB.build_from_points(points)

        sampled = aabb.sample_random_point(100)

        self.assertIsInstance(sampled, torch.Tensor)
        self.assertTrue(sampled.shape, torch.Size([100, 2]))
        self.assertTrue(torch.all(aabb.is_inside(sampled)))

    def test_gridandvec(self):
        m = 10
        n = 8
        u, v = torch.meshgrid(torch.tensor(range(0, m)), torch.tensor(range(0, n)))

        vec = dm.usefulfunctions.grid2vec(u, v)
        self.assertIsInstance(vec, torch.Tensor)
        self.assertEqual(vec.shape, torch.Size([m*n, 2]))

        u_out, v_out = dm.usefulfunctions.vec2grid(vec, u.shape[0], v.shape[1])
        self.assertIsInstance(u_out, torch.Tensor)
        self.assertIsInstance(v_out, torch.Tensor)
        self.assertTrue(torch.all(torch.eq(u, u_out)))
        self.assertTrue(torch.all(torch.eq(v, v_out)))

