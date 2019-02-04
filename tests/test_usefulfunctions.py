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

    def test_scal(self):
        m = 10
        x = torch.rand(m, 2)
        y = torch.rand(m, 2)

        self.assertIsInstance(dm.usefulfunctions.scal(x, y), torch.Tensor)
        self.assertTrue(torch.allclose(
            dm.usefulfunctions.scal(x, y), torch.dot(x.view(-1), y.view(-1))))

    def test_distancematrix(self):
        m = 10
        x = torch.rand(m, 2)
        y = torch.rand(m, 2)

        dist_matrix = dm.usefulfunctions.distances(x, y)
        self.assertIsInstance(dist_matrix, torch.Tensor)
        self.assertEqual(dist_matrix.shape, torch.Size([m, m]))
        self.assertTrue(torch.allclose(dist_matrix[1, 2], torch.dist(x[1], y[2])))

    def test_sqdistancematrix(self):
        m = 10
        x = torch.rand(m, 2)
        y = torch.rand(m, 2)

        sqdist_matrix = dm.usefulfunctions.sqdistances(x, y)
        self.assertIsInstance(sqdist_matrix, torch.Tensor)
        self.assertEqual(sqdist_matrix.shape, torch.Size([m, m]))
        self.assertTrue(torch.allclose(sqdist_matrix[1, 2], torch.dist(x[1], y[2])**2))

    def test_kernelxxmatrix(self):
        m = 10
        x = torch.rand(m, 2)

        kxx_matrix = dm.usefulfunctions.K_xx(x)
        self.assertIsInstance(kxx_matrix, torch.Tensor)
        self.assertEqual(kxx_matrix.shape, torch.Size([m, m]))

    def test_kernelxymatrix(self):
        m = 10
        n = 5
        x = torch.rand(m, 2)
        y = torch.rand(n, 2)

        kxy_matrix = dm.usefulfunctions.K_xy(x, y)
        self.assertIsInstance(kxy_matrix, torch.Tensor)
        self.assertEqual(kxy_matrix.shape, torch.Size([m, n]))

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

