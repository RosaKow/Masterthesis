import unittest

import torch

import defmod as dm


class TestHamiltonian(unittest.TestCase):
    def setUp(self):
        self.nb_pts = 10
        self.sigma = 0.5
        self.trans = dm.deformationmodules.Translations(2, self.nb_pts, self.sigma)

    def test_good_init(self):        
        h = dm.hamiltonian.Hamiltonian(self.trans)

        self.assertEqual(torch.all(torch.eq(h.init_controls, torch.zeros(self.trans.dim_controls))),
                         True)
        self.assertIsInstance(h.def_module, dm.deformationmodules.DeformationModule)

    def test_apply_mom(self):
        pass

    def test_call(self):
        pass

    
