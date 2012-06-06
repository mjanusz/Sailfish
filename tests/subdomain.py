import numpy as np
import unittest
from sailfish import subdomain
from sailfish.backend_dummy import DummyBackend
from sailfish.config import LBConfig
from sailfish.lb_base import LBSim
from sailfish.node_type import NTEquilibriumVelocity, NTFullBBWall, multifield
from sailfish.subdomain import Subdomain2D, Subdomain3D, SubdomainSpec2D, SubdomainSpec3D, LBConnection
from sailfish.subdomain_runner import SubdomainRunner
from sailfish.sym import D2Q9, D3Q19
from dummy import *


class SubdomainTest2D(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        where = (hx == hy)
        self.set_node(where,
                NTEquilibriumVelocity(
                    multifield((0.01 * (hx - self.gy / 2)**2, 0.0), where)))

class TestNodeTypeSetting2D(unittest.TestCase):
    lattice_size = 64, 64

    def setUp(self):
        config = LBConfig()
        config.precision = 'single'
        config.block_size = 8
        # Does not affect behaviour of any of the functions tested here.
        config.lat_nx, config.lat_ny = self.lattice_size
        config.logger = DummyLogger()
        config.grid = 'D2Q9'
        self.sim = LBSim(config)
        self.config = config
        self.backend = DummyBackend()

    def test_array_setting(self):
        envelope = 1
        spec = SubdomainSpec2D((0, 0), self.lattice_size,
                envelope_size=envelope, id_=0)
        spec.runner = SubdomainRunner(self.sim, spec, output=None,
                backend=self.backend, quit_event=None)
        spec.runner._init_shape()
        sub = SubdomainTest2D(list(reversed(self.lattice_size)), spec, D2Q9)
        sub.reset()

        center = 64 / 2
        for y in range(0, 64):
            np.testing.assert_array_almost_equal(
                    np.float64([0.01 * (y - center)**2, 0.0]),
                    np.float64(sub._encoder.get_param((y + envelope, y + envelope), 2)))


class SubdomainTest3D(Subdomain3D):
    def boundary_conditions(self, hx, hy, hz):
        where = np.logical_and((hx == hy), (hy == hz))
        self.set_node(where,
                NTEquilibriumVelocity(
                    multifield((0.01 * (hy - self.gy / 2)**2,
                        0.03 * (hz - self.gz / 2)**2, 0.0), where)))

        wall = np.logical_and(hx == 31, hz > 7)
        self.set_node(wall, NTFullBBWall)

class TestNodeTypeSetting3D(unittest.TestCase):
    lattice_size = 32, 32, 16

    def setUp(self):
        config = LBConfig()
        config.precision = 'single'
        config.block_size = 8
        # Does not affect behaviour of any of the functions tested here.
        config.lat_nx, config.lat_ny, config.lat_nz = self.lattice_size
        config.logger = DummyLogger()
        config.grid = 'D3Q19'
        self.sim = LBSim(config)
        self.backend = DummyBackend()

    def make_sub(self, envelope):
        spec = SubdomainSpec3D((0, 0, 0), self.lattice_size,
                envelope_size=envelope, id_=0)
        spec.runner = SubdomainRunner(self.sim, spec, output=None,
                backend=self.backend, quit_event=None)
        spec.runner._init_shape()
        sub = SubdomainTest3D(list(reversed(self.lattice_size)), spec, D3Q19)
        sub.reset()
        return sub

    def test_array_setting(self):
        """Verifies that node type with parameters store the params correctly."""
        envelope = 1
        sub = self.make_sub(envelope)

        center_y = 32 / 2
        center_z = 16 / 2
        for y in range(0, 16):
            np.testing.assert_array_almost_equal(
                    np.float64([0.01 * (y - center_y)**2,
                        0.03 * (y - center_z)**2, 0.0]),
                    np.float64(sub._encoder.get_param(
                        (y + envelope, y + envelope, y + envelope), 3)))

    def test_interface_wet_map(self):
        """Verifies that wet nodes are correctly identified on the interface
        between two subdomains."""

        face = SubdomainSpec3D.X_HIGH

        # Overlap area is: (y, z): [10:32, 0:16] in global coordinates.
        # Source area is: [10:32, 0:16]
        # Dry area is: (y, z): [10:32, 8:]
        other = SubdomainSpec3D((32, 10, 0), (32, 32, 32), envelope_size=1, id_=1)
        sub = self.make_sub(envelope=1)
        connection = LBConnection.make(sub.spec, other, face, D3Q19)
        wet_map = sub.interface_wet_map(face, connection)
        self.assertEqual(np.sum(wet_map == True), 22 * 8)
        exp_map = np.zeros((16, 22), dtype=np.bool)
        exp_map[:8,:] = True
        np.testing.assert_array_equal(exp_map, wet_map)

class TestUtilFunctions(unittest.TestCase):

    def test_expand_slice(self):
        ss = SubdomainSpec3D((3, 13, 10), (64, 128, 256))

        slc = [slice(10, 14), slice(18, 35)]

        self.assertEqual(subdomain._expand_slice(slc, ss.X_HIGH, ss),
                [63, slice(10, 14), slice(18, 35)])
        self.assertEqual(subdomain._expand_slice(slc, ss.X_HIGH, ss, True),
                [66, slice(10, 14), slice(18, 35)])

        self.assertEqual(subdomain._expand_slice(slc, ss.X_LOW, ss),
                [0, slice(10, 14), slice(18, 35)])
        self.assertEqual(subdomain._expand_slice(slc, ss.X_LOW, ss, True),
                [3, slice(10, 14), slice(18, 35)])

        self.assertEqual(subdomain._expand_slice(slc, ss.Z_HIGH, ss),
                [slice(10, 14), slice(18, 35), 255])
        self.assertEqual(subdomain._expand_slice(slc, ss.Z_HIGH, ss, True),
                [slice(10, 14), slice(18, 35), 265])

if __name__ == '__main__':
    unittest.main()
