import ctypes
import operator
import os
import unittest
import numpy as np
import zmq

from sailfish.config import LBConfig
from sailfish.connector import ZMQSubdomainConnector
from sailfish.lb_base import LBSim
from sailfish.lb_single import LBFluidSim
from sailfish.lb_binary import LBBinaryFluidShanChen
from sailfish.backend_dummy import DummyBackend
from sailfish.subdomain_runner import SubdomainRunner, NNSubdomainRunner
from sailfish.subdomain import SubdomainSpec2D, SubdomainSpec3D, Subdomain3D
from sailfish.io import LBOutput
from sailfish.sym import D2Q9, D3Q19
from sailfish.node_type import NTFullBBWall
from sailfish.util import mor, mand
from dummy import *

class SubdomainRunnerTestCase(unittest.TestCase):
    def setUp(self):
        config = LBConfig()
        config.precision = 'single'
        config.block_size = 8
        config.logger = DummyLogger()
        config.bulk_boundary_split = False
        config.output = ''
        config.grid = 'D2Q9'
        self.config = config
        self.backend = DummyBackend()
        self.ctx = zmq.Context()

    def _connect_subdomains(self, ss1, ss2, grid):
        """
        :param ss1: SubdomainSpec
        :param ss2: SubdomainSpec
        :param grid: a DxQy-class object
        """
        # Establish connection between ss1 and ss2..
        self.assertTrue(ss1.connect(ss2, grid=grid))
        cpair = ss1.get_connection(*ss1.connecting_subdomains()[0])
        c1, c2 = ZMQSubdomainConnector.make_ipc_pair((ss1.id, ss2.id))
        ss1.add_connector(ss2.id, c1)
        ss2.add_connector(ss1.id, c2)
        return c1, c2


class TestDomain(Subdomain3D):
    def boundary_conditions(self, hx, hy, hz):
        self.set_node(mand(hx > 10, hy > 10), NTFullBBWall)

class TestFluidSim(LBFluidSim):
    subdomain = TestDomain


class BasicFunctionalityTest(SubdomainRunnerTestCase):
    location = 0, 0
    size = 10, 3

    location_3d = 0, 0, 0
    size_3d = 3, 5, 7

    def setUp(self):
        super(BasicFunctionalityTest, self).setUp()
        self.config.lat_nz, self.config.lat_nx, self.config.lat_ny = self.size_3d
        self.config.grid = 'D3Q19'
        self.sim = LBSim(self.config)

    def get_subdomain_runner(self, block):
        return SubdomainRunner(self.sim, block, output=None,
                backend=self.backend, quit_event=None)

    def test_block_connection(self):
        block = SubdomainSpec2D(self.location, self.size)
        runner = self.get_subdomain_runner(block)
        self.assertEqual(block.runner, runner)

    def test_strides_and_size_2d(self):
        block = SubdomainSpec2D(self.location, self.size)
        block.set_actual_size(0)
        runner = self.get_subdomain_runner(block)
        runner._init_shape()

        # Last dimension is rounded up to a multiple of block_size
        real_size = [3, 16]
        self.assertEqual(runner._physical_size, real_size)

        strides = runner._get_strides(np.float32)
        self.assertEqual(strides, [4 * 16, 4])
        nodes = runner._get_nodes()
        self.assertEqual(nodes, reduce(operator.mul, real_size))

    def test_strides_and_size_3d(self):
        block = SubdomainSpec3D(self.location_3d, self.size_3d)
        block.set_actual_size(0)
        runner = self.get_subdomain_runner(block)
        runner._init_shape()

        # Last dimension is rounded up to a multiple of block_size
        real_size = [7, 5, 8]
        self.assertEqual(runner._physical_size, real_size)
        strides = runner._get_strides(np.float64)
        self.assertEqual(strides, [8*8*5, 8*8, 8])
        nodes = runner._get_nodes()
        self.assertEqual(nodes, reduce(operator.mul, real_size))

    def test_wet_map_interchange(self):
        """Verifies that the maps of wet nodes on the interface between
        subdomains are correctly exchanged between two subdomain runners."""
        ss1 = SubdomainSpec3D((0, 0, 0), (20, 20, 20), envelope_size=1, id_=0)
        ss2 = SubdomainSpec3D((0, 0, 20), (20, 20, 20), envelope_size=1, id_=1)

        c1, c2 = self._connect_subdomains(ss1, ss2, D3Q19)

        # Create simulation object and block runnners.
        sim1 = TestFluidSim(self.config)
        sim2 = TestFluidSim(self.config)
        sr1 = SubdomainRunner(sim1, ss1, output=LBOutput(self.config, ss1.id),
                backend=self.backend, quit_event=DummyEvent())
        sr2 = SubdomainRunner(sim2, ss2, output=LBOutput(self.config, ss2.id),
                backend=self.backend, quit_event=DummyEvent())

        # Initialize a local IPC connection between the subdomains.
        c1.init_runner(self.ctx)
        c2.init_runner(self.ctx)

        sr1._init_geometry()
        sr2._init_geometry()

        sr1._send_wet_map()
        sr2._send_wet_map()
        sr1._recv_wet_map()
        sr2._recv_wet_map()

        os.unlink(c1.ipc_file)

class NNSubdomainRunnerTest(SubdomainRunnerTestCase):

    def setUp(self):
        super(NNSubdomainRunnerTest, self).setUp()
        self.config.lat_nx, self.config.lat_ny = 40, 80
        self.config.grid = 'D2Q9'

    def tearDown(self):
        try:
            self.ctx.destroy()
        # Workaround for differences between zmq versions.
        except AttributeError:
            pass

    def test_macro_transfer_2d(self):
        """Verifies that macroscopic fields are correctly exchanged between two
        2D subddomains."""
        b1 = SubdomainSpec2D((0, 0), (40, 40), id_=0)
        b2 = SubdomainSpec2D((0, 40), (40, 40), id_=1)
        b1.set_actual_size(envelope_size=1)
        b2.set_actual_size(envelope_size=1)

        c1, c2 = self._connect_subdomains(b1, b2, D2Q9)

        # Create simulation object and block runnners.
        sim1 = LBBinaryFluidShanChen(self.config)
        sim2 = LBBinaryFluidShanChen(self.config)
        br1 = NNSubdomainRunner(sim1, b1, output=LBOutput(self.config, b1.id),
                backend=self.backend,
                quit_event=DummyEvent())
        br2 = NNSubdomainRunner(sim2, b2, output=LBOutput(self.config, b2.id),
                backend=self.backend,
                quit_event=DummyEvent())

        br1._init_shape()
        br2._init_shape()
        sim1.init_fields(br1)
        sim2.init_fields(br2)
        br1._init_buffers()
        br2._init_buffers()
        br1._init_streams()
        br2._init_streams()

        # Initialize a local IPC connection between the subdomains.
        c1.init_runner(self.ctx)
        c2.init_runner(self.ctx)

        # Verify transfer of the macroscopic fields.
        rho_cbuf, phi_cbuf = br1._block_to_macrobuf[1]
        rho = rho_cbuf.coll_buf.host
        phi = phi_cbuf.coll_buf.host
        rho[:] = np.mgrid[0:len(rho)]
        phi[:] = np.mgrid[100:100 + len(phi)]

        br1._send_macro()
        br2._recv_macro()

        rho_recv, phi_recv = br2._block_to_macrobuf[0]
        np.testing.assert_equal(rho_recv.recv_buf.host, rho)
        np.testing.assert_equal(phi_recv.recv_buf.host, phi)

        # Verify transfer of the distributions.
        f_cbuf, g_cbuf = br1._block_to_connbuf[1]
        fdist = f_cbuf.coll_buf.host
        gdist = g_cbuf.coll_buf.host
        fdist.flat = np.mgrid[0:len(fdist)]
        gdist.flat = np.mgrid[500:500 + len(gdist)]

        br1._send_dists()
        br2._recv_dists()

        f_recv, g_recv = br2._block_to_connbuf[0]
        np.testing.assert_equal(f_recv.recv_buf, fdist)
        np.testing.assert_equal(g_recv.recv_buf, gdist)

        os.unlink(c1.ipc_file)

if __name__ == '__main__':
    unittest.main()
