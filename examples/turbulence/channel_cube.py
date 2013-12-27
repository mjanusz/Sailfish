#!/usr/bin/env python
"""
Turbulent channel flow around a wall-mounted cube.

The simulation is driven by a fully developed turbulent channel
flow simulated in a separate subdomain (id=0), called the recirculation
buffer. This subdomain has periodic boundary conditions enabled in the
streamwise direction and is completely independent of the main simulation
area.

  buffer:               main:
=============|==========================O
            R|R                         O
            R|R                         O
            R|R                         O
            R|R                         O
=============|==========================O

Legend:
 | - subdomain boundary
 = - wall
 R - replicated node (all distributions are synced from buffer to main after
     every step
 O - outflow nodes
"""

# TODO:
#  single axis averaging
#  measurement point for Strouhal

import math
import tempfile
import zmq
import numpy as np
from sailfish.node_type import NTHalfBBWall, NTDoNothing, NTCopy, NTEquilibriumDensity, NTFullBBWall
from sailfish.geo import LBGeometry3D
from sailfish.controller import LBSimulationController, GeometryError
from sailfish.subdomain import Subdomain3D, SubdomainSpec3D
from sailfish.stats import ReynoldsStatsMixIn
from sailfish.subdomain_runner import SubdomainRunner
from sailfish.lb_single import LBFluidSim
from sailfish.lb_base import LBForcedSim

from channel_flow import ChannelSubdomain, ChannelSim
import scipy.ndimage.filters


class CubeChannelGeometry(LBGeometry3D):
    @classmethod
    def add_options(cls, group):
        LBGeometry3D.add_options(group)
        group.add_argument('--subdomains', help='number of subdomains for '
                           'the real simulation region',
                           type=int, default=1)

    @classmethod
    def buf_nz(cls, config):
        return int(config.buf_az * config.H * 2 / 3)

    @classmethod
    def main_nz(cls, config):
        return int(config.main_az * config.H * 2 / 3)

    def subdomains(self):
        c = self.config

        # Recirculation buffer.
        buf = SubdomainSpec3D((0, 0, 0), (c.lat_nx, c.lat_ny, self.buf_nz(c)))
        # Enable PBC along the Z axis.
        buf.enable_local_periodicity(2)
        ret = [buf]

        # Actual simulation domain.
        n = self.config.subdomains
        z = self.buf_nz(c)
        dz = self.main_nz(c) / n
        rz = self.main_nz(c) % n
        for i in range(0, n):
            ret.append(SubdomainSpec3D((0, 0, z),
                                       (c.lat_nx, c.lat_ny, dz if i < n - 1 else dz + rz)))
            z += dz
        return ret


class CubeChannelSubdomain(ChannelSubdomain):
    wall_bc = NTFullBBWall

    def boundary_conditions(self, hx, hy, hz):
        # Channel walls.
        wall_map = ((hx == 0) | (hx == self.gx - 1))
        self.set_node(wall_map, self.wall_bc)

        h = self.config.H * 2 / 3
        buf_len = CubeChannelGeometry.buf_nz(self.config)

        # Cube.
        cube_map = ((hx > 0) & (hx < h) &
                    (hz >= buf_len + 3 * h) & (hz < buf_len + 4 * h) &
                    (hy >= 2.7 * h) & (hy < 3.7 * h))
        self.set_node(cube_map, self.wall_bc)

        # Outlet
        outlet_map = (hz == self.gz - 1) & np.logical_not(wall_map)
        self.set_node(outlet_map, NTEquilibriumDensity(1.0))


def ceil_div(x, y):
    return (x + y - 1) / y


class CubeChannelSubdomainRunner(SubdomainRunner):
    def _init_distrib_kernels(self, *args, **kwargs):
        # No distribution in the recirculation buffer.
        if self._spec.id == 0:
            return [], []
        else:
            return super(CubeChannelSubdomainRunner, self)._init_distrib_kernels(
                *args, **kwargs)


class CubeChannelSim(ChannelSim):
    subdomain = CubeChannelSubdomain
    subdomain_runner = CubeChannelSubdomainRunner

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'access_pattern': 'AA',
            'grid': 'D3Q19',
            'force_implementation': 'guo',
            'model': 'bgk',
            'minimize_roundoff': True,
            'precision': 'single',
            'seed': 11341351351,
            'periodic_y': True,
            'periodic_z': True,

            # Performance tuning.
            'check_invalid_results_gpu': True,
            'block_size': 128,

            # Output.
            'max_iters': 3500000,
            'every': 200000,
            'perf_stats_every': 5000,
            'final_checkpoint': True,
            'checkpoint_every': 500000,
            })

    @classmethod
    def modify_config(cls, config):
        h = 2 if cls.subdomain.wall_bc.location == 0.5 else 0

        # 1/3 of the channel height
        cube_h = config.H * 2 / 3

        config.lat_nx = config.H * 2 + h  # wall normal
        config.lat_ny = int(config.ay * cube_h)  # spanwise (PBC)
        config.lat_nz = (int(config.buf_az * cube_h) +
                         int(config.main_az * cube_h))  # streamwise
        config.visc = cls.subdomain.u_tau(config.Re_tau) * config.H / config.Re_tau

        cls.show_info(config)

    @classmethod
    def show_info(cls, config):
        cube_h = config.H * 2 / 3
        print 'cube:   %d' % cube_h
        print 'buffer: %d x %d x %d' % (int(config.buf_az * cube_h), config.lat_ny, config.lat_nx)
        print 'main:   %d x %d x %d' % (int(config.main_az * cube_h), config.lat_ny, config.lat_nx)
        ChannelSim.show_info(config)

    @classmethod
    def add_options(cls, group, dim):
        ChannelSim.add_options(group, dim)
        group.add_argument('--buf_az', type=float, default=9.0)
        group.add_argument('--main_az', type=float, default=8.0)
        group.add_argument('--ay', type=float, default=6.4)

    def before_main_loop(self, runner):
        return

############################################################
# Disabled code blow this line
############################################################

        if runner._spec.id == 1:
            self._ntcopy_kernel = runner.get_kernel(
                'HandleNTCopyNodes',
                [runner.gpu_geo_map(), runner.gpu_dist(0, 0)],
                'PP', needs_iteration=True)

        self._prepare_reynolds_stats_global(runner)

    def _prepare_reynolds_stats_slice(self, runner):
        num_stats = 3 * 2 + 3
        self._stats_bufs = []
        self._gpu_stats_bufs = []
        self._x_stats_kern = []
        c = self.config

        for i in range(5):
            bufs = []
            gpu_bufs = []

            for j in range(num_stats):
                h = np.zeros([c.lat_ny * c.lat_nz], dtype=np.float64)
                bufs.append(h)
                gpu_bufs.append(runner.backend.alloc_buf(like=h))

            self._stats_bufs.append(bufs)
            self._gpu_stats_bufs.append(gpu_bufs)

        bufs = []
        gpu_bufs = []
        for j in range(num_stats):
            h = np.zeros([c.lat_nx * c.lat_nz], dtype=np.float64)
            bufs.append(h)
            gpu_bufs.append(runner.backend.alloc_buf(like=h))

        self._stats_bufs.append(bufs)
        self._gpu_stats_bufs.append(gpu_bufs)

        gpu_v = runner.gpu_field(self.v)
        for i, y in enumerate((0, 0.1 * c.lat_ny, 0.25 * c.lat_ny, 0.5 * c.lat_ny, 0.75 *
                  c.lat_ny)):
            k = runner.get_kernel('ReynoldsX64', [int(y)] + gpu_v +
                                  self._gpu_stats_bufs[i], 'iPPP' + 'P' *
                                  num_stats, block_size=(128,))
            self._x_stats_kern.append(k)

        self._y_stats_kern = runner.get_kernel(
            'ReynoldsY64', [c.lat_ny / 2] + gpu_v + self._gpu_stats_bufs[-1],
            'iPPP' + 'P' * num_stats, block_size=(128,))

    def _prepare_reynolds_stats_global(self, runner):
        num_stats = 3 * 2 + 3

        self._stats = []
        self._gpu_stats = []

        for i in range(0, num_stats):
            f = runner.make_scalar_field(dtype=np.float64, register=False)
            f[:] = 0.0
            self._stats.append(f)
            self._gpu_stats.append(runner.backend.alloc_buf(like=runner.field_base(f)))

        gpu_v = runner.gpu_field(self.v)
        self._stats_kern = runner.get_kernel(
            'ReynoldsGlobal', gpu_v + self._gpu_stats, 'PPP' + 'P' * num_stats)



    num_stats = 0
    def _collect_stats(self, runner):
#        runner.backend.run_kernel(self._y_stats_kern, [ceil_div()
        runner.backend.run_kernel(self._stats_kern, runner._kernel_grid_full)
        self.num_stats += 1

        if self.num_stats == 10000:
            for gpu_buf in self._gpu_stats:
                runner.backend.from_buf(gpu_buf)
            np.savez('%s_reyn_stat_%s.%s' % (self.config.output, runner._spec.id,
                                             self.iteration),
                     ux_m1=self._stats[0],
                     ux_m2=self._stats[1],
                     uy_m1=self._stats[2],
                     uy_m2=self._stats[3],
                     uz_m1=self._stats[4],
                     uz_m2=self._stats[5],
                     ux_uy=self._stats[6],
                     ux_uz=self._stats[7],
                     uy_uz=self._stats[8])

            self.num_stats = 0
            for buf in self._stats:
                buf[:] = 0.0
            for gpu_buf in self._gpu_stats:
                runner.backend.to_buf(gpu_buf)

    def after_step(self, runner):
        return

        # Handle NTCopy nodes in the AA access pattern.
        if self.config.access_pattern == 'AA' and runner._spec.id == 1:
            runner.backend.run_kernel(self._ntcopy_kernel,
                                      [ceil_div(NX + 2, self.config.block_size), NY + 2])


# averaged data on slices:
#  - symmetry
#  - y/h = 0.003, 0.1, 0.25, 0.5, 0.75

# reynolds stresses:
#  - symmetry plane
#  - TKE at y/h = 0.1m 0.25, 0.5, 0.75
#  -> TKE = 0.5 (u'^2 + v'^2 + w'^2)

# average data on lines:
#  - streamwise velocity in the symmetry plane


        if self.iteration < 500000 or runner._spec.id != 1:
            return

        every = 10
        mod = self.iteration % every

        if mod == every - 1:
            self.need_fields_flag = True
        elif mod == 0:
            self._collect_stats(runner)


if __name__ == '__main__':
#    TMPDIR = tempfile.mkdtemp()
    ctrl = LBSimulationController(CubeChannelSim, CubeChannelGeometry)
    ctrl.run()

