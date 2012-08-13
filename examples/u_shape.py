#!/usr/bin/python -u
"""Demonstrates how to load geometry from a boolean numpy array."""

import numpy as np

from sailfish.subdomain import Subdomain3D
from sailfish.node_type import NTFullBBWall, NTEquilibriumVelocity
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim
from examples.u_shape_3d import UGeometry

class USubdomain(Subdomain3D):
    max_v = 0.05

    def _maps(self, hx, hy, hz):
        x0 = np.min(hx)
        x1 = np.max(hx)
        y0 = np.min(hy)
        y1 = np.max(hy)
        z0 = np.min(hz)
        z1 = np.max(hz)

        partial_wall_map = self.config._wall_map[z0:z1+1, y0:y1+1, x0:x1+1]

        if x0 == 0 and y0 == 0:
            inlet_map = np.logical_not(partial_wall_map) & (hy == 0)
        else:
            inlet_map = None

        if x1 == self.gx - 1 and y0 == 0:
            outlet_map = np.logical_not(partial_wall_map) & (hy == 0)
        else:
            outlet_map = None

        return partial_wall_map, inlet_map, outlet_map

    def boundary_conditions(self, hx, hy, hz):
        if not hasattr(self.config, '_wall_map'):
            return

        wall_map, inlet_map, outlet_map = self._maps(hx, hy, hz)

        self.set_node(wall_map, NTFullBBWall)
        if inlet_map is not None:
            self.set_node(inlet_map, NTEquilibriumVelocity((0.0, self.max_v, 0.0)))

    def initial_conditions(self, sim, hx, hy, hz):
        sim.rho[:] = 1.0
        wall_map, inlet_map, outlet_map = self._maps(hx, hy, hz)
        if inlet_map is not None:
            sim.vy[inlet_map] = self.max_v

class UShapeSim(LBFluidSim):
    subdomain = USubdomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'grid': 'D3Q19'
            })

    @classmethod
    def add_options(cls, group, dim):
        LBFluidSim.add_options(group, dim)

        group.add_argument('--geometry', type=str,
                default='ushape3.npy',
                help='file defining the geometry')

    @classmethod
    def modify_config(cls, config):
        if not config.geometry:
            return

        # Override lattice size based on the geometry file.
        wall_map = np.load(config.geometry)
        wall_map = np.rollaxis(wall_map, 2)  # make z the smallest dimension
        wall_map = wall_map[:,1:,:]          # remove wall blocking the inlet/outlet

        config.lat_nz, config.lat_ny, config.lat_nx = wall_map.shape
        config._wall_map = wall_map


if __name__ == '__main__':
    LBSimulationController(UShapeSim, UGeometry).run()
