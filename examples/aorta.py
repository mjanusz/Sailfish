#!/usr/bin/python -u
"""Simulates pulsatile blood flow through an aorta."""

import numpy as np

from sailfish.geo import EqualSubdomainsGeometry3D
from sailfish.subdomain import Subdomain3D
from sailfish.node_type import NTFullBBWall, NTEquilibriumVelocity, DynamicValue, multifield
from sailfish.controller import LBSimulationControlle
from sailfish.lb_single import LBFluidSim
from sailfish import sym

class AortaSubdomain(Subdomain3D):

    inflow_velocity = -0.02

    def maps(self, hx, hy, hz):
        x0 = np.min(hx)
        x1 = np.max(hx)
        y0 = np.min(hy)
        y1 = np.max(hy)
        z0 = np.min(hz)
        z1 = np.max(hz)

        partial_wall_map = self.config._wall_map[z0:z1+1, y0:y1+1, x0:x1+1]

        if x1 == self.gx - 1:
            inflow_map = (np.logical_not(partial_wall_map) & (hx == x1))
        else:
            inflow_map = None

        return partial_wall_map, inflow_map

    def velocity_profile(self, hx, hy, hz, partial_wall_map, inflow_map):
        # This assumes that the whole inlet is contained in the current subdomain.
        inflow_plane = partial_wall_map[:,:,-1]

        # Locate the geometric center of the inlet.
        z, y = np.where(np.logical_not(inflow_plane))
        zm = (min(z) + max(z)) / 2
        ym = (min(y) + max(y)) / 2
        diam = min(max(z) - min(z), max(y) - min(y))

        r = np.sqrt((hz - hz[zm, 0, 0])**2 + (hy - hy[0, ym, 0])**2)
        v = self.inflow_velocity / (diam / 2.0)**2 * ((diam / 2.0)**2 - r**2)
        return v

    def initial_conditions(self, sim, hx, hy, hz):
        sim.rho[:] = 1.0
        partial_wall_map, inflow_map = self.maps(hx, hy, hz)
        if inflow_map is not None:
            v = self.velocity_profile(hx, hy, hz, partial_wall_map, inflow_map)
            sim.vx[inflow_map] = v[inflow_map]

    def boundary_conditions(self, hx, hy, hz):
        wall_bc = NTFullBBWall
        partial_wall_map, inflow_map = self.maps(hx, hy, hz)
        self.set_node(partial_wall_map, wall_bc)
        if inflow_map is not None:
            # Simplest (unphysical) case: constant inflow velocity everywhere at the inlet.
            # self.set_node(inflow_map, NTEquilibriumVelocity((self.inflow_velocity, 0.0, 0.0)))

            v = self.velocity_profile(hx, hy, hz, partial_wall_map, inflow_map)
            self.set_node(inflow_map, NTEquilibriumVelocity(multifield((v, 0.0, 0.0), inflow_map)))

            # sym.gy, sym.gz -> coordinates for DynamicValue
            # sym.time


class AortaSimulation(LBFluidSim):
    subdomain = AortaSubdomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'grid': 'D3Q19',
            })

    @classmethod
    def add_options(cls, group, dim):
        LBFluidSim.add_options(group, dim)

        group.add_argument('--geometry', type=str,
                help='file defining the geometry')

    @classmethod
    def modify_config(cls, config):
        if not config.geometry:
            return

        wall_map = np.load(config.geometry)
        # Get rid of the walls perpendicular to the direction of the flow.
        wall_map = wall_map[:,:,1:-1]
        # Override lattice size based on the geometry file.
        config.lat_nz, config.lat_ny, config.lat_nx = wall_map.shape
        config._wall_map = wall_map


if __name__ == '__main__':
    LBSimulationController(AortaSimulation, EqualSubdomainsGeometry3D).run()
