#!/usr/bin/python
"""Numerical estimation of rock permeability with the LBM.

quantity                symbol          SI unit
-----------------------------------------------
kinematic viscosity     nu              m^2 / s
fluid density           rho             kg / m^3
dynamic visosity        mu = nu * rho   Pa s
porosity                n               1
fluid velocity          u               m / s
Darcy flux              q = u * n       m / s
force density           F = dP / L      N / m^3
pressure difference     dP              Pa
length of flow domain   L               m
permeability            k               m^2

Darcy's law: q = -k / mu * (dP / L) = -k / mu * F

k = -u * n * rho * nu / F

n is defined as the fraction of fluid nodes in the rock domain.

Physical units vs LB units:

    u_phys = (dx / dt) u_lb
    nu_phys = (dx^2 / dt) nu_lb
    rho_phys = (dm / dx^3) rho_lb
    F_phys = (dm / dx^2 dt^2) F_lb

    k = -dx^5 dt^2 dm / (dm dx^3 dt^2) * n * u_lb * rho_lb * nu_lb / F_lb

Final formula to calculate permeability;

    k = -dx^2 * n * u_lb * rho_lb * nu_lb / F_lb
"""

import numpy as np

from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim, LBForcedSim
from sailfish.subdomain import Subdomain3D
from sailfish.node_type import NTFullBBWall
from sailfish.geo import EqualSubdomainsGeometry3D

class RockDomain(Subdomain3D):
    free_space = 5

    def initial_conditions(self, sim, hx, hy, hz):
        sim.rho[:] = 1.0

    def boundary_conditions(self, hx, hy, hz):
        wall_bc = NTFullBBWall

        x0 = np.min(hx)
        x1 = np.max(hx)
        y0 = np.min(hy)
        y1 = np.max(hy)
        z0 = np.min(hz)
        z1 = np.max(hz)

        lor = np.logical_or

        container = lor(lor(hx == 0, hx == self.gx - 1),
                        lor(hy == 0, hy == self.gy - 1))

        self.set_node(container, wall_bc)

        partial_wall_map = self.config._wall_map[z0:z1+1, y0:y1+1, x0:x1+1]
        self.set_node(partial_wall_map, wall_bc)


class RockSimulation(LBFluidSim, LBForcedSim):
    subdomain = RockDomain
    dx = 3.6e-6     # meters per voxel length
    visc = 1.0 / 6.0
    force = 0.001

    @classmethod
    def add_options(cls, group, dim):
        LBFluidSim.add_options(group, dim)
        LBForcedSim.add_options(group, dim)

        group.add_argument('--geometry', type=str,
                help='file defining the geometry')

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'conn_axis': 'z',
            'visc': cls.visc,
            'periodic_z': True,
        })

    @classmethod
    def modify_config(cls, config):
        assert config.geometry

        # Override lattice size based on the geometry file.
        wall_map = np.load(config.geometry)['arr_0']
        orig_shape = wall_map.shape
        config.lat_nz, config.lat_ny, config.lat_nx = orig_shape
        # Add space for walls.
        config.lat_nx += 2
        config.lat_ny += 2
        # Add some free space to faciliate PBC.
        fs = cls.subdomain.free_space
        config.lat_nz += fs * 2

        # TODO(michalj): Use np.pad here when numpy 1.7 is available.
        config._wall_map = np.zeros((config.lat_nz, config.lat_ny,
            config.lat_nx), dtype=np.bool)
        config._wall_map[fs:orig_shape[0] + fs,
                1:orig_shape[1] + 1,
                1:orig_shape[2] + 1] = wall_map

    def __init__(self, config):
        super(RockSimulation, self).__init__(config)
        self.add_body_force((0, 0, self.force), accel=False)


def permeability(geometry, rho, v):
    """Calculates permeability in Darcys."""

    fluid_nodes = np.sum(geometry == False)
    porosity = fluid_nodes / float(geometry.size)
    fs = RockSimulation.subdomain.free_space

    # Remove walls and free space around the rock sample.
    rho = rho[fs:-fs, 1:-1, 1:-1]
    v = v[fs:-fs, 1:-1, 1:-1]

    dx = RockSimulation.dx
    f = RockSimulation.force
    visc = RockSimulation.visc

    momentum = rho * v
    k = dx**2 * np.sum(momentum[geometry == False]) * porosity * visc / (f * fluid_nodes)

    return k / 9.869233e-13


if __name__ == '__main__':
    ctrl = LBSimulationController(RockSimulation, EqualSubdomainsGeometry3D)
    ctrl.run()
