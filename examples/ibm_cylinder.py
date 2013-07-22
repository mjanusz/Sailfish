#!/usr/bin/env python -u

import math
import numpy as np
from sailfish.subdomain import Subdomain2D
from sailfish.node_type import NTFullBBWall, NTEquilibriumDensity, NTCopy, NTRegularizedVelocity
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBIBMFluidSim, Particle

class CylinderSubdomain(Subdomain2D):

    def boundary_conditions(self, hx, hy):
        wall_map = (hy == 0) | (hy == self.gy - 1)
        self.set_node(wall_map, NTFullBBWall)
        self.set_node(np.logical_not(wall_map) & (hx == 0),
                      NTRegularizedVelocity((0.03, 0.0)))
        self.set_node(np.logical_not(wall_map) & (hx == self.gx - 1),
                      NTCopy)

    def initial_conditions(self, sim, hx, hy):
        sim.rho[:] = 1.0
        sim.vy[:] = 0.0
        sim.vx[:] = 0.0

        cx = 0.25 * self.config.lat_nx
        cy = 0.5 * self.config.lat_ny
        r = 10
        N = 40 #36

        for i in range(0, N):
            x = cx + r * math.cos(i / float(N) * 2.0 * math.pi)
            y = cy + r * math.sin(i / float(N) * 2.0 * math.pi)
            sim.add_particle(Particle((x, y), stiffness=0.01,
                                      ref_position=(x, y)))

class CylinderSimulation(LBIBMFluidSim):
    subdomain = CylinderSubdomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 512,
            'lat_ny': 128,
            #'periodic_x': True,
            'visc': 0.01,
            'perf_stats_every': 500,})

    def __init__(self, config):
        super(CylinderSimulation, self).__init__(config)
        self.add_body_force((1e-7, 0.0))


if __name__ == '__main__':
    ctrl = LBSimulationController(CylinderSimulation)
    ctrl.run()
