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
        N = 100 #36

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
            'perf_stats_every': 500,
        })

    def __init__(self, config):
        super(CylinderSimulation, self).__init__(config)
        self.add_body_force((1e-7, 0.0))

    def after_step(self, runner):
        every = 20
        mod = self.iteration % every

        if mod == every - 1:
            self.need_sync_flag = True
        elif mod == 0:
            print self.iteration, self.vy[self.config.lat_ny / 2, self.config.lat_nx * 0.75]

        # TODO: compare against Lattice Boltzmann method on a curvilinear
        # coordinate system: Vortex shedding behind a circular cylinder
        # for Re = 50, 100, 150


        # meausure frequency: plot(np.fft.fftfreq(stat[:,1].size, d=20)[:100],
        # np.abs(np.fft.rfft(stat[:,1]))[:100])
        # find top peak, corresponding x value is freq (1/T)

        # Due to the alternating vortex wake (“Karman street”) the
        #oscillations in lift force occur at the vortex shedding frequency
        #and oscillations in drag force occur at twice the vortex
        #shedding frequency.

        #  Re 40 - 150: Laminar vortex street
        # < 3 e5: laminar boundary layer, turbulent wake
        # < 3.5e6 bounary layer transition to turbulent
        # > 3.5e6 tubulent vortex street

if __name__ == '__main__':
    ctrl = LBSimulationController(CylinderSimulation)
    ctrl.run()
