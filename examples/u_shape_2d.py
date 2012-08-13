#!/usr/bin/python

import numpy as np
from sailfish.controller import LBSimulationController
from sailfish.geo import LBGeometry2D
from sailfish.node_type import NTFullBBWall, NTEquilibriumVelocity
from sailfish.subdomain import SubdomainSpec2D, Subdomain2D
from sailfish.lb_single import LBFluidSim

class UGeometry(LBGeometry2D):
    def subdomains(self, n=None):
        subdomains = []

        dx = self.gx / 3
        dy = 2 * self.gy / 3

        subdomains.append(SubdomainSpec2D((0, 0), (dx, self.gy)))
        subdomains.append(SubdomainSpec2D((dx, dy), (dx, self.gy - dy)))
        subdomains.append(SubdomainSpec2D((2 *dx, 0), (self.gx - 2 * dx,
            self.gy)))

        return subdomains

class USubdomain(Subdomain2D):
    max_v = 0.05
    w = 20

    def boundary_conditions(self, hx, hy):
        dx = self.gx / 3
        dy = self.gy / 3
        w = self.w

        fluid = ((np.abs(hx - dx / 2) < w) | (np.abs(hx - 2.5 * dx) < w)) & (
                hy < 2.5 * dy + w)
        fluid |= (np.abs(hy - 2.5 * dy) < w) & (hx > dx / 2 - w) & (
                hx < 2.5 * dx + w)
        self.set_node(np.logical_not(fluid), NTFullBBWall)

        inlet = (hy == 0) & (np.abs(hx - dx / 2) < w)
        self.set_node(inlet,
                NTEquilibriumVelocity((0.0, self.max_v)))

    def initial_conditions(self, sim, hx, hy):
        dx = self.gx / 3
        sim.rho[:] = 1.0
        inlet = (hy == 0) & (np.abs(hx - dx / 2) < self.w)
        sim.vy[inlet] = self.max_v

class USim(LBFluidSim):
    subdomain = USubdomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 256,
            'lat_ny': 256})


if __name__ == '__main__':
    ctrl = LBSimulationController(USim, UGeometry)
    ctrl.run()
