#!/usr/bin/python

import numpy as np
from sailfish.controller import LBSimulationController
from sailfish.geo import LBGeometry3D
from sailfish.node_type import NTFullBBWall, NTEquilibriumVelocity
from sailfish.subdomain import SubdomainSpec3D, Subdomain3D
from sailfish.lb_single import LBFluidSim

class UGeometry(LBGeometry3D):
    def subdomains(self, n=None):
        subdomains = []

        dx = self.gx / 3
        dy = 2 * self.gy / 3

        subdomains.append(SubdomainSpec3D((0, 0, 0), (dx, self.gy, self.gz)))
        subdomains.append(SubdomainSpec3D((dx, dy, 0), (dx, self.gy - dy, self.gz)))
        subdomains.append(SubdomainSpec3D((2 * dx, 0, 0), (self.gx - 2 * dx,
            self.gy, self.gz)))

        return subdomains

class USubdomain(Subdomain3D):
    max_v = 0.05
    w = 10

    def boundary_conditions(self, hx, hy, hz):
        dx = self.gx / 3
        dy = self.gy / 3
        w = self.w

        r_z_sq = np.square(hz - self.gz/2.0)

        r1 = np.sqrt(np.square(hx - dx/2.0) + r_z_sq) < w
        r2 = np.sqrt(np.square(hx - 2.5 * dx) + r_z_sq) < w
        r3 = np.sqrt(np.square(hy - 2.5 * dy) + r_z_sq) < w

        fluid = (r1 | r2) & ((hy < 2.5 * dy) | r3)
        fluid |= r3 & ((hx > dx / 2) | r1) & ((hx < 2.5 * dx) | r2)

        self.set_node(np.logical_not(fluid), NTFullBBWall)

        inlet = (hy == 0) & r1
        self.set_node(inlet,
                NTEquilibriumVelocity((0.0, self.max_v, 0.0)))

    def initial_conditions(self, sim, hx, hy, hz):
        dx = self.gx / 3
        r_z_sq = np.square(hz - self.gz/2.0)
        r1 = np.sqrt(np.square(hx - dx/2.0) + r_z_sq) < self.w
        sim.rho[:] = 1.0
        inlet = (hy == 0) & r1
        sim.vy[inlet] = self.max_v

class USim(LBFluidSim):
    subdomain = USubdomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 120,
            'lat_ny': 120,
            'lat_nz': 60,
            'grid': 'D3Q15'})


if __name__ == '__main__':
    ctrl = LBSimulationController(USim, UGeometry)
    ctrl.run()
