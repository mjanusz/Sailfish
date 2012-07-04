#!/usr/bin/python

import numpy as np

from sailfish.geo import EqualSubdomainsGeometry2D
from sailfish.subdomain import Subdomain2D
from sailfish.node_type import NTHalfBBWall
from sailfish.controller import LBSimulationController
from sailfish.lb_binary import LBBinaryFluidFreeEnergy
from sailfish.lb_base import LBForcedSim

class MicrochannelDomain(Subdomain2D):
    def initial_conditions(self, sim, hx, hy):
        sim.rho[:] = 1.0
        sim.phi[:] = 1.0

        length = 500
        width = 10

        phase_map = ((hx > self.gx / 2 - length) &
            (hx <= self.gx / 2 + length) &
            (hy > width) &
            (hy <= self.gy - width - 1))
        sim.phi[phase_map] = -1.0

    def boundary_conditions(self, hx, hy):
        wall_map = (hy <= 1) | (hy >= self.gy - 2)
        self.set_node(wall_map, NTHalfBBWall)

class MicrochannelSim(LBBinaryFluidFreeEnergy, LBForcedSim):
    subdomain = MicrochannelDomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'bc_wall_grad_order': 1,
            'lat_nx': 3000,
            'lat_ny': 204,
            'tau_a': 0.7,   # gas
            'tau_b': 2.5,   # liquid
            'tau_phi': 1.0,
            'Gamma': 1.0,
            'kappa': 4e-4,
            'A': 4e-2,
            'periodic_x': True,
            'periodic_y': True})

    def __init__(self, config):
        super(MicrochannelSim, self).__init__(config)

        self.add_body_force((6.0e-6, 0.0), grid=0, accel=False)

        # Use the fluid velocity in the relaxation of the order parameter field,
        # and the molecular velocity in the relaxation of the density field.
        self.use_force_for_equilibrium(None, target_grid=0)
        self.use_force_for_equilibrium(0, target_grid=1)


if __name__ == '__main__':
    ctrl = LBSimulationController(MicrochannelSim,
            EqualSubdomainsGeometry2D)
    ctrl.run()
