import numpy as np

from sailfish.controller import LBSimulationController
from sailfish.geo import EqualSubdomainsGeometry3D
from sailfish.vis_mixin import Vis2DSliceMixIn

import common

class UshapeSubdomain(common.InflowOutflowSubdomain):
    def _inflow_outflow(self, hx, hy, hz, wall_map):
        inlet = None
        outlet = None

        if np.min(hx) == -1 and np.min(hy) == -1:
            inlet = np.logical_not(wall_map) & (hy == 0) & (hx < self.gx / 2)

        if np.max(hx) == self.gx and np.min(hy) == -1:
            outlet = np.logical_not(wall_map) & (hy == 0) & (hx > self.gx / 2)

        return inlet, outlet


class UshapeSim(common.HemoSim, Vis2DSliceMixIn):
    subdomain = UshapeSubdomain
    phys_diam = 2.54e-2

    @classmethod
    def update_defaults(cls, defaults):
        super(UshapeSim, cls).update_defaults(defaults)
        defaults.update({
            'max_iters': 2500000,
            'every': 200000,

            'log': 'ushape.log',
            'checkpoint_file': 'ushape',
            'output': 'ushape',

            # Subdomains configuration.
            'subdomains': 4,
            'conn_axis': 'y',
            'geometry': 'ushape.py',
            'reynolds': 100,
        })

    @classmethod
    def modify_config(cls, config):
        if not config.geometry:
            return

        wall_map = np.load(config.geometry)
        wall_map = np.rollaxis(wall_map, 2)  # make z the smallest dimension

        # Remove wall blocking the inlet/outlet
        # 2002: 17,  1669: 14,  1002: 9,  802: 7
        wall_map = wall_map[:,7:,:]

        # Smooth out the wall...
        # 2002: 1500,  1669: 1270,  1002: 768,  802: 610
        for i in range(1, 610):
            wall_map[:,i,:] = wall_map[:,0,:]

        # Make it symmetric.
        # 2002: 459,  1669: 394,  1002: 230,  802: 184
        wall_map[:,:,-184:] = wall_map[:,:,:184][:,:,::-1]

        # Override lattice size based on the geometry file.
        config.lat_nz, config.lat_ny, config.lat_nx = wall_map.shape

        # Geometry is:
        # - diameter: 1in
        # - bow radius: 3in
        # - arm length: 10in

        # U_avg = 1.3111e-2 m/s, U_max = 2*U_avg
        # visc = 3.33 e-6
        # D_h = 2.54e-2 m
        #
        # Flow length: 1860 [lattice units] (2 * L + pi * R_1 + 0.5) * 60
        # The oscillation period (in lattice units) should be significantly longer
        # than this so that pressure wave propagation effects are not visible.

        # Add ghost nodes.
        wall_map = np.pad(wall_map, (1, 1), 'constant', constant_values=True)
        config._wall_map = wall_map

        super(UshapeSim, cls).modify_config(config)

if __name__ == '__main__':
    LBSimulationController(UshapeSim, EqualSubdomainsGeometry3D).run()
