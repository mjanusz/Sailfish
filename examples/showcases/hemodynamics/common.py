import math
import numpy as np

from sailfish.node_type import NTRegularizedVelocity, DynamicValue, NTDoNothing, LinearlyInterpolatedTimeSeries, NTFullBBWall
from sailfish.subdomain import Subdomain3D
from sailfish.lb_single import LBFluidSim
from sailfish.sym import S, D3Q19
from sympy import sin

class UnitConverter(object):
    """Performs unit conversions."""

    def __init__(self, visc=None, length=None, velocity=None, Re=None, freq=None):
        """Initializes the converter.

        :param visc: physical viscosity
        :param length: physical reference length
        :param velocity: physical reference velocity
        :param Re: Reynolds number
        :param freq: physical frequency
        """
        self._phys_visc = visc
        self._phys_len = length
        self._phys_vel = velocity
        self._phys_freq = freq

        if Re is not None:
            if visc is None:
                self._phys_visc = length * velocity / Re
            elif length is None:
                self._phys_len = Re * visc / velocity
            elif velocity is None:
                self._phys_vel = Re * visc / length

        self._lb_visc = None
        self._lb_len = None
        self._lb_vel = None

    def set_lb(self, visc=None, length=None, velocity=None):
        if visc is not None:
            self._lb_visc = visc
        if length is not None:
            self._lb_len = length
        if velocity is not None:
            self._lb_vel = velocity

        self._update_missing_parameters()

    def _update_missing_parameters(self):
        if (self._lb_visc is None and self._lb_len is not None and
            self._lb_vel is not None):
            self._lb_visc = self._lb_len * self._lb_vel / self.Re
            return

        if (self._lb_len is None and self._lb_visc is not None and
            self._lb_vel is not None):
            self._lb_len = self.Re * self._lb_visc / self._lb_vel
            return

        if (self._lb_vel is None and self._lb_len is not None and
            self._lb_visc is not None):
            self._lb_vel = self.Re * self._lb_visc / self._lb_len
            return

    @property
    def Re(self):
        return self._phys_len * self._phys_vel / self._phys_visc

    @property
    def Womersley(self):
        return math.sqrt(self._phys_freq * self._phys_len**2 / self.phys_visc)

    @property
    def Re_lb(self):
        return self._lb_len * self._lb_vel / self._lb_visc

    @property
    def Womersley_lb(self):
        return math.sqrt(self.freq_lb * self.len_lb**2 / self.visc_lb)

    @property
    def visc_lb(self):
        return self._lb_visc

    @property
    def velocity_lb(self):
        return self._lb_vel

    @property
    def len_lb(self):
        return self._lb_len

    @property
    def freq_lb(self):
        if self._phys_freq is None:
            return 1.0
        return self._phys_freq * self.dt

    @property
    def dx(self):
        if self._lb_len is None:
            return 0
        return self._phys_len / self._lb_len

    @property
    def dt(self):
        if self._lb_visc is None:
            return 0
        return self._lb_visc / self._phys_visc * self.dx**2

    @property
    def info_lb(self):
        return 'Re:%.2f  Wo:%.2f  visc:%.3e  vel:%.3e  len:%.3e  T:%d' % (
                self.Re_lb, self.Womersley_lb, self.visc_lb, self.velocity_lb,
                self.len_lb, 1.0 / self.freq_lb)


class InflowOutflowSubdomain(Subdomain3D):
    # Override this to return a boolean array selecting inflow
    # and outflow nodez, or None if the inflow/outflow is not contained in
    # the current subdomain. wall_map is the part of the global wall map
    # corresponding to the current subdomain (with ghosts).
    def _inflow_outflow(self, hx, hy, hz, wall_map):
        return None, None

    def _inflow_velocity(self):
        if self.config.velocity == 'constant':
            return self.config._converter.velocity_lb
        elif self.config.velocity == 'oscillatory':
            return self.config._converter.velocity_lb * (
                1 + 0.1 * sin(self.config._converter.freq_lb * S.time))
        elif self.config.velocity_profile is not None:
            np.loadtxt(self.config.velocity_profile)

    # Assumes the global wall map is stored in config._wall_map.
    def _wall_map(self, hx, hy, hz):
        return self.select_subdomain(self.config._wall_map, hx, hy, hz)

    # Only used with node_addressing = 'indirect'.
    def load_active_node_map(self, hx, hy, hz):
        self.set_active_node_map_from_wall_map(self._wall_map(hx, hy, hz))

    def _velocity_params(self, hx, hy, hz, wall_map):
        """Finds the center of the inlet and its diameter."""
        inflow, _ = self._inflow_outflow(hx, hy, hz, wall_map)
        z, _, x = np.where(inflow)
        zm = (min(z) + max(z)) / 2.0
        xm = (min(x) + max(x)) / 2.0

        # XXX: compute actual diameter here.
        diam = min(max(z) - min(z), max(x) - min(x))
        return xm, zm, diam

    def _velocity_profile(self, hx, hy, hz, wall_map):
        xm, zm, diam = self._velocity_params(hx, hy, hz, wall_map)
        xm = int(round(xm))
        zm = int(round(zm))
        r = np.sqrt((hz - hz[zm, 0, 0])**2 + (hx - hx[0, xm, 0])**2)

        R = diam / 2.0
        v = vel * 2.0 * (1.0 - r**2 / R**2)
        return v

    def boundary_conditions(self, hx, hy, hz):
        self.config.logger.info(self.config._converter.info_lb)
        wall_map = self._wall_map(hx, hy, hz)
        inlet, outlet = self._inflow_outflow(hx, hy, hz, wall_map)

        self.set_node(wall_map, NTFullBBWall)
        # Vector pointing into the flow domain. The direction of the
        # flow is y+.
        o = D3Q19.vec_to_dir([0, 1, 0])
        if inlet is not None:
            xm, zm, diam = self._velocity_params(hx, hy, hz, wall_map)
            radius_sq = (diam / 2.0)**2
            self.config.logger.info('.. setting inlet, center at (%d, %d), diam=%f',
                                    xm, zm, diam)
            v = self._inflow_velocity()
            self.set_node(inlet, NTRegularizedVelocity(
                DynamicValue(0.0,
                             2.0 * v *
                             (1.0 - ((S.gz - zm)**2 + (S.gx - xm)**2) / radius_sq),
                             0.0),
                orientation=o))

        if outlet is not None:
            self.config.logger.info('.. setting outlet')
            self.set_node(outlet, NTDoNothing(orientation=o))


    def initial_conditions(self, sim, hx, hy, hz):
        sim.rho[:] = 1.0
        wall_map = self._wall_map(hx, hy, hz)
        inlet, outlet = self._inflow_outflow(hx, hy, hz, wall_map)

        if inlet is not None:
            v = self.velocity_profile(hx, hy, hz, wall_map, inlet)
            sim.vy[inlet] = v[inlet]


class HemoSim(LBFluidSim):
    phys_visc = 3.33e-6
    phys_diam = 0.0
    phys_freq = 2.0 * np.pi

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'grid': 'D3Q19',
            'perf_stats_every': 1000,
            'block_size': 128,

            # Data layout. Optimize for size.
            'access_pattern': 'AA',
            'node_addressing': 'indirect',
            'compress_intersubdomain_data': True,

            # Output and checkpointing.
            'checkpoint_every': 500000,
            'final_checkpoint': True,
        })

    @classmethod
    def add_options(cls, group, dim):
        LBFluidSim.add_options(group, dim)
        group.add_argument('--geometry', type=str,
                           default='',
                           help='file defining the geometry')
        group.add_argument('--velocity', type=str,
                           choices=['constant', 'oscillating', 'external'],
                           default='constant')
        group.add_argument('--velocity_profile', type=str,
                           default='', help='external velocity profile')
        group.add_argument('--reynolds', type=float,
                           default=10.0, help='Reynolds number')


    @classmethod
    def modify_config(cls, config):
        converter = UnitConverter(cls.phys_visc, cls.phys_diam, Re=config.reynolds)
        _, _, xs = config._wall_map.shape
        diam = 2.0 * np.sqrt(np.sum(np.logical_not(config._wall_map[:,1,:(xs/2)])) / np.pi)
        converter.set_lb(velocity=0.25, length=diam)
        config.visc = converter.visc_lb
        config._converter = converter
