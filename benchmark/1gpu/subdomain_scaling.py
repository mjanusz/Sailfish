#!/usr/bin/env python
"""Performance scaling for multiple subdomains on a single GPU."""

from examples.ldc_3d import LDCSim
from sailfish.controller import LBSimulationController
from sailfish.geo import EqualSubdomainsGeometry3D
from benchmark.scaling import util


def run_benchmark(num_blocks):
    # Lattice size is optimized for Tesla C2050.
    settings = {
        'max_iters': 1000,
        'every': 500,
        'quiet': True,
        'block_size': 128,
        'subdomains': num_blocks,
        'conn_axis': 'z',
        'mode': 'benchmark',
        'lat_nx': 254,
        'lat_ny': 235,
        'lat_nz': 235,
        }

    ctrl = LBSimulationController(LDCSim, EqualSubdomainsGeometry3D, settings)
    timing_infos, min_timings, max_timings, subdomains = ctrl.run()
    util.save_result('1gpu', num_blocks, timing_infos, min_timings,
            max_timings, subdomains)


if __name__ == '__main__':
    args = util.process_cmdline()
    run_benchmark(args.num_blocks)
