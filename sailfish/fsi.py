import math

from collections import namedtuple
from sailfish import sym

Box = namedtuple('Box', 'x0 y0 z0 x1 y1 z1')

def box_union(box1, box2):
    x0 = min(box1.x0, box2.x0)
    y0 = min(box1.y0, box2.y0)
    z0 = min(box1.z0, box2.z0)
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    z1 = max(box1.z1, box2.z1)
    return Box(x0, y0, z0, x1, y1, z1)

def total_vector_components(dim):
    if dim == 3:
        return 6
    else:
        return 3

def partial_block_grid(sim, bbox):
    # FIXME: we should query the backend for this info
    shmem = 16 * 1024
    vector_components = total_vector_components(sim.grid.dim)

    # Calculate the max number of nodes in a block.
    nodes = shmem / vector_components / sim.float().nbytes

    # FIXME: Also use the hardware limit here.
    if nodes > 512:
        nodes = 512

    # Estimated maximum distance the particle can move in two
    # time steps (needed for geo_update).
    move_buffer = 6

    dx = int(math.ceil(bbox.x1 - bbox.x0)) + 2 + move_buffer
    dy = int(math.ceil(bbox.y1 - bbox.y0)) + 2 + move_buffer
    dz = int(math.ceil(bbox.z1 - bbox.z0)) + 2 + move_buffer

    # TODO: Change the strategy here.
    block_w = ((dx + 31) / 32) * 32

    if block_w < nodes:
        grid_w = 1
        block_h = nodes / block_w
        if block_h > dy:
            block_h = dy
            grid_h = 1
        else:
            grid_h = (dy + block_h-1) / block_h
    else:
        block_w = (nodes / 32) * 32
        grid_w = (dx + block_w-1) / block_w
        block_h = 1
        grid_h = dy

    if sim.grid.dim == 3:
        block_w *= block_h
        block_h = 1
        grid_w *= grid_h
        grid_h = dz

    shmem_req = block_w * block_h * vector_components * sim.float().nbytes

    return ((block_w, block_h), (grid_w, grid_h), shmem_req)


class FSIObject(object):
    def __init__(self, sim, mass, position, velocity, orientation, ang_velocity):
        self.sim = sim
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.orientation = orientation
        self.ang_velocity = ang_velocity

        bbox = self.bounding_box(self.position, self.orientation)
        block_size, grid_size, shmem = partial_block_grid(sim, bbox)

        self.block_size = block_size
        self.grid_size = grid_size
        self.shmem = shmem


    def draw_2d(self, surf):
        pass

    def bouding_box(self, position, orientation):
        pass

class SphericalParticle(FSIObject):
    def __init__(self, sim, mass, position, velocity, orientation, ang_velocity, radius):
        self.radius = radius
        super(SphericalParticle, self).__init__(sim, mass, position, velocity, orientation, ang_velocity)

    def bounding_box(self, position, orientation):
        x0 = position[0] - self.radius - 1
        y0 = position[1] - self.radius - 1
        x1 = position[0] + self.radius + 1
        y1 = position[1] + self.radius + 1

        if len(position) > 2:
            z0 = position[2] - self.radius - 1
            z1 = position[2] + self.radius + 1
        else:
            z0 = 0
            z1 = 0

        return Box(x0, y0, z0, x1, y1, z1)

    def init_compute(self):
        # Kernel used to calculate the total force from partial sums.
        args_format = (7 + self.sim.grid.dim) * 'P' + 'i' + 'f' * (1 *
                self.sim.grid.dim) + 'f'

        if self.sim.grid.dim == 3:
            args_format += 'i'

        self.shmem_part2tot = (total_vector_components(self.sim.grid.dim) *
                self.grid_size[0] * self.grid_size[1] * self.sim.float().nbytes)

        self.kern_geo_update = self.sim.backend.get_kernel(
                self.sim.mod, 'SphericalParticle_GeoUpdate', args=None,
                args_format=args_format, block=self.block_size, shared=self.shmem)

    def geo_update(self, obj_id, pos, ort, vel, avel, prev_pos, prev_ort, prev_vel, prev_avel):
        sim = self.sim

        bbox = self.bounding_box(pos, ort)
        bbox_prev = self.bounding_box(prev_pos, prev_ort)
        bbox = box_union(bbox, bbox_prev)

        args = ([sim.geo.gpu_map] + sim.gpu_velocity + sim.curr_dists_out() +
                [sim.gpu_fsi_partial_force, sim.gpu_fsi_partial_torque,
                 sim.gpu_fsi_pos, sim.gpu_fsi_vel, sim.gpu_fsi_avel,
                 obj_id, sim.float(bbox.x0), sim.float(bbox.y0)])

        if sim.grid.dim == 3:
            args.append(sim.float(bbox.z0))

#        args.extend([sim.float(x) for x in prev_pos])
#        args.extend([sim.float(x) for x in prev_vel])
#        args.extend([sim.float(x) for x in prev_avel])
        args.append(sim.float(self.radius))

        if sim.grid.dim == 3:
            args.append(numpy.uint32(bbox.x1 - bbox.x0))

        sim.backend.run_kernel(self.kern_geo_update, self.grid_size, args=args)

        return self.grid_size[0] * self.grid_size[1]

    def draw_2d(self, surf, scale_x, scale_y, pos, ang):
        import pygame
        import pygame.draw

        pos_x = pos[0] * scale_x
        pos_y = pos[1] * scale_y

        sw, sh = surf.get_size()

        pygame.draw.circle(surf, (255, 255, 255), (pos_x, sh - pos_y), self.radius *
                (scale_x + scale_y) / 2)
        pygame.draw.line(surf, (255, 0, 0), (pos_x, sh - pos_y),
                (pos_x + self.radius * scale_x * math.cos(ang[0]),
                 sh - pos_y - self.radius * scale_y * math.sin(ang[0])))
