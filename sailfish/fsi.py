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

class FSIObject(object):
    def __init__(self, sim, mass, position, velocity, orientation, ang_velocity):
        self.sim = sim
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.orientation = orientation
        self.ang_velocity = ang_velocity

    def draw_2d(self, surf):
        pass

    def bouding_box(self, position, orientation):
        pass

class SphericalParticle(FSIObject):
    def __init__(self, sim, mass, position, velocity, orientation, ang_velocity, radius):
        super(SphericalParticle, self).__init__(sim, mass, position, velocity, orientation, ang_velocity)
        self.radius = radius

    def bounding_box(self, position, orientation):
        x0 = position[0] - self.radius
        y0 = position[1] - self.radius
        x1 = position[0] + self.radius
        y1 = position[1] + self.radius

        if len(position) > 2:
            z0 = position[2] - self.radius
            z1 = position[2] + self.radius
        else:
            z0 = 0
            z1 = 0

        return Box(x0, y0, z0, x1, y1, z1)

    def geo_update_kernel(self):
        # Kernel used to calculate the total force from partial sums.
        args_format = (4 + self.sim.grid.dim) * 'P' + 'i' + 'f' * (7 *
                self.sim.grid.dim) + 'f'

        if self.sim.grid.dim == 3:
            args_format += 'i'

        self.kern_geo_update = self.sim.backend.get_kernel(
                self.sim.mod, 'SphericalParticle_GeoUpdate', args=None,
                args_format=args_format, block=(1,))

        ## FIXME: Rename this function and actually call it somewhere.

    def geo_update(self, obj_id, pos, ort, vel, avel, prev_pos, prev_ort, prev_vel, prev_avel):
        sim = self.sim

        bbox = self.bounding_box(pos, ort)
        bbox_prev = self.bounding_box(prev_pos, prev_ort)
        bbox = box_union(bbox, bbox_prev)

        ## FIXME: Dist here.

        args = [sim.geo.gpu_map] + sim.gpu_velocity + DIST +
               [sim.gpu_partial_force, sim.gpu_partial_torque,
                obj_id, sim.float(bbox.x0), sim.float(bbox.y0)]

        if sim.grid.dim == 3:
            args.append(sim.float(bbox.z0))

        args.extend([sim.float(x) for x in pox])
        args.extend([sim.float(x) for x in vel])
        args.extend([sim.float(x) for x in avel])
        args.extend([sim.float(x) for x in prev_pos])
        args.extend([sim.float(x) for x in prev_vel])
        args.extend([sim.float(x) for x in prev_avel])
        args.append(sim.float(self.radius))

        if sim.grid.dim == 3:
            args.append(numpy.uint32(bbox.x1 - bbox.x0))

        sim.backend.run_kernel(self.kern_geo_update, grid_size, args=args)

#    def draw_2d(self, surf):
#        import pygame
#        import pygame.draw
#        pygame.draw.circle(surf, (255, 255, 255), self.position, self.radius)
