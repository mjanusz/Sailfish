## Code supporting the immersed bondary method.
<%namespace file="utils.mako" import="*"/>
<%namespace file="kernel_common.mako" import="kernel_args_1st_moment" />

// Interpolates fluid velocity and updates particle positions.
${kernel} void UpdateParticlePosition(
		${kernel_args_1st_moment('fluid_v')}
		${kernel_args_1st_moment('r')}
		const int num_particles) {
	// Particle ID.
	int pid = get_global_id(0);
	if (pid >= num_particles) {
		return;
	}

	float lrx = rx[pid];
	float lry = ry[pid];
	${ifdim3('float lrz = rz[pid];')}

	int xmin = lrx;
	int ymin = lry;
	${ifdim3('int zmin = lrz;')}

	// Particle velocity.
	float lvx = 0.0f, lvy = 0.0f${ifdim3(', lvz = 0.0f')};

	// \phi_2 kernel: 1 - |x| for |x| <= 1.
	${ifdim3('for (int z = zmin; z < zmin + 1; z++)')}
	{
		for (int y = ymin; y <= ymin + 1; y++) {
			for (int x = xmin; x <= xmin + 1; x++) {
				float dx = fabsf(lrx - x);
				float dy = fabsf(lry - y);
				${ifdim3('float dz = fabsf(lrz - z);')}
				float w = (1.0f - dx) * (1.0f - dy) ${ifdim3('* (1.0f - dz)')};
				int idx = getGlobalIdx(x, y ${ifdim3(', z')});
				lvx += fluid_vx[idx] * w;
				lvy += fluid_vy[idx] * w;
				${ifdim3('lvz += fluid_vz[idx] * w;')}
			}
		}
	}

	rx[pid] = lrx + lvx;
	ry[pid] = lry + lvy;
	${ifdim3('rz[pid] = lrz + lvz;')}
}

// Generate particle forces.
${kernel} void SpreadParticleForcesStiff(
		${global_ptr} float* stiffness,
		${kernel_args_1st_moment('r')}
		${kernel_args_1st_moment('ref')}
		${kernel_args_1st_moment('force')}
		const int num_particles)
{
	// Particle ID.
	int pid = get_global_id(0);
	if (pid >= num_particles) {
		return;
	}

	float lrx = rx[pid];
	float lry = ry[pid];
	${'float lrz = rz[pid];' if dim == 3 else ''}

	float lref_x = refx[pid];
	float lref_y = refy[pid];
	${'float lref_z = refz[pid];' if dim == 3 else ''}

	float lstiffness = stiffness[pid];

	int xmin = lrx;
	int ymin = lry;
	${'int zmin = lrz;' if dim == 3 else ''}

	${'for (int z = zmin; z < zmin + 1; z++)' if dim == 3 else ''}
	{
		for (int y = ymin; y <= ymin + 1; y++) {
			for (int x = xmin; x <= xmin + 1; x++) {
				float dx = fabsf(lrx - x);
				float dy = fabsf(lry - y);
				${'float dz = fabsf(lrz - z)' if dim == 3 else ''};
				float w = (1.0f - dx) * (1.0f - dy) ${'* (1.0f - dz)' if dim == 3 else ''};
				int idx = getGlobalIdx(x, y${ifdim3(', z')});
				// Assumes particles forces do not overlap.
				forcex[idx] += -lstiffness * (lrx - lref_x) * w;
				forcey[idx] += -lstiffness * (lry - lref_y) * w;
				${'forcez[idx] += -lstiffness * (lrz - lref_z) * w' if dim == 3 else ''};
			}
		}
	}
}
