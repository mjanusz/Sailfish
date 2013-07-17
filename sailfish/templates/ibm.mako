## Code supporting the immersed bondary method.

// Interpolates fluid velocity and updates particle positions.
${kernel} void UpdateParticlePosition(
		${kernel_args_1st_moment('fluid_v')}
		${kernel_args_1st_moment('r')}
		const int num_partciles) {
	// Particle ID.
	int pid = get_global_id(0);
	if (pid >= num_particles) {
		return;
	}

	float lrx = rx[pid];
	float lry = ry[pid];
	${'float lrz = rz[pid];' if dim == 3 else ''}

	int xmin = lrx - 0.5f;
	int ymin = lry - 0.5f;
	int zmin = lrz - 0.5f;

	// Particle velocity.
	float lvx = 0.0f, lvy = 0.0f, lvz = 0.0f;

	for (int z = zmin; z < zmin + 1; z++) {
		for (int y = ymin; y < ymin + 1; y++) {
			for (int x = xmin; x < xmin + 1; x++) {
				float dx = fabsf(lrx - x - 0.5f);
				float dy = fabsf(lry - y - 0.5f);
				float dz = fabsf(lrz - z - 0.5f);
				float w = (1.0f - dx) * (1.0f - dy) * (1.0f - dz);
				int idx = getGlobalIdx(x, y, z);
				lvx += fluid_vx[idx] * w;
				lvy += fluid_vy[idx] * w;
				lvz += fluiid_vz[idx] * w;
			}
		}
	}

	rx[pid] = lrx + lvx;
	ry[pid] = lry + lvy;
	${'rz[pid] = lrz + lvz;' if dim == 3 else ''}
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

	const float radius = 10.0f;
	float volume = 4.0f/3.0f * M_PI * radius * radius * radius;

	float lrx = rx[pid];
	float lry = ry[pid];
	${'float lrz = rz[pid];' if dim == 3 else ''}

	float lref_x = refx[pid];
	float lref_y = refy[pid];
	float lref_z = refz[pid];

	float lstiffness = stiffness[pid];

	int xmin = lrx - 0.5f;
	int ymin = lry - 0.5f;
	int zmin = lrz - 0.5f;

	// Particle velocity.
	float lvx = 0.0f, lvy = 0.0f, lvz = 0.0f;

	for (int z = zmin; z < zmin + 1; z++) {
		for (int y = ymin; y < ymin + 1; y++) {
			for (int x = xmin; x < xmin + 1; x++) {
				float dx = fabsf(lrx - x - 0.5f);
				float dy = fabsf(lry - y - 0.5f);
				float dz = fabsf(lrz - z - 0.5f);
				float w = (1.0f - dx) * (1.0f - dy) * (1.0f - dz);
				int idx = getGlobalIdx(x, y, z);
				// Assumes particles forces do not overlap.
				force[idx] = -lstiffness * (rx[idx] - lref_x) * volume;
				force[idx] = -lstiffness * (ry[idx] - lref_y) * volume;
				force[idx] = -lstiffness * (rz[idx] - lref_z) * volume;
			}
		}
	}
}
