<%namespace file="kernel_common.mako" import="*" name="kernel_common"/>
<%namespace file="code_common.mako" import="*"/>
<%namespace file="propagation.mako" import="*"/>

## FIXME: This can probably be made global.
<%def name="decl_nvec(name)">
	float ${name}x, float ${name}y
	%if dim == 3:
		, float ${name}z
	%endif
</%def>

// Pseudo-vector.
<%def name="decl_pvec(name)">
	float ${name}x
	%if dim == 3:
		, float ${name}y, float ${name}z
	%endif
</%def>

<%def name="nvec(name)">
	${name}x, ${name}y
	%if dim == 3:
		, ${name}z
	%endif
</%def>

<%def name="pvec(name)">
	${name}x
	%if dim == 3:
		, ${name}y, ${name}z
	%endif
</%def>

<%def name="if3d(name)">
	%if dim == 3:
		${name}
	%endif
</%def>

<%def name="local_part_velocity(prefix='')">
	// Calculate the local effective velocity at the boundary node (linear velocity
	// plus the velocity from the rotational motion).
	%if dim == 2:
		v0[0] = ${prefix}part_vx - ${prefix}part_avx * dry;
		v0[1] = ${prefix}part_vy + ${prefix}part_avx * drx;
	%else:
		v0[0] = ${prefix}part_vx + ${prefix}part_avy * drz - ${prefix}part_avz * dry;
		v0[1] = ${prefix}part_vy + ${prefix}part_avz * drx - ${prefix}part_avx * drz;
		v0[2] = ${prefix}part_vz + ${prefix}part_avx * dry - ${prefix}part_avy * drx;
	%endif
</%def>

<%def name="process_fsi_boundary()">
	%if fsi_enabled:
		if (isBoundaryNode(type)) {
			unsigned int obj_id, dir_mask;
			decodeBoundaryNode(ncode, &obj_id, &dir_mask);

			float forcex = 0.0f;
			float forcey = 0.0f;
			${if3d('float forcez = 0.0f;')}

			float torquex = 0.0f;
			${if3d('float torquey = 0.0f;')}
			${if3d('float torquez = 0.0f;')}

			// Load particle position, velocity and angular velocity into local variables.
			// TODO: Use int2float below?
			// TODO: To optimize this, the texture cache could be used.
			// Calculate the difference between the center of the fluid node and the
			// center of mass of the particle.
			float ${nvec('dr')}, ${nvec('part_v')}, ${pvec('part_av')};
			drx = gx + 0.5f - fsi_pos[obj_id];
			dry = gy + 0.5f - fsi_pos[obj_id + ${fsi_stride}];
			%if dim == 3:
				drz = gz + 0.5f - fsi_pos[obj_id + ${2 * fsi_stride}];
			%endif

			part_vx = fsi_vel[obj_id];
			part_vy = fsi_vel[obj_id + ${fsi_stride}];
			%if dim == 3:
				part_vz = fsi_vel[obj_id + ${2 * fsi_stride}];
			node_%endif

			part_avx = fsi_avel[obj_id];
			%if dim == 3:
				part_avy = fsi_avel[obj_id + ${fsi_stride}];
				part_avz = fsi_avel[obj_id + ${2 * fsi_stride}];
			%endif

			float delta_m;

			%for i, ei in enumerate(grid.basis[1:]):
				if (dir_mask & ${1 << i}) {

					// Calculate the position difference between the actual mid-link boundary node
					// instead of the center of the fluid node.
					drx += ${ei[0]};
					dry += ${ei[1]};
					%if dim == 3:
						drz += ${ei[2]};
					%endif

					${local_part_velocity()}

					// Bounce-back rule and force equation from PhysRevE 66 046708.
					// FIXME: This assumes rho0 = 1
					delta_m = 2.0f * ${cex(grid.weights[i+1] / grid.cssq * ei.dot(grid.v), vectors=True)};
					${get_odist('dist_out', grid.idx_opposite[i+1])} = d0.${grid.idx_name[i+1]} - delta_m;

					delta_m = 2.0f * d0.${grid.idx_name[i+1]} - delta_m;
					forcex += delta_m * ${ei[0]};
					forcey += delta_m * ${ei[1]};
					%if dim == 3:
						forcez += delta_m * ${ei[2]};
					%endif

					%if dim == 2:
						// TODO: Verify this.
						torquex += drx * forcey - dry * forcex;
					%else:
						torquex += dry * forcez - drz * forcey;
						torquey += drz * forcex - drx * forcez;
						torquez += drx * forcey - dry * forcex;
					%endif

					drx -= ${ei[0]};
					dry -= ${ei[1]};
					%if dim == 3:
						drz -= ${ei[2]};
					%endif
				}
			%endfor

			fsi_force[gi] = forcex;
			fsi_torque[gi] = torquex;
			fsi_force[gi + ${dist_size}] = forcey;
			%if dim == 3:
				fsi_force[gi + ${2 * dist_size}] = forcez;
				fsi_torque[gi + ${dist_size}] = torquey;
				fsi_torque[gi + ${2 * dist_size}] = torquez;
			%endif
		}
	%endif
</%def>

<%def name="sum_local_forces()">
	${barrier()}

	// Sum the individual contributions to the force and torque from the whole block.
	// This is a parallel reduction operation.
	for (unsigned int i = get_local_size(0) * get_local_size(1) / 2; i > ${warp_size}; i >>= 1) {
		if (thread_id < i) {
			s_force_x[thread_id] += s_force_x[thread_id + i];
			s_force_y[thread_id] += s_force_y[thread_id + i];
			s_torque_x[thread_id] += s_torque_x[thread_id + i];
			%if dim == 3:
				s_force_z[thread_id] += s_force_z[thread_id + i];
				s_torque_y[thread_id] += s_torque_y[thread_id + i];
				s_torque_z[thread_id] += s_torque_z[thread_id + i];
			%endif
		}
		${barrier()}
	}

	// FIXME: Figure out what to do with this in OpenCL.
	if (thread_id < ${warp_size}) {
		warpReduce(s_force_x, thread_id);
		warpReduce(s_force_y, thread_id);
		warpReduce(s_torque_x, thread_id);
		%if dim == 3:
			warpReduce(s_force_z, thread_id);
			warpReduce(s_torque_y, thread_id);
			warpReduce(s_torque_z, thread_id);
		%endif
	}

	if (thread_id == 0) {
		const int gid = get_group_id(0) + get_group_size(0) * get_group_id(1);
		partial_force[gid] = s_force_x[0];
		partial_torque[gid] = s_torque_x[0];
		partial_force[gid + ${fsi_partial_blocks}] = s_force_y[0];
		%if dim == 3:
			partial_force[gid + ${2*fsi_partial_blocks}] = s_force_z[0];
			partial_torque[gid + ${fsi_partial_blocks}] = s_torque_y[0];
			partial_torque[gid + ${2*fsi_partial_blocks}] = s_torque_z[0];
		%endif
	}
</%def>

%if fsi_enabled:
${device_func} void warpReduce(volatile float *sdata, int idx)
{
	sdata[idx] += sdata[idx + 32];
	sdata[idx] += sdata[idx + 16];
	sdata[idx] += sdata[idx + 8];
	sdata[idx] += sdata[idx + 4];
	sdata[idx] += sdata[idx + 2];
	sdata[idx] += sdata[idx + 1];
}

${kernel} void ProcessParticleForceAndTorque(${global_ptr} unsigned int *map,
	${global_ptr} float *fsi_force, ${global_ptr} float *fsi_torque,
	${global_ptr} float *partial_force,
	${global_ptr} float *partial_torque,
	${decl_nvec('p0')},
	unsigned int obj_id,
	${if3d(', int x_size')})
{
	%if dim == 2:
		int gx = get_global_id(0);
		int gy = get_global_id(1);
	%else:
		int gx = get_global_id(0) % x_size;
		int gy = get_global_id(0) / x_size;
		int gz = get_global_id(1);
	%endif

	// Transform gx and gy into global coordinates (in the whole simulation domain
	// instead of just in the bounding box).
	// TODO: use float2int here?
	gx += (int)p0x;
	gy += (int)p0y;
	${if3d('gz += (int)p0z;')}

	int gi = gx + gy*${arr_nx};
	%if dim == 3:
		gi += ${arr_nx*arr_ny}*gz;
	%endif

	float *s_force_x = (float*)shared;
	float *s_force_y = s_force_x + get_local_size(0) * get_local_size(1);
	float *s_torque_x = s_force_y + get_local_size(0) * get_local_size(1);
	%if dim == 3:
		float *s_force_z = s_torque_x + get_local_size(0) * get_local_size(1);
		float *s_torque_y = s_force_z + get_local_size(0) * get_local_size(1);
		float *s_torque_z = s_torque_y + get_local_size(0) * get_local_size(1);
	%endif

	int thread_id = get_local_id(0) + get_local_size(0) * get_local_id(1);

	s_force_x[thread_id] = 0.0f;
	s_force_y[thread_id] = 0.0f;
	s_torque_x[thread_id] = 0.0f;
	%if dim == 3:
		s_force_z[thread_id] = 0.0f;
		s_torque_y[thread_id] = 0.0f;
		s_torque_z[thread_id] = 0.0f;
	%endif

	int pcode = map[gi];
	int ptype = decodeNodeType(pcode);

	if (ptype == ${geo_boundary}) {
		unsigned int lobj_id, unsigned int dir_mask;
		decodeBoundaryNode(pcode, &lobj_id, &dir_mask);

		// Make sure the force is acting on the object that is being processed
		// in this thread.
		if (obj_id == lobj_id) {
			s_force_x[thread_id] = fsi_force[gi];
			s_force_y[thread_id] = fsi_force[gi + ${dist_size}];
			s_torque_x[thread_id] = fsi_torque[gi];
			%if dim == 3:
				s_force_z[thread_id] = fsi_force[gi + ${2*dist_size}];
				s_torque_y[thread_id] = fsi_torque[gi + ${dist_size}];
				s_torque_z[thread_id] = fsi_torque[gi + ${2*dist_size}];
			%endif
		}
	}

	${sum_local_forces()}
}

${device_func} inline bool SphericalParticle_isinside(${decl_nvec('')}, ${decl_nvec('r0')}, float r)
{
	return (r0x - x) * (r0x - x) + (r0y - y) * (r0y - y) ${if3d('+ (r0z - z) * (r0z - z)')} < r*r;
}

// TODO: The size of the blocks in this kernel should be adjusted so that there
// are no bank conflicts for shared memory accesses.

// NOTE: The reduction operation used in this kernel assumes that the total amount of
// threads in the block is a multiply of 2.
extern ${shared_var} float shared[];
// obj_id: global ID of the particle
// p0: the starting point of the bounding box (location of the midlink node, not the fluid one)
// r0: position of the center of mass (COM) of the particle
// r0_prev: position of the COM during the previous call to this function
// radius: radius of the particle
// x_size: (3D only) width of the bounding box
${kernel} void SphericalParticle_GeoUpdate(${global_ptr} unsigned int *map,
	${kernel_args_1st_moment('iv')}
	${global_ptr} float *dist1_in,
	${global_ptr} float *partial_force,
	${global_ptr} float *partial_torque,
	unsigned int obj_id,
	${decl_nvec('p0')}, ${decl_nvec('r0')}, ${decl_nvec('part_v')}, ${decl_pvec('part_av')},
	${decl_nvec('prev_r0')}, ${decl_nvec('prev_part_v')}, ${decl_pvec('prev_part_av')},
	float radius
	${if3d(', int x_size')})
{
	%if dim == 2:
		int gx = get_global_id(0);
		int gy = get_global_id(1);
	%else:
		int gx = get_global_id(0) % x_size;
		int gy = get_global_id(0) / x_size;
		int gz = get_global_id(1);
	%endif

	// Excess momentum and angular momentum to distribute to the particle.
	float *s_force_x = (float*)shared;
	float *s_force_y = s_force_x + get_local_size(0) * get_local_size(1);
	float *s_torque_x = s_force_y + get_local_size(0) * get_local_size(1);
	%if dim == 3:
		float *s_force_z = s_torque_x + get_local_size(0) * get_local_size(1);
		float *s_torque_y = s_force_z + get_local_size(0) * get_local_size(1);
		float *s_torque_z = s_torque_y + get_local_size(0) * get_local_size(1);
	%endif

	int thread_id = get_local_id(0) + get_local_size(0) * get_local_id(1);

	s_force_x[thread_id] = 0.0f;
	s_force_y[thread_id] = 0.0f;
	s_torque_x[thread_id] = 0.0f;
	%if dim == 3:
		s_force_z[thread_id] = 0.0f;
		s_torque_y[thread_id] = 0.0f;
		s_torque_z[thread_id] = 0.0f;
	%endif

	// Global simulation coordinates of the node processed by this thread.
	float px = p0x + gx;
	float py = p0y + gy;
	${if3d('float pz = p0z + gz;')}

	// Transform gx and gy into global coordinates (in the whole simulation domain
	// instead of just in the bounding box).
	// TODO: use float2int here?
	gx += (int)p0x;
	gy += (int)p0y;
	${if3d('gz += (int)p0z;')}

	int gi = gx + gy*${arr_nx};
	%if dim == 3:
		gi += ${arr_nx*arr_ny}*gz;
	%endif

	int pcode = map[gi];
	int ptype = decodeNodeType(pcode);

	// If the node is inside the particle, simply mark it as unused.
	if (SphericalParticle_isinside(${nvec('p')}, ${nvec('r0')}, radius)) {
		if (ptype != ${geo_unused}) {
			map[gi] = encodeNode(${geo_unused}, obj_id);
			// Node becomes covered by the particle -- update the force and torque
			// acting on the particle to maintain momentum balance.

			// JStatPhys 113 3/4 (2003)
			// F = rho * (u - u_r)
			// T = (x - x_COM) x F

			// FIXME: Assumes rho0 = 1.0.
			float ${nvec('dr')}, v0[${dim}];

			drx = px - r0x;
			dry = py - r0y;
			${if3d('drz = pz - r0z;')}

			${local_part_velocity()}

			s_force_x[thread_id] = ivx[gi] - v0[0];
			s_force_y[thread_id] = ivy[gi] - v0[1];
			${if3d('s_force_z[thread_id] = ivz[gi] - v0[2];')}

			%if dim == 2:
				s_torque_x[thread_id] = drx * s_force_y[thread_id] - dry * s_force_x[thread_id];
			%else:
				s_torque_x[thread_id] = dry * s_force_z[thread_id] - drz * s_force_y[thread_id];
				s_torque_y[thread_id] = drz * s_force_x[thread_id] - drx * s_force_z[thread_id];
				s_torque_z[thread_id] = drx * s_force_y[thread_id] - dry * s_force_x[thread_id];
			%endif
		} else {
			int pobj = decodeNodeMisc(pcode);
			if (pobj != obj_id) {
				map[gi] = encodeNode(${geo_unused}, obj_id);
			}
		}
	// Node is outside of the particle.
	} else {
		int pobj = decodeNodeMisc(pcode);
		unsigned int bmask = 0;
		// Scan all neighboring nodes.  Find ones that are inside the particle
		// and mark their directions as those across which momentum is transferred.
		float tx, ty ${if3d(', tz')};
		%for i, dr in enumerate(grid.basis[1:]):
			tx = px + ${dr[0]};
			ty = py + ${dr[1]};
			%if dim == 3:
				tz = pz + ${dr[2]};
			%endif

			if (SphericalParticle_isinside(${nvec('t')}, ${nvec('r0')}, radius)) {
				bmask |= (1 << ${i});
			}
		%endfor

		map[gi] = encodeBoundaryNode(bmask, obj_id);

		// If this node was inside the particle before but is now uncovered,
		// initialize its distributions with equilibrium values from the previous
		// time step.
		if (ptype == ${geo_unused} && pobj == obj_id) {
			## FIXME: This assumes rho0 = 1, which is not necessarily the case
			## and does not work for binary fluids.
			const float rho = 1.0f;
			float ${nvec('dr')}, v0[${dim}];

			drx = px - prev_r0x;
			dry = py - prev_r0y;
			${if3d('drz = pz - prev_r0z;')}

			${local_part_velocity('prev_')}

			%for local_var in bgk_equilibrium_vars:
				float ${cex(local_var.lhs)} = ${cex(local_var.rhs, vectors=True)};
			%endfor

			%for i, (feq, idx) in enumerate(bgk_equilibrium[0]):
				${get_odist('dist1_in', i)} = ${cex(feq, vectors=True)};
			%endfor

			// Node becomes uncovered.  Modify the torque and force on the particle
			// to maintain momentum balance.
			s_force_x[thread_id] = -rho * (ivx[gi] - v0[0]);
			s_force_y[thread_id] = -rho * (ivy[gi] - v0[1]);
			${if3d('s_force_z[thread_id] = ivz[gi] - v0[2];')}

			%if dim == 2:
				s_torque_x[thread_id] = drx * s_force_y[thread_id] - dry * s_force_x[thread_id];
			%else:
				s_torque_x[thread_id] = dry * s_force_z[thread_id] - drz * s_force_y[thread_id];
				s_torque_y[thread_id] = drz * s_force_x[thread_id] - drx * s_force_z[thread_id];
				s_torque_z[thread_id] = drx * s_force_y[thread_id] - dry * s_force_x[thread_id];
			%endif
		}
	}

	${sum_local_forces()}
}

${kernel} void FSI_SumPartialForceTorques(${global_ptr} float *iforce, ${global_ptr} float *itorque,
		${global_ptr} float *oforce, ${global_ptr} float *otorque, int obj_id,
		${global_ptr} float *particle_force, ${global_ptr} float *particle_torque)
{
	const unsigned int tid = get_local_id(0);
	const unsigned int gid = get_global_id(0);

	float *s_force_x = shared;
	float *s_force_y = s_force_x + get_group_size(0);
	float *s_torque_x = s_force_y + get_group_size(0);
	%if dim == 3:
		float *s_force_z = s_torque_x + get_group_size(0);
		float *s_torque_y = s_force_z + get_group_size(0);
		float *s_torque_z = s_torque_y + get_group_size(0);
	%endif

	s_force_x[tid] = iforce[gid];
	s_force_y[tid] = iforce[gid + ${fsi_partial_blocks}];
	s_torque_x[tid] = itorque[gid];
	%if dim == 3:
		s_force_z[tid] = iforce[gid + ${fsi_partial_blocks*2}];
		s_torque_y[tid] = itorque[gid + ${fsi_partial_blocks}];
		s_torque_z[tid] = itorque[gid + ${fsi_partial_blocks*2}];
	%endif

	${barrier()};

	for (unsigned int i = get_local_size(0) / 2; i > ${warp_size}; i >>= 1) {
		if (tid < i) {
			s_force_x[tid] += s_force_x[tid + i];
			s_force_y[tid] += s_force_y[tid + i];
			s_torque_x[tid] += s_torque_x[tid + i];
			%if dim == 3:
				s_force_z[tid] += s_force_z[tid + i];
				s_torque_y[tid] += s_torque_y[tid + i];
				s_torque_z[tid] += s_torque_z[tid + i];
			%endif
		}
		${barrier()}
	}

	// FIXME: Figure out what to do with this in OpenCL.
	if (tid < ${warp_size}) {
		warpReduce(s_force_x, tid);
		warpReduce(s_force_y, tid);
		warpReduce(s_torque_x, tid);
		%if dim == 3:
			warpReduce(s_force_z, tid);
			warpReduce(s_torque_y, tid);
			warpReduce(s_torque_z, tid);
		%endif
	}

	if (tid == 0) {
		// If we were executing in a single block, the computed value is final and can
		// be written in the particle array.
		if (get_group_size(0) == 1) {
			particle_force[obj_id] = s_force_x[0];
			particle_force[obj_id + ${fsi_stride}] = s_force_y[0];
			particle_torque[obj_id] = s_torque_x[0];
			%if dim == 3:
				particle_force[obj_id + ${fsi_stride*2}] = s_force_z[0];
				particle_torque[obj_id + ${fsi_stride}] = s_torque_y[0];
				particle_torque[obj_id + ${fsi_stride*2}] = s_torque_z[0];
			%endif
		// Otherwise, this is just another partial result which needs to be written to
		// a special array.
		} else {
			const unsigned int grpid = get_group_id(0);
			oforce[grpid] = s_force_x[0];
			otorque[grpid] = s_torque_x[0];
			oforce[grpid + ${fsi_partial_blocks}] = s_force_y[0];
			%if dim == 3:
				oforce[grpid + ${2*fsi_partial_blocks}] = s_force_z[0];
				otorque[grpid + ${fsi_partial_blocks}] = s_torque_y[0];
				otorque[grpid + ${2*fsi_partial_blocks}] = s_torque_z[0];
			%endif
		}
	}
}

${kernel} void FSI_Move(${global_ptr} float *pos, ${global_ptr} float *prev_pos,
	${global_ptr} float *vel,
	${global_ptr} float *ang, ${global_ptr} float *prev_ang,
	${global_ptr} float *avel, ${global_ptr} float *force, ${global_ptr} float *torque)
{
	int i = get_global_id(0);
	float lpos, lang, lvel, lavel;

	%for idim in range(0, dim):
		lpos = pos[i];
		lang = pos[i];
		lvel = vel[i];
		lavel = avel[i];

		prev_pos[i] = lpos;
		prev_ang[i] = lang;

		// FIXME: This assumes the mass and the moment of inertia to be equal to 1.
		lvel += force[i];
		lavel += torque[i];

		pos[i] = lpos + lvel;
		ang[i] = lang + lavel;

		force[i] = 0.0f;
		torque[i] = 0.0f;

		i += ${fsi_stride};
	%endfor
}

%endif  ## fsi_enabled
