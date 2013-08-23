<%!
    from sailfish import sym
%>

<%namespace file="propagation.mako" import="propagate_inplace"/>
<%namespace file="code_common.mako" import="dump_dists"/>
<%namespace file="kernel_common.mako" import="iteration_number_arg_if_required,iteration_number_if_required,get_dist"/>

<%
	def rel_offset(x, y, z=0):
		if grid.dim == 2:
			return x + y * arr_nx
		else:
			return x + arr_nx * (y + arr_ny * z)
%>

// Code to read distribution data directly from the memory of the recirculation
// buffer subdomain. This works when unified addressing is supported and
// both subdomains are simulated on a single device.

<%def name="get_dist_other(array, i, idx, offset=0)" filter="trim">
	${array}[${idx} + dist_size * ${i} + ${offset}]
</%def>

${kernel} void CopyDataFromRecirculationBuffer(
	${global_ptr} ${const_ptr} float *dist_in,
	${global_ptr} float *dist_out, int z, int dist_size
	${iteration_number_if_required()})
{
	int gx = get_global_id(0) + 1;
	int gy = get_global_id(1) + 1;

	if (gx >= ${lat_nx-1} || gy >= ${lat_ny-1}) {
		return;
	}

	// This works because the X and Y dimensions are the same for the
	// recirculation buffer (and so is any padding).
	int gi = getGlobalIdx(gx, gy, z);
	Dist fi;
	// This has to be the opposite to what happens in the propagation step,
	// as this kernel is called with an updated time step (one step ahead of
	// the iteration number in the propagation step).
	%if access_pattern == 'AA':
		if ((iteration_number & 1) == 0) {
			%for i, dname in enumerate(grid.idx_name):
				fi.${dname} = ${get_dist_other('dist_in', i, 'gi')};
				if (!isfinite(fi.${dname})) {
					printf("invalid distribution: %f @ %d, %d\n", fi.${dname}, gx, gy);
					die();
				}
			%endfor
		} else {
			%for i, (dname, ei) in enumerate(zip(grid.idx_name, grid.basis)):
				fi.${dname} = ${get_dist_other('dist_in', grid.idx_opposite[i], 'gi', offset=rel_offset(*(-ei)))};
				if (!isfinite(fi.${dname})) {
					printf("invalid distribution: %f @ %d, %d\n", fi.${dname}, gx, gy);
					die();
				}
			%endfor
		}
	%endif

	gi = getGlobalIdx(gx, gy, 1);

	%if access_pattern == 'AA':
		if ((iteration_number & 1) == 0) {
			${propagate_inplace('dist_out', 'fi')}
		} else {
			%for i, (dname, ei) in enumerate(zip(grid.idx_name, grid.basis)):
				${get_dist('dist_out', grid.idx_opposite[i], 'gi', offset=rel_offset(*(-ei)))} = fi.${dname};
			%endfor
		}
	%endif
}

// Code to collect data and distribute data between the recirculation buffer
// and the actual simulation domain. This has to be used if direct access to the
// memory of the recirculation buffer is not available to the main simulation.

<%def name="buf_index(dist_idx)" filter="trim">
	${(lat_nx - envelope_size) * (lat_ny - envelope_size) * dist_idx} + (gy - ${envelope_size}) * ${(lat_nx - envelope_size)} + (gx - ${envelope_size})
</%def>

${kernel} void CollectDataFromRecirculationBuffer(
	${global_ptr} float *dist_in,
	${global_ptr} float *dist_out, int z
	${iteration_number_if_required()})
{
	int gx = get_global_id(0) + ${envelope_size};
	int gy = get_global_id(1) + ${envelope_size};

	if (gx >= ${lat_nx - envelope_size} || gy >= ${lat_ny - envelope_size}) {
		return;
	}

	// This works because the X and Y dimensions are the same for the
	// recirculation buffer (and so is any padding).
	int gi = getGlobalIdx(gx, gy, z);
	Dist fi;
	%if access_pattern == 'AA':
		// This has to be the opposite to what happens in the propagation step,
		// as this kernel is called with an updated time step (one step ahead of
		// the iteration number in the propagation step).
		if ((iteration_number & 1) == 1) {
			%for i, dname in enumerate(grid.idx_name):
				fi.${dname} = ${get_dist('dist_in', i, 'gi')};
				if (!isfinite(fi.${dname})) {
					printf("invalid distribution: %f @ %d, %d\n", fi.${dname}, gx, gy);
					die();
				}
			%endfor
		} else {
			%for i, (dname, ei) in enumerate(zip(grid.idx_name, grid.basis)):
				fi.${dname} = ${get_dist('dist_in', grid.idx_opposite[i], 'gi', offset=rel_offset(*(-ei)))};
				if (!isfinite(fi.${dname})) {
					printf("invalid distribution: %f @ %d, %d\n", fi.${dname}, gx, gy);
					die();
				}
			%endfor
		}
	%else:
		%for i, dname in enumerate(grid.idx_name):
			fi.${dname} = ${get_dist('dist_in', i, 'gi')};
			if (!isfinite(fi.${dname})) {
				printf("invalid distribution: %f @ %d, %d\n", fi.${dname}, gx, gy);
				die();
			}
		%endfor
	%endif

	%for i, dname in enumerate(grid.idx_name):
		dist_out[${buf_index(i)}] = fi.${dname};
	%endfor
}

${kernel} void DistributeDataFromRecirculationBuffer(
	${global_ptr} float *dist_in,
	${global_ptr} float *dist_out
	${iteration_number_if_required()})
{
	int gx = get_global_id(0) + ${envelope_size};
	int gy = get_global_id(1) + ${envelope_size};

	if (gx >= ${lat_nx - envelope_size} || gy >= ${lat_ny - envelope_size}) {
		return;
	}

	int gi = getGlobalIdx(gx, gy, ${envelope_size});
	Dist fi;
	%for i, dname in enumerate(grid.idx_name):
		fi.${dname} = dist_in[${buf_index(i)}];
		if (!isfinite(fi.${dname})) {
			printf("invalid value in distribute: %f @ %d, %d\n", fi.${dname}, gx, gy);
			die();
		}
	%endfor

	%if access_pattern == 'AA':
		if ((iteration_number & 1) == 0) {
			%for i, dname in enumerate(grid.idx_name):
				%if grid.basis[i][2] >= 0:
					${get_dist('dist_out', i, 'gi')} = fi.${dname};
				%endif
			%endfor
		} else {
			%for i, (dname, ei) in enumerate(zip(grid.idx_name, grid.basis)):
				%if grid.basis[i][2] >= 0:
					${get_dist('dist_out', grid.idx_opposite[i], 'gi', offset=rel_offset(*(-ei)))} = fi.${dname};
				%endif
			%endfor
		}
	%else:
		%for i, dname in enumerate(grid.idx_name):
	##		%if grid.basis[i][2] == 1:
				${get_dist('dist_out', i, 'gi')} = fi.${dname};
	##		%endif
		%endfor
	%endif
}

${kernel} void HandleNTCopyNodes(
	${global_ptr} int *map,
	${global_ptr} float* dist_in
	${iteration_number_if_required()}
) {
	int gx = get_global_id(0) + ${envelope_size};
	int gy = get_global_id(1) + ${envelope_size};

	if (gx >= ${lat_nx - envelope_size} || gy >= ${lat_ny - envelope_size}) {
		return;
	}

	int gi_dst = getGlobalIdx(gx, gy, 280);
	int gi_src = getGlobalIdx(gx, gy, 279);
	float t;
	// Called with an updated iteration number.
	if ((iteration_number & 1) == 0) {
		%for dist_idx in sym.get_missing_dists(grid, 6):
			t = ${get_dist('dist_in', dist_idx, 'gi_src')};
			${get_dist('dist_in', dist_idx, 'gi_dst')} = t;
		%endfor
	} else {
		%for dist_idx in sym.get_missing_dists(grid, 6):
			t = ${get_dist('dist_in', grid.idx_opposite[dist_idx], 'gi_src', offset=rel_offset(*(-grid.basis[dist_idx])))};
			${get_dist('dist_in', grid.idx_opposite[dist_idx], 'gi_dst', offset=rel_offset(*(-grid.basis[dist_idx])))} = t;
		%endfor
	}
}
