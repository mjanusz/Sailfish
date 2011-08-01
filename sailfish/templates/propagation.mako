<%!
    from sailfish import sym
%>

<%namespace file="opencl_compat.mako" import="*"/>

<%def name="prop_bnd(dist_out, dist_in, effective_dir, i, di, local, offset)">
## Generate the propagation code for a specific base direction.
##
## This is a generic function which should work for any dimensionality and grid
## type.
##
## Args:
##   effective_dir: X propagation direction (1 for East, -1 for West)
##   offset: target offset in the distribution array
##   i: index of the base vector along which to propagate
##   di: dimension index

	## This is the final dimension, generate the actual propagation code.
	%if di == dim:
		%if dim == 2:
			${set_odist(dist_out, dist_in, i, effective_dir, grid.basis[i][1], 0, offset, local)}
		%else:
			${set_odist(dist_out, dist_in, i, effective_dir, grid.basis[i][1], grid.basis[i][2], offset, local)}
		%endif
	## Make a recursive call to prop_bnd to process the remaining dimensions.
	## The recursive calls are done to generate checks for out-of-domain
	## propagation.
	%else:
		## Make sure we're not propagating outside of the simulation domain.
		%if grid.basis[i][di] > 0:
			if (${loc_names[di]} < ${bnd_limits[di]-1}) { \
		%elif grid.basis[i][di] < 0:
			if (${loc_names[di]} > 0) { \
		%endif
			## Recursive call for the next dimension.
			${prop_bnd(dist_out, dist_in, effective_dir, i, di+1, local, offset)}
		%if grid.basis[i][di] != 0:
			} \
		%endif

		## In case we are about to propagate outside of the simulation domain,
		## check for periodic boundary conditions for the current dimension.
		## If they are enabled, update the offset by a value precomputed in
		## pbc_offsets and proceed to the following dimension.
		%if periodicity[di] and grid.basis[i][di] != 0:
			else {
				${prop_bnd(dist_out, dist_in, effective_dir, i, di+1, local, offset+pbc_offsets[di][int(grid.basis[i][di])])}
			}
		%endif
	%endif
</%def>

## Propagate eastwards or westwards knowing that there is an east/westward
## node layer to propagate to.
<%def name="prop_block_bnd(dist_out, dist_in, dir, dist_source, offset=0)">
## Generate the propagation code for all directions with a X component.  The X component
## is special as shared-memory propogation is done in the X direction.
##
## Args:
##   dir: X propagation direction (1 for East, -1 for West, 0 for orthogonal to X axis)
##
	%for i in sym.get_prop_dists(grid, dir):
		%if dist_source == 'prop_local':
			${prop_bnd(dist_out, dist_in, 0, i, 1, True, offset)}
		%else:
			${prop_bnd(dist_out, dist_in, dir, i, 1, False, offset)}
		%endif
	%endfor
</%def>

<%def name="rel_offset(x, y, z)" filter="trim">
	%if grid.dim == 2:
		${x + y * arr_nx}
	%else:
		${x + arr_nx * (y + arr_ny*z)}
	%endif
</%def>

<%def name="get_odist(dist_out, idir, xoff=0, yoff=0, zoff=0, offset=0)" filter="trim">
	${dist_out}[gi + ${dist_size*idir + offset} + ${rel_offset(xoff, yoff, zoff)}]
</%def>

<%def name="set_odist(dist_out, dist_in, idir, xoff, yoff, zoff, offset, local)">
	%if local:
		${get_odist(dist_out, idir, xoff, yoff, zoff, offset)} = prop_${grid.idx_name[idir]}[lx];
	%else:
		${get_odist(dist_out, idir, xoff, yoff, zoff, offset)} = ${dist_in}.${grid.idx_name[idir]};
	%endif
</%def>

// Propagate distributions using global memory only.
// TODO: This function is DEPRECATED and should be removed.
<%def name="propagate2(dist_out, dist_in='fi')">
	// update the 0-th direction distribution
	${dist_out}[gi] = ${dist_in}.fC;

	// E propagation in global memory
	if (gx < ${lat_nx-1}) {
		${prop_block_bnd(dist_out, dist_in, 1, 'prop_global')}
	}
	%if periodic_x:
	// periodic boundary conditions in the X direction
	else {
		${prop_block_bnd(dist_out, dist_in, 1, 'prop_global', pbc_offsets[0][1])}
	}
	%endif

	// Propagation in directions orthogonal to the X axis (global memory)
	${prop_block_bnd(dist_out, dist_in, 0, 'prop_global')}

	// W propagation in global memory
	if (gx > 0) {
		${prop_block_bnd(dist_out, dist_in, -1, 'prop_global')}
	}
	%if periodic_x:
	// periodic boundary conditions in the X direction
	else {
		${prop_block_bnd(dist_out, dist_in, -1, 'prop_global', pbc_offsets[0][-1])}
	}
	%endif
</%def>

// Propagate distributions using a 1D shared memory array to make the propagation
// in the X direction more efficient.
<%def name="propagate(dist_out, dist_in='fi')">
	<%
		first_prop_dist = grid.idx_name[sym.get_prop_dists(grid, 1)[0]]
	%>

	// Update the 0-th direction distribution
	${dist_out}[gi] = ${dist_in}.fC;

	%if propagation_sentinels:
		// Initialize the shared array with invalid sentinel values.  If the sentinel
		// value is not subsequently overridden, it will not be propagated.
		prop_${first_prop_dist}[lx] = -1.0f;
		${barrier()}
	%endif

	// E propagation in shared memory
	if (gx < ${lat_nx-1}) {
		// Note: propagation to ghost nodes is done directly in global memory as there
		// are no threads running for the ghost nodes.
		if (lx < ${block_size-1} && gx != ${lat_nx-1-envelope_size}) {
			%for i in sym.get_prop_dists(grid, 1):
				prop_${grid.idx_name[i]}[lx+1] = ${dist_in}.${grid.idx_name[i]};
			%endfor
		// E propagation in global memory (at right block boundary)
		} else {
			${prop_block_bnd(dist_out, dist_in, 1, 'prop_global')}
		}
	}
	%if periodic_x:
	// periodic boundary conditions in the X direction
	else {
		${prop_block_bnd(dist_out, dist_in, 1, 'prop_global', pbc_offsets[0][1])}
	}
	%endif

	${barrier()}

	// Save locally propagated distributions into global memory.
	// The leftmost thread is not updated in this block.
	if (lx > 0 && gx < ${lat_nx})
	%if propagation_sentinels:
		if (prop_${first_prop_dist}[lx] != -1.0f)
	%endif
	{
		${prop_block_bnd(dist_out, dist_in, 1, 'prop_local')}
	}

	// Propagation in directions orthogonal to the X axis (global memory)
	${prop_block_bnd(dist_out, dist_in, 0, 'prop_global')}

	%if propagation_sentinels:
		${barrier()}
		// Refill the propagation buffer with sentinel values.
		prop_${first_prop_dist}[lx] = -1.0f;
	%endif

	${barrier()}

	// W propagation in shared memory
	// Note: propagation to ghost nodes is done directly in global memory as there
	// are no threads running for the ghost nodes.
	if (lx > ${envelope_size} || (lx > 0 && gx >= ${block_size})) {
		%for i in sym.get_prop_dists(grid, -1):
			prop_${grid.idx_name[i]}[lx-1] = ${dist_in}.${grid.idx_name[i]};
		%endfor
	// W propagation in global memory (at left block boundary)
	} else if (gx > 0) {
		${prop_block_bnd(dist_out, dist_in, -1, 'prop_global')}
	}
	%if periodic_x:
	// periodic boundary conditions in the X direction
	else {
		${prop_block_bnd(dist_out, dist_in, -1, 'prop_global', pbc_offsets[0][-1])}
	}
	%endif

	${barrier()}

	// The rightmost thread is not updated in this block.
	if (lx < ${block_size-1} && gx < ${lat_nx-1})
	%if propagation_sentinels:
		if (prop_${first_prop_dist}[lx] != -1.0f)
	%endif
	{
		${prop_block_bnd(dist_out, dist_in, -1, 'prop_local')}
	}
</%def>


