#include <algorithm>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <hash_map>

#include <cvmlcpp/base/Matrix>
#include <cvmlcpp/volume/Geometry>
#include <cvmlcpp/volume/VolumeIO>
#include <cvmlcpp/volume/Voxelizer>
#include <cvmlcpp/volume/VoxelTools>

using namespace cvmlcpp;
using namespace std;
using namespace __gnu_cxx;

// TODO: tranform this program into a Python module so that STL geometry can
// be used to directly initialize a simulation
//
// TODO: consider using the distances() function to provide an orientation
// for the walls

typedef DTree<char, 3u> Octree;

const char kFluid = 0;
const char kWall = 1;

hash_map<size_t, int> fluid_cache;

int CountFluidNodes(Octree::DNode node) {
	auto it = fluid_cache.find(node.id());
	if (it != fluid_cache.end()) {
		return it->second;
	}

	int ret = 0;
	if (node.isLeaf()) {
		if (node() == kFluid) {
			return 1;
		} else {
			return 0;
		}
	}

	for (int i = 0; i < Octree::N; i++) {
		ret += CountFluidNodes(node[i]);
	}

	fluid_cache[node.id()] = ret;
	return ret;
}

void RemoveEmptyAreas(Octree::DNode node) {
	if (node.isLeaf()) {
		return;
	}

	if (CountFluidNodes(node) == 0) {
		node.collapse(0);
	} else {
		for (int i = 0; i < Octree::N; i++) {
			RemoveEmptyAreas(node[i]);
		}
	}
}

int main(int argc, char **argv)
{
	Matrix<char, 3u> voxels;
	Geometry<float> geometry;

	double voxel_size = 1.0 / 200.0;

	if (argc < 2) {
		cerr << "Usage: ./voxelizer <STL file> [voxel_size]" << endl;
		return -1;
	}

	if (argc >= 3) {
		voxel_size = atof(argv[2]);
	}

	readSTL(geometry, argv[1]);

	geometry.scaleTo(1.0);
	std::cout << "Bounding box: "
	       << geometry.max(0) - geometry.min(0) << " "
	       << geometry.max(1) - geometry.min(1) << " "
	       << geometry.max(2) - geometry.min(2) << std::endl;

	Octree octree(0);
	voxelize(geometry, octree, voxel_size, kFluid, kWall);

	RemoveEmptyAreas(octree.root());
	/*
	int fluid = count(voxels.begin(), voxels.end(), 0);
	std::cout << "Nodes total: " << voxels.size() << " active: "
		<< round(fluid / (double)voxels.size() * 10000) / 100.0 << "%" << std::endl;

	const std::size_t *ext = voxels.extents();
	std::cout << "Lattice size: " << ext[0] << " " << ext[1]
		<< " " << ext[2] << std::endl;

	std::ofstream out("output.npy");
	out << "\x93NUMPY\x01";

	char buf[128] = {0};

	out.write(buf, 1);

	snprintf(buf, 128, "{'descr': 'bool', 'fortran_order': False, 'shape': (%lu, %lu, %lu)}",
			ext[0], ext[1], ext[2]);

	int i, len = strlen(buf);
	unsigned short int dlen = (((len + 10) / 16) + 1) * 16;

	for (i = 0; i < dlen - 10 - len; i++) {
		buf[len+i] = ' ';
	}
	buf[len+i] = 0x0;
	dlen -= 10;

	out.write((char*)&dlen, 2);
	out << buf;

	out.write(&(voxels.begin()[0]), voxels.size());
	out.close();

	// Export a VTK file with the voxelized geometry.
	outputVTK(voxels, "output.vtk");
*/
	return 0;
}
