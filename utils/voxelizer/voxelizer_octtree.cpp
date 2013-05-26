#include <algorithm>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <unordered_map>

#include <cvmlcpp/base/Matrix>
#include <cvmlcpp/math/Euclid>
#include <cvmlcpp/volume/Geometry>
#include <cvmlcpp/volume/VolumeIO>
#include <cvmlcpp/volume/Voxelizer>
#include <cvmlcpp/volume/VoxelTools>

using namespace cvmlcpp;
using namespace std;

// TODO: tranform this program into a Python module so that STL geometry can
// be used to directly initialize a simulation
//
// TODO: consider using the distances() function to provide an orientation
// for the walls

typedef DTree<char, 3u> Octree;
const char kFluid = 0;
const char kWall = 1;


// Returns the location (origin) of a node.
iPoint3D NodeLocation(const Octree::DNode& node) {
	// Start with the location of the root node.
	int h = 1 << (node.max_depth() - 1);
	iPoint3D location(h, h, h);

	int shift = node.max_depth() - 1;

	for (const Octree::index_t it : node.index_trail()) {
		int x = (it & 1) ? 0 : -1;
		int y = (it & 2) ? 0 : -1;
		int z = (it & 4) ? 0 : -1;
		int dh = 1 << shift;
		location = location.jmp(x * dh, y * dh, z * dh);
		shift--;
	}

	return location;
}

iPoint3D NodeExtent(const Octree::DNode& node) {
	// Start with the extent of the root node.
	int h = 1 << (node.max_depth() - 1);
	iPoint3D extent(h, h, h);

	int shift = node.max_depth() - 1;

	for (const Octree::index_t it : node.index_trail()) {
		int x = (it & 1) ? 1 : 0;
		int y = (it & 2) ? 1 : 0;
		int z = (it & 4) ? 1 : 0;
		int dh = 1 << shift;
		extent = extent.jmp(x * dh, y * dh, z * dh);
		shift--;
	}

	return extent.jmp(-1, -1, -1);
}

// Returns the number of children fluid nodes.
int CountFluidNodes(const Octree::DNode& node) {
	// node_id -> number of fluid nodes.
	static unordered_map<size_t, int> fluid_cache;

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

class Subdomain {
  public:
    Subdomain(iPoint3D origin, iPoint3D extent):
		origin_(origin), extent_(extent),
		fluid_nodes_(0)	{};

    Subdomain(iPoint3D origin, iPoint3D extent, int fluid_nodes):
		origin_(origin), extent_(extent),
		fluid_nodes_(fluid_nodes) {};

	Subdomain(Octree::DNode& node):
		origin_(NodeLocation(node)), extent_(NodeExtent(node)),
		fluid_nodes_(CountFluidNodes(node)) {};

	// Builds the union of two subdomains.
	const Subdomain operator+(const Subdomain& rhs) const {
		Subdomain result = *this;
		result.origin_.set(
				min(result.origin_.x(), rhs.origin_.x()),
				min(result.origin_.y(), rhs.origin_.y()),
				min(result.origin_.z(), rhs.origin_.z()));
		result.origin_.set(
				max(result.origin_.x(), rhs.origin_.x()),
				max(result.origin_.y(), rhs.origin_.y()),
				max(result.origin_.z(), rhs.origin_.z()));
		result.fluid_nodes_ += rhs.fluid_nodes_;
		return result;
	}

	// Returns the number of nodes contained within the subdomain.
	int volume() {
		return (extent_.x() - origin_.x() + 1) *
			   (extent_.y() - origin_.y() + 1) *
			   (extent_.z() - origin_.z() + 1);
	}

	double fill_ratio() {
		return static_cast<double>(fluid_nodes_) / volume();
	}

  private:
	iPoint3D origin_, extent_;  // location of the origin point and the point
							    // opposite to the origin
	int fluid_nodes_;
};

// Removes all children nodes that do no contain any fluid.
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

vector<Subdomain> ToSubdomains(Octree::DNode node) {

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

	/*
	 * Check organization of data in the tree.
	 */
	{
		Octree octree(0);
		octree.expand('a');
		for (int i = 0; i < 8; i++) {
			(octree.root())[i]() = (char)('a' + i);
		}

		expand(octree, voxels);
		cout.write(&(voxels.begin()[0]), voxels.size());
		cout << endl;
		cout << octree.max_depth() << endl;
		cout << NodeLocation(octree.root()[0]) << " " << NodeExtent(octree.root()[0])  << endl;
		// aecgbfdh
		return 0;
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
