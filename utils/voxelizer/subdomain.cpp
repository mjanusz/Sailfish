#include <unordered_map>
#include "subdomain.hpp"

using namespace cvmlcpp;
using namespace std;

const char kFluid = 0;
const char kWall = 1;
const double kMinFillFraction = 0.5;

ostream& operator<<(ostream& os, const Subdomain& s) {
	os << "Subdomain(" << s.origin_ << ", " << s.extent_ << ")";
	return os;
}

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

vector<Subdomain> ToSubdomains(const Octree::DNode node) {
	if (node.isLeaf()) {
		if (CountFluidNodes(node) == 0) {
			return vector<Subdomain>();
		} else {
			return vector<Subdomain>({Subdomain(node)});
		}
	}

	vector<Subdomain> children;

	for (int i = 0; i < Octree::N; i++) {
		auto tmp = ToSubdomains(node[i]);

		for (auto t : tmp) {
			bool merged = false;
			for (auto child : children) {
				auto h = t + child;
				if (h.fill_fraction() < kMinFillFraction) {
					merged = true;
					child = h;
					break;
				}
			}
			if (!merged) {
				children.push_back(t);
			}
		}
	}
	return children;
}

