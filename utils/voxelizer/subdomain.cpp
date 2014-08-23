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
iPoint3D NodeLocation(const Octree::DNode& node, const int max_depth) {
	iPoint3D location(0, 0, 0);
	int shift = max_depth - 1;

	for (const Octree::index_t it : node.index_trail()) {
		int x = (it & 4) ? 1 : 0;
		int y = (it & 2) ? 1 : 0;
		int z = (it & 1) ? 1 : 0;
		int dh = 1 << shift;
		location = location.jmp(x * dh, y * dh, z * dh);
		shift--;
	}

	return location;
}

// Returns the location of the point opposite to the origin.
iPoint3D NodeExtent(const Octree::DNode& node, const int max_depth) {
	// Start with the extent of the root node.
	int h = 1 << max_depth;
	iPoint3D extent(h, h, h);

	int shift = max_depth - 1;

	for (const Octree::index_t it : node.index_trail()) {
		int x = (it & 4) ? 0 : -1;
		int y = (it & 2) ? 0 : -1;
		int z = (it & 1) ? 0 : -1;
		int dh = 1 << shift;
		extent = extent.jmp(x * dh, y * dh, z * dh);
		shift--;
	}

	return extent.jmp(-1, -1, -1);
}

// node_id -> number of fluid nodes.
static unordered_map<size_t, int> fluid_cache;

void FlushFluidCache() {
	fluid_cache.clear();
}

// Returns the number of children fluid nodes.
int CountFluidNodes(const Octree::DNode& node) {

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

vector<Subdomain> MergeSubdomains(vector<Subdomain> a, vector<Subdomain> b) {
	vector<Subdomain> ret;

	for (int i = 0; i < a.size(); i++) {
		double max_fraction = 0.0;
		int max_j = -1;

		for (int j = 0; j < b.size(); j++) {
			auto h = a[i] + b[j];
			if (h.fill_fraction() > max_fraction) {
				max_j = j;
				max_fraction = h.fill_fraction();
			}
		}

		if (max_fraction >= kMinFillFraction) {
			ret.push_back(a[i] + b[max_j]);
			b.erase(b.begin() + max_j);
		} else {
			ret.push_back(a[i]);
		}
	}

	ret.insert(ret.end(), b.begin(), b.end());
	return ret;
}

vector<Subdomain> ToSubdomains(const Octree::DNode node) {
	if (node.isLeaf()) {
		if (CountFluidNodes(node) == 0) {
			return vector<Subdomain>();
		} else {
			return vector<Subdomain>({Subdomain(node, node.max_depth())});
		}
	}

	auto p1 = MergeSubdomains(ToSubdomains(node[0]), ToSubdomains(node[1]));
	auto p2 = MergeSubdomains(ToSubdomains(node[2]), ToSubdomains(node[3]));
	auto p3 = MergeSubdomains(ToSubdomains(node[4]), ToSubdomains(node[5]));
	auto p4 = MergeSubdomains(ToSubdomains(node[6]), ToSubdomains(node[7]));

	auto p5 = MergeSubdomains(p1, p2);
	auto p6 = MergeSubdomains(p3, p4);

	return MergeSubdomains(p5, p6);
}

