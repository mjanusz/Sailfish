#include <gtest/gtest.h>
#include "subdomain.hpp"

using namespace std;

TEST(DTreeNode, NodeLocation) {
	Octree octree(0);
	octree.expand('a');
	for (int i = 0; i < 8; i++) {
		(octree.root())[i]() = (char)('a' + i);
	}
/*
	expand(octree, voxels);
	cout.write(&(voxels.begin()[0]), voxels.size());
	cout << endl;
	cout << octree.max_depth() << endl;
	cout << NodeLocation(octree.root()[0]) << " " << NodeExtent(octree.root()[0])  << endl;
*/
}
