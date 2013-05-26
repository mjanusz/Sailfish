#include <gtest/gtest.h>
#include "subdomain.hpp"

using namespace std;

Octree MakeTestTree() {
	Octree octree(0);
	octree.expand('a');
	for (int i = 0; i < 8; i++) {
		octree.root()[i]() = (char)('a' + i);
	}

	auto h = octree.root()[0];
	h.expand('A');
	for (int i = 0; i < 8; i++) {
		h[i]() = (char)('A' + i);
	}

	h = octree.root()[5];
	h.expand('0');
	for (int i = 0; i < 8; i++) {
		h[i]() = (char)('0' + i);
	}
	return octree;
}

TEST(DTreeNode, NodeLocation) {
	Octree octree = MakeTestTree();

	EXPECT_EQ(iPoint3D(2, 2, 2), NodeLocation(octree.root()));
	EXPECT_EQ(iPoint3D(1, 1, 1), NodeLocation(octree.root()[0]));
	EXPECT_EQ(iPoint3D(0, 0, 0), NodeLocation(octree.root()[0][0]));
}

TEST(DTreeNode, NodeExtent) {
	Octree octree = MakeTestTree();

	EXPECT_EQ(iPoint3D(2, 2, 2), NodeLocation(octree.root()));
	EXPECT_EQ(iPoint3D(0, 0, 0), NodeExtent(octree.root()[0][0]));
	EXPECT_EQ(iPoint3D(1, 1, 1), NodeExtent(octree.root()[0]));
}

/*
	expand(octree, voxels);
	cout.write(&(voxels.begin()[0]), voxels.size());
	cout << endl;
	cout << octree.max_depth() << endl;
	cout << NodeLocation(octree.root()[0]) << " " << NodeExtent(octree.root()[0])  << endl;
*/

