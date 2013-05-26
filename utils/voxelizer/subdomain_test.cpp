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
	FlushFluidCache();
	Octree octree = MakeTestTree();

	EXPECT_EQ(iPoint3D(0, 0, 0), NodeLocation(octree.root()));
	EXPECT_EQ(iPoint3D(0, 0, 0), NodeLocation(octree.root()[0]));
	EXPECT_EQ(iPoint3D(0, 0, 0), NodeLocation(octree.root()[0][0]));

	EXPECT_EQ(iPoint3D(1, 0, 0), NodeLocation(octree.root()[0][4]));
	EXPECT_EQ(iPoint3D(1, 0, 1), NodeLocation(octree.root()[0][5]));
	EXPECT_EQ(iPoint3D(1, 1, 0), NodeLocation(octree.root()[0][6]));
	EXPECT_EQ(iPoint3D(1, 1, 1), NodeLocation(octree.root()[0][7]));
}

TEST(DTreeNode, NodeExtent) {
	FlushFluidCache();
	Octree octree = MakeTestTree();

	EXPECT_EQ(iPoint3D(3, 3, 3), NodeExtent(octree.root()));
	EXPECT_EQ(iPoint3D(1, 1, 1), NodeExtent(octree.root()[0]));
	EXPECT_EQ(iPoint3D(0, 0, 0), NodeExtent(octree.root()[0][0]));
}

TEST(SubdomainConversion, SingleSubdomain) {
	Octree octree(0);
	octree.expand(kWall);
	octree.root()[0]() = kFluid;

	{
		auto res = ToSubdomains(octree.root());
		EXPECT_EQ(1, res.size());
		EXPECT_EQ(Subdomain(iPoint3D(0, 0, 0), iPoint3D(0, 0, 0), 1), res[0]);
	}

	FlushFluidCache();
	octree.root()[1]() = kFluid;
	{
		auto res = ToSubdomains(octree.root());
		EXPECT_EQ(1, res.size());
		EXPECT_EQ(Subdomain(iPoint3D(0, 0, 0), iPoint3D(0, 0, 1), 2), res[0]);
	}

	FlushFluidCache();
	octree.root()[3]() = kFluid;
	{
		auto res = ToSubdomains(octree.root());
		EXPECT_EQ(1, res.size());
		EXPECT_EQ(Subdomain(iPoint3D(0, 0, 0), iPoint3D(0, 1, 1), 3), res[0]);
	}

	FlushFluidCache();
	octree = Octree(0);
	octree.expand(kWall);
	octree.root()[0].expand(kWall);
	octree.root()[7].expand(kWall);
	octree.root()[0][7]() = kFluid;
	octree.root()[7][0]() = kFluid;

	{
		auto res = ToSubdomains(octree.root());
		EXPECT_EQ(1, res.size());
		EXPECT_EQ(Subdomain(iPoint3D(1, 1, 1), iPoint3D(2, 2, 2), 2), res[0]);
	}
}

/*
	expand(octree, voxels);
	cout.write(&(voxels.begin()[0]), voxels.size());
	cout << endl;
	cout << octree.max_depth() << endl;
	cout << NodeLocation(octree.root()[0]) << " " << NodeExtent(octree.root()[0])  << endl;
*/

