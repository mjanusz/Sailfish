all: voxelizer voxelizer_octree

voxelizer:
	$(CXX) -O3 -DNDEBUG -fopenmp -std=gnu++0x $(CXXFLAGS) voxelizer.cpp -I. -Icvmlcpp -o voxelizer

voxelizer_octree:
	$(CXX) -O3 -DNDEBUG -fopenmp -std=gnu++0x $(CXXFLAGS) voxelizer_octtree.cpp -I. -Icvmlcpp -o voxelizer_octtree
