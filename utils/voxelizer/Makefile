all: voxelizer voxelizer_octree

CXX=clang++
CXXFLAGS=-stdlib=libc++ -I/opt/local/include

voxelizer:
	$(CXX) -O3 -DNDEBUG -fopenmp -std=c++11 $(CXXFLAGS) voxelizer.cpp -I. -Icvmlcpp -o voxelizer

voxelizer_octree:
	$(CXX) -O3 -DNDEBUG -fopenmp -std=c++11 $(CXXFLAGS) voxelizer_octtree.cpp subdomain.cpp -I. -Icvmlcpp -o voxelizer_octtree
