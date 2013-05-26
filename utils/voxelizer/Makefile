all: voxelizer voxelizer_octree

CXX=clang++
CXXFLAGS=-stdlib=libc++ -I/opt/local/include
#-fopenmp

.PHONY: subdomain_test voxelizer voxelizer_octree

voxelizer: voxelizer.cpp
	$(CXX) -O3 -DNDEBUG -std=c++11 $(CXXFLAGS) voxelizer.cpp -I. -Icvmlcpp -o $@

voxelizer_octree: voxelizer_octtree.cpp
	$(CXX) -O3 -DNDEBUG -std=c++11 $(CXXFLAGS) voxelizer_octtree.cpp subdomain.cpp -I. -Icvmlcpp -o $@

subdomain_test: subdomain_test.cpp
	$(CXX) -std=c++11 $(CXXFLAGS) subdomain_test.cpp subdomain.cpp -I. -Icvmlcpp -lgtest -lgtest_main -o $@
