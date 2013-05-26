all: voxelizer voxelizer_octree

CXX=clang++
CXXFLAGS=-stdlib=libc++ -I/opt/local/include

voxelizer:
	$(CXX) -O3 -DNDEBUG -fopenmp -std=c++11 $(CXXFLAGS) voxelizer.cpp -I. -Icvmlcpp -o $@

voxelizer_octree:
	$(CXX) -O3 -DNDEBUG -fopenmp -std=c++11 $(CXXFLAGS) voxelizer_octtree.cpp subdomain.cpp -I. -Icvmlcpp -o $@

subdomain_test:
	$(CXX) -std=c++11 $(CXXFLAGS) subdomain_test.cpp -I. -Icvmlcpp -lgtest -lgtest_main -o $@
