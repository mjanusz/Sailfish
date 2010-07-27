#include <Python.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "palabos.h"

#include <latticeBoltzmann/nearestNeighborLattices2D.hh>
#include <latticeBoltzmann/nearestNeighborLattices3D.hh>

using namespace plb;
using namespace std;

extern "C" {

void getBoundaries(void *dataPtr, int boundaryWidth)
{


}

void updateBoundaries(void *dataPtr, int boundaryWidth)
{

}

float getEfficiency(int processorId)
{
	// FIXME: The efficiency should actually be evaluated here.
	return 25.0f;
}

int initializeDomain(int processorId, void *dataPtr, int *extent, void *processorParams)
{
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 2; j++) {
			cout << descriptors::D2Q9Constants<float>::c[i][j] << " ";
		}
		cout << endl;
	}

	Py_Initialize();
	PyRun_SimpleString("from time import time,ctime\n"
		"print 'Today is',ctime(time())\n");
	Py_Finalize();

	return 0;
}

void refreshDomains()
{


}

void releaseDomains()
{


}

void execute(int processorId)
{


}

}
