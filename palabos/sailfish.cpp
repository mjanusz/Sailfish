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
	// If i-th element of this array is j, then i-th Sailfish basis vector corresponds
	// to Palabos vector number j.
	int velocity_permutation[9] = {-1};
	char *argv[1] = {"dummy"};

	Py_Initialize();
	PySys_SetArgv(1, argv);
	PyObject *pTmp, *pModule, *pName;

	// Calculate the velocity permutation array.
	pModule = PyImport_ImportModule("sailfish.sym");

	if (pModule == NULL) {
		PyErr_Print();
		cerr << "Failed to load \"sailfish.sym\"" << endl;
		return 1;

	}

	PyObject *pGrid, *pBasis, *pVec, *pComp, *pNum;
	pGrid = PyObject_GetAttrString(pModule, "D2Q9");
	if (pGrid == NULL) {
		PyErr_Print();
		cerr << "D2Q9 class missing" << endl;
		return 1;
	}

	pBasis = PyObject_GetAttrString(pGrid, "basis");
	if (pBasis == NULL) {
		PyErr_Print();
		cerr << "D2Q9 class does not have the 'basis' attribute" << endl;
		return 1;
	}

	for (int i = 0; i < 9; i++) {
		pVec = PySequence_GetItem(pBasis, i);
		long comps[2];

		for (int j = 0; j < 2; j++) {
			pComp = PySequence_GetItem(pVec, j);
			comps[j] = PyInt_AsLong(pComp);
			if (comps[j] == -1 && PyErr_Occurred()) {
				PyErr_Print();
				cerr << "Error when processing basis vector " << i << " component " << j << endl;
				return 1;
			}
			Py_DECREF(pComp);
		}

		Py_DECREF(pVec);

		for (int j = 0; j < 9; j++) {
			if (descriptors::D2Q9Constants<float>::c[j][0] == comps[0] &&
				descriptors::D2Q9Constants<float>::c[j][1] == comps[1]) {
				velocity_permutation[i] = j;
				break;
			}
		}

		if (velocity_permutation[i] == -1) {
			cerr << "Failed to find a mapping between Palabos and Sailfish basis vector " << i << endl;
			return 1;
		}
	}

	Py_DECREF(pBasis);
	Py_DECREF(pGrid);

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
