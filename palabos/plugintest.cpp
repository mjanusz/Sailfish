#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <dlfcn.h>

#include "palabos.h"

using namespace std;

int main(int argc, char **argv)
{
	void *handle;
	void (*initializeDomain)(int, void*, int*, void*);
	void (*cleanUp)(void);
	char *error;

	// The flags here are important. If RTLD_GLOBAL is missing, some Python
	// modules will not work.
	handle = dlopen("plb_sailfish.so", RTLD_LAZY | RTLD_GLOBAL);
	if (!handle) {
		cerr << dlerror() << endl;
		exit(EXIT_FAILURE);
	}

	dlerror();

	*(void **) (&initializeDomain) = dlsym(handle, "initializeDomain");
	if ((error = dlerror()) != NULL)  {
		cerr << error << endl;
		exit(EXIT_FAILURE);
	}

	*(void **) (&cleanUp) = dlsym(handle, "cleanUp");
	if ((error = dlerror()) != NULL)  {
		cerr << error << endl;
		exit(EXIT_FAILURE);
	}

	int domain_size[2] = {64, 64};

	struct D2Q9_DistF *grid_data = new struct D2Q9_DistF[64 * 64];

	for (int y = 0; y < 64; y++) {
		for (int x = 0; x < 64; x++) {
			int idx = x + y * 64;
			grid_data[idx].mask = false;
			for (int i = 0; i < 9; i++) {
				grid_data[idx].dist[i] = 0.0f;
			}
		}
	}

	(*initializeDomain)(1, grid_data, domain_size, NULL);

	(*cleanUp)();

	dlclose(handle);
	exit(EXIT_SUCCESS);

	return 0;
}
