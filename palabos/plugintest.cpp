#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <dlfcn.h>

using namespace std;

int main(int argc, char **argv)
{
	void *handle;
	void (*initializeDomain)(int, void*, int*, void*);
    char *error;

    handle = dlopen("sailfish.so", RTLD_LAZY);
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

	(*initializeDomain)(1, NULL, NULL, NULL);
	dlclose(handle);
	exit(EXIT_SUCCESS);

	return 0;
}
