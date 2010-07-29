INCLUDES = -I/usr/include/python2.6 -I$(PALABOS_ROOT)/src
LDLIBS = -lpython2.6

all: plb_sailfish.so plugintest

plb_sailfish.so: sailfish.cpp
	$(CXX) $(CXXFLAGS) -fPIC $(INCLUDES) $(LDLIBS) -L/usr/lib64 -shared -o $@ $<

plugintest: plugintest.cpp
	$(CXX) $(CXXFLAGS) -rdynamic -Wl,-rpath=. -ldl -o $@ $<

clean:
	rm -f plb_sailfish.so plugintest
