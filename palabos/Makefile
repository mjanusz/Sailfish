INCLUDES = -I/usr/include/python2.6 -I$(PALABOS_ROOT)/src
LDLIBS = -lpython2.6

all: sailfish.so plugintest

sailfish.so: sailfish.cpp
	$(CXX) $(CXXFLAGS) -fPIC $(INCLUDES) $(LDLIBS) -shared -o $@ $<

plugintest: plugintest.cpp
	$(CXX) $(CXXFLAGS) -rdynamic -Wl,-rpath=. -ldl -o $@ $<

clean:
	rm -f sailfish.so plugintest
