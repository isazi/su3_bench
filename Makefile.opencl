#
LAT_CHECK = false

CC = g++
CFLAGS = -std=c++14 -O3 -fopenmp -Wno-ignored-attributes -Wno-deprecated-declarations
LIBS = -lOpenCL

DEFINES = -DUSE_OPENCL
ifeq ($(LAT_CHECK),true)
 DEFINES += -DLAT_CHECK
endif

DEPENDS = su3.h lattice.hpp mat_nn_opencl.hpp

bench_f32_opencl.exe: su3_nn_bench.cpp $(DEPENDS)
	$(CC) -DPRECISION=1 $(CFLAGS) $(DEFINES) -o $@ su3_nn_bench.cpp $(LIBS)

bench_f64_opencl.exe: su3_nn_bench.cpp $(DEPENDS)
	$(CC) $(CFLAGS) $(DEFINES) -o $@ su3_nn_bench.cpp $(LIBS)

all: bench_f64_opencl.exe bench_f32_opencl.exe

clean:
	rm -f *opencl.exe
