#
LAT_CHECK = false

CC = g++
CFLAGS = -std=c++14 -O3 -fopenmp -Wno-ignored-attributes -Wno-deprecated-declarations
# Nominally uses Khronos OpenCL headers, but any CL/cl.h will do
CFLAGS += -I$(OCLHPP_INCLUDE) -I$(OCL_INCLUDE)
INCLUDES = -DUSE_OPENCL
ifeq ($(LAT_CHECK),true)
 INCLUDES += -DLAT_CHECK
endif
DEPENDS = su3.h lattice.hpp mat_nn_opencl.hpp
# Assumes libOpenCL.so is in LD_LIBRARY_PATH, if not add it explicitly
LIBS = -lOpenCL

bench_f32_opencl.exe: su3_nn_bench.cpp $(DEPENDS)
	$(CC) -DPRECISION=1 $(CFLAGS) $(INCLUDES) -o $@ su3_nn_bench.cpp $(LIBS)

bench_f64_opencl.exe: su3_nn_bench.cpp $(DEPENDS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ su3_nn_bench.cpp $(LIBS)

all: bench_f64_opencl.exe bench_f32_opencl.exe

clean:
	rm -f *opencl.exe
