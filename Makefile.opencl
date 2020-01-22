#
LAT_CHECK = false

CC = g++
CFLAGS = -std=c++14 -O3 -fopenmp -Wno-ignored-attributes -Wno-deprecated-declarations

# Nominally uses Khronos OpenCL headers, but any CL/cl.h will do
OCL_INCLUDE = /global/cfs/cdirs/mpccc/dwdoerf/cori-gpu/OpenCL-Headers
OCLHPP_INCLUDE = /global/cfs/cdirs/mpccc/dwdoerf/cori-gpu/OpenCL-CLHPP/include
INCLUDES = -I$(OCLHPP_INCLUDE) -I$(OCL_INCLUDE)

# Nominally uses Khronos OpenCL ICD loader, by any libOpenCL.so will do
OCLICD_LIB = /global/cfs/cdirs/mpccc/dwdoerf/cori-gpu/OpenCL-ICD-Loader/build
LIBS = -L$(OCLICD_LIB) -lOpenCL

DEFINES = -DUSE_OPENCL
ifeq ($(LAT_CHECK),true)
 DEFINES += -DLAT_CHECK
endif

DEPENDS = su3.h lattice.hpp mat_nn_opencl.hpp

bench_f32_opencl.exe: su3_nn_bench.cpp $(DEPENDS)
	$(CC) -DPRECISION=1 $(CFLAGS) $(INCLUDES) $(DEFINES) -o $@ su3_nn_bench.cpp $(LIBS)

bench_f64_opencl.exe: su3_nn_bench.cpp $(DEPENDS)
	$(CC) $(CFLAGS) $(INCLUDES) $(DEFINES) -o $@ su3_nn_bench.cpp $(LIBS)

all: bench_f64_opencl.exe bench_f32_opencl.exe

clean:
	rm -f *opencl.exe
