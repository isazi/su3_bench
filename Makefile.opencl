#
LAT_CHECK = false
OCL_INC = $(CUDA_ROOT)/include
OCL_LIB = $(CUDA_ROOT)/lib64

CC = g++
CFLAGS = -std=c++14 -O3 -fopenmp -Wno-ignored-attributes -Wno-deprecated-declarations
CFLAGS += -I$(OCL_INC)
INCLUDES = -DUSE_OPENCL
ifeq ($(LAT_CHECK),true)
 INCLUDES += -DLAT_CHECK
endif
DEPENDS = su3.h lattice.hpp mat_nn_opencl.hpp
LIBS = -L$(OCL_LIB) -lOpenCL

bench_f32_opencl.exe: su3_nn_bench.cpp $(DEPENDS)
	$(CC) -DPRECISION=1 $(CFLAGS) $(INCLUDES) -o $@ su3_nn_bench.cpp $(LIBS)

bench_f64_opencl.exe: su3_nn_bench.cpp $(DEPENDS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ su3_nn_bench.cpp $(LIBS)

all: bench_f64_opencl.exe bench_f32_opencl.exe

clean:
	rm -f *opencl.exe
