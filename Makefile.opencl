#
CC = g++
CFLAGS = -std=c++11 -g -O3 -fopenmp -Wno-ignored-attributes -Wno-deprecated-declarations
CFLAGS += -I$(CUDA_ROOT)/include

INCLUDES = -DUSE_OPENCL -DITERATIONS=100 -DLDIM=32 #-DLAT_CHECK
LDLIBS = -lm 
LDLIBS += -L$(CUDA_ROOT)/lib64 -lOpenCL

DEPENDS = su3.h lattice.hpp mat_nn_opencl.hpp

bench_f32_opencl.exe: su3_nn_bench.cpp $(DEPENDS)
	$(CC) -DPRECISION=1 $(CFLAGS) $(INCLUDES) -o $@ su3_nn_bench.cpp $(LDLIBS)

bench_f64_opencl.exe: su3_nn_bench.cpp $(DEPENDS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ su3_nn_bench.cpp $(LDLIBS)

all: bench_f64_opencl.exe bench_f32_opencl.exe

clean:
	rm -f *opencl.exe
