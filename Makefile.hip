#
HIP_PLATFORM=$(shell hipconfig --platform)
ifeq (${HIP_PLATFORM}, nvcc)
	HIPCC_FLAGS = -gencode=arch=compute_70,code=sm_70 -DUSE_CUDA
else
	# std::complex no longer works with hipcc 3.6, so use MILC_COMPLEX
	HIPCC_FLAGS = -DMILC_COMPLEX
endif

CC = hipcc $(HIPCC_FLAGS)
CFLAGS = --std c++14 
INCLUDES = -DUSE_HIP
DEPENDS = su3.hpp lattice.hpp mat_nn_hip.hpp

bench_f32_hip.exe: su3_nn_bench.cpp $(DEPENDS)
	$(CC) $(CFLAGS) $(INCLUDES) -DPRECISION=1 -o $@ su3_nn_bench.cpp $(LIBS)

bench_f64_hip.exe: su3_nn_bench.cpp $(DEPENDS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ su3_nn_bench.cpp $(LIBS)

all: bench_f64_hip.exe bench_f32_hip.exe

clean:
	rm -f *.exe
