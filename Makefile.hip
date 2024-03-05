#
# Vendor can be nvidia | amd
ifndef VENDOR
  VENDOR = nvidia
endif

ifeq ($(VENDOR),nvidia)
  HIPCC_FLAGS = -gencode arch=compute_$(ARCH),code=sm_$(ARCH)
else ifeq ($(VENDOR),amd)
  HIPCC_FLAGS = --offload-arch=gfx$(ARCH)
endif

# USE MILC Complex data types
#HIPCC_FLAGS += -DMILC_COMPLEX

CC = hipcc
CFLAGS = -O3 $(HIPCC_FLAGS) -DUSE_HIP
DEPENDS = su3.hpp lattice.hpp mat_nn_hip.hpp

bench_f32_hip.exe: su3_nn_bench.cpp $(DEPENDS)
	$(CC) $(CFLAGS) -DPRECISION=1 -o $@ su3_nn_bench.cpp $(LIBS)

bench_f64_hip.exe: su3_nn_bench.cpp $(DEPENDS)
	$(CC) $(CFLAGS) -o $@ su3_nn_bench.cpp $(LIBS)

all: bench_f64_hip.exe bench_f32_hip.exe

clean:
	rm -f *.exe
