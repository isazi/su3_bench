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

DEFINES = -DUSE_HIP -DMILC_COMPLEX

CC = hipcc
CFLAGS = -O3
DEPENDS = su3.hpp lattice.hpp mat_nn_hip.hpp

bench_f32_hip.exe: su3_nn_bench.cpp $(DEPENDS)
	$(CC) $(CFLAGS) $(DEFINES) -DPRECISION=1 -o $@ su3_nn_bench.cpp $(LIBS)

bench_f64_hip.exe: su3_nn_bench.cpp $(DEPENDS)
	$(CC) $(CFLAGS) $(DEFINES) -o $@ su3_nn_bench.cpp $(LIBS)

all: bench_f64_hip.exe bench_f32_hip.exe

clean:
	rm -f *.exe
