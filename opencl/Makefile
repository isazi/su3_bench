#
CC = clang++-9
CFLAGS = -std=c++11 -g -O3 -fopenmp -Wno-ignored-attributes -Wno-deprecated-declarations
CFLAGS += -I$(ROCM_OCL)/include

INCLUDES = -DVERBOSE=2 -DITERATIONS=10 -DLDIM=32 #-DPRECISION=1
LDLIBS = -lm 
LDLIBS += -L$(ROCM_OCL)/lib/x86_64 -lOpenCL

DEPENDS = su3.hpp lattice.hpp

mult_su3_nn_bench.exe: Makefile mult_su3_nn_bench.cpp make_lattice.o $(DEPENDS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ make_lattice.o mult_su3_nn_bench.cpp $(LDLIBS)

make_lattice.o: Makefile make_lattice.cpp $(DEPENDS)
	$(CC) $(CFLAGS) $(INCLUDES) -c make_lattice.cpp

all: mult_su3_nn_bench.exe

clean:
	rm -f *.exe *.o
