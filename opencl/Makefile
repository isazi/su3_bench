#
CC = g++
CFLAGS = -std=c++11 -g -O3 -fopenmp -Wno-ignored-attributes
CFLAGS += -I$(CUDA_ROOT)/include

INCLUDES = -DITERATIONS=1 -DLDIM=2 #-DPRECISION=1
LDLIBS = -lm 
LDLIBS += -L$(CUDA_ROOT)/lib64 -lOpenCL

DEPENDS = su3.hpp lattice.hpp

mult_su3_nn_bench.exe: Makefile mult_su3_nn_bench.cpp make_lattice.o $(DEPENDS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ make_lattice.o mult_su3_nn_bench.cpp $(LDLIBS)

make_lattice.o: Makefile make_lattice.cpp $(DEPENDS)
	$(CC) $(CFLAGS) $(INCLUDES) -c make_lattice.cpp

all: mult_su3_nn_bench.exe

clean:
	rm -f *.exe *.o
