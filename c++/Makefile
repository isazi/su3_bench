# GCC or Clang
#CC = g++-8
CC = clang++
CFLAGS = -std=c++11 -g -O3 -fopenmp -Wno-ignored-attributes -Wno-deprecated-declarations
CFLAGS += -fopenmp-targets=nvptx64-nvidia-cuda

# Mentor Graphics gcc
# Note, -O2 and -O3 result in seg faults
#CC = x86_64-none-linux-gnu-g++ 
#CFLAGS = -std=c++11 -fopenmp -foffload=-march=gfx900

INCLUDES = -DVERBOSE=2 -DITERATIONS=100 -DLDIM=32 #-DPRECISION=1
LDLIBS = -lm 

DEPENDS = Makefile su3.hpp lattice.hpp

all: mult_su3_nn-wtarget.exe mult_su3_nn-wotarget.exe

wotarget: mult_su3_nn-wotarget.exe

mult_su3_nn-wtarget.exe: mult_su3_nn_bench.cpp make_lattice.o $(DEPENDS)
	$(CC) $(CFLAGS) -DOMP_TARGET $(INCLUDES) -o $@ make_lattice.o mult_su3_nn_bench.cpp $(LDLIBS)

mult_su3_nn-wotarget.exe: mult_su3_nn_bench.cpp make_lattice.o $(DEPENDS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ make_lattice.o mult_su3_nn_bench.cpp $(LDLIBS)

make_lattice.o: make_lattice.cpp $(DEPENDS)
	$(CC) $(CFLAGS) $(INCLUDES) -c make_lattice.cpp

clean:
	rm -f *.exe *.o
