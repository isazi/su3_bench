#
CC = nvcc -ccbin g++ -m64
CFLAGS = -gencode arch=compute_70,code=sm_70

INCLUDES = -DITERATIONS=100 -DLDIM=32 #-DDEBUG
LDLIBS = -lm 

DEPENDS = su3.hpp lattice.hpp

mult_su3_nn_f32.exe: mult_su3_nn_bench.cu $(DEPENDS)
	$(CC) -DPRECISION=1 $(CFLAGS) $(INCLUDES) -o $@ make_lattice.cu mult_su3_nn_bench.cu $(LDLIBS)

mult_su3_nn_f64.exe: mult_su3_nn_bench.cu $(DEPENDS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ make_lattice.cu mult_su3_nn_bench.cu $(LDLIBS)

all: mult_su3_nn_f64.exe mult_su3_nn_f32.exe

clean:
	rm -f *.exe *.o
