#
# GCC
CC = clang
CFLAGS = -std=c99 -O3 -fopenmp
CFLAGS += -ffast-math -fopenmp-targets=nvptx64-nvidia-cuda

# Intel
#CC = icc
#CFLAGS = -std=c99 -O3 -qopenmp -qopt-report=5 -qopt-report-phase=vec

INCLUDES = -DN=32 #-DPRECISION=1
LDLIBS = -lm 

#all: mult_su3_nn_wtarget.exe mult_su3_nn_wotarget.exe \
#     mult_su3_mat_vec_bench.exe \
#     mult_su3_mat_vec_sum_bench.exe \
#     mult_adj_su3_mat_vec_bench.exe \
#     scalar_mult_add_su3_vector_bench.exe

#mult_su3_nn_wtarget.exe: Makefile mult_su3_nn_bench.c make_lattice.o lattice.h
#	$(CC) $(CFLAGS) -DOMP_TARGET $(INCLUDES) -o $@ mult_su3_nn_bench.c make_lattice.o $(LDLIBS)

mult_su3_nn_wtarget.exe: Makefile mult_su3_nn_bench.c lattice.h
	$(CC) $(CFLAGS) $(INCLUDES) -c make_lattice.c
	$(CC) $(CFLAGS) -DOMP_TARGET $(INCLUDES) -o $@ mult_su3_nn_bench.c make_lattice.o $(LDLIBS)

mult_su3_nn_wotarget.exe: Makefile mult_su3_nn_bench.c make_lattice.o lattice.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ mult_su3_nn_bench.c make_lattice.o $(LDLIBS)

scalar_mult_add_su3_vector_bench.exe: Makefile scalar_mult_add_su3_vector_bench.c su3.h c99_su3_inline.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ scalar_mult_add_su3_vector_bench.c $(LDLIBS)

mult_adj_su3_mat_vec_bench.exe: Makefile mult_adj_su3_mat_vec_bench.c make_lattice.o lattice.h su3.h c99_su3_inline.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ mult_adj_su3_mat_vec_bench.c make_lattice.o  $(LDLIBS)

mult_su3_mat_vec_sum_bench.exe: Makefile mult_su3_mat_vec_sum_bench.c make_lattice.o lattice.h su3.h c99_su3_inline.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ mult_su3_mat_vec_sum_bench.c make_lattice.o  $(LDLIBS)

mult_su3_mat_vec_bench.exe: Makefile mult_su3_mat_vec_bench.c make_lattice.o lattice.h su3.h c99_su3_inline.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ mult_su3_mat_vec_bench.c make_lattice.o  $(LDLIBS)

make_lattice.o: make_lattice.c Makefile
	$(CC) $(CFLAGS) $(INCLUDES) -c make_lattice.c

clean:
	rm -f *.exe *.o *.optrpt
