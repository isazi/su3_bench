#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <math.h>
#include "su3.h"
#include "lattice.h"
#include "c99_su3_inline.h"

#undef DEBUG

#ifndef ITERATIONS
#  define ITERATIONS 100
#endif

#ifndef N
#  define N 7
#endif

// externs
extern double mysecond();

// global definitions for MILC routines
int sites_on_node;	// number of sites in lattice
int nx,ny,nz,nt;	// lattice dimensions

int main(int argc, char *argv[])
{
  int i, j, k, l;
  int iters;
  double tstart, ttotal, tflop;
  double complex sum;
  struct rusage usage;

  su3_vector a[4], b[4], *c[4];
  double s = 1.0/3.0;

  // initialize
  nx=ny=nz=nt=N;
  sites_on_node=nx*ny*nz*nt;
  for(j=0; j<4; ++j) for(k=0;k<3;++k) {
    b[j].c[k] = 1.0 + 0.0*I;
    a[j].c[k] = (2.0/3.0) + 0.0*I;
  }
  for(j=0; j<4; ++j) {
    if ((c[j] = (su3_vector *)malloc(sites_on_node*sizeof(su3_vector))) == NULL) {
      printf("ERROR: no room for c vectors\n");
      exit(1);
    }
  }
  for(j=0; j<4; ++j) for(i=0; i<sites_on_node; ++i) for(k=0; k<3; ++k)
      c[j][i].c[k] = 0.0+0.0*I;

  // Test loop
  printf("Number of sites = %d^4\n", N);
  printf("Executing %d iterations\n", ITERATIONS);
  tstart = mysecond();
  for (iters=0; iters<ITERATIONS; ++iters) {
# ifdef OMP_TARGET
    #pragma omp target teams distribute parallel for
# else
    #pragma omp parallel for
# endif
    for(int i=0;i<sites_on_node;++i) for(int j=0;j<4;++j)
      scalar_mult_add_su3_vector(&a[j], &b[j], s, &c[j][i]);
  }
  ttotal = mysecond() - tstart;
  printf("Total execution time = %.2f secs\n", ttotal);

  // each iter of above loop is 3*(4 mult + 2 add) + (2 add) = 12 mult + 8 add = 20 ops
  tflop = (double)iters * sites_on_node * 4.0 * 20;
  printf("Total GFLOP/s = %f\n", tflop / ttotal / 1.0e9);
  
  // calculate a checksum
  sum = 0.0+0.0*I;
  for (i=0;i<sites_on_node;++i) for(j=0;j<4;++j) for(k=0;k<3;++k)
    sum += c[j][i].c[k];
  sum /= (double)sites_on_node+0.0*I;

  if ( round(creal(sum)) != (4.0*sizeof(su3_vector)/(sizeof(complex double))))
    printf("Checksum FAILED: Sum = (%f + %fi)\n", creal(sum), cimag(sum));

  printf("Total allocation for matrices = %.3f MB\n", sites_on_node*(4*sizeof(su3_vector))/1048576.0);
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
    printf("Approximate memory usage = %.3f MB\n", (float)usage.ru_maxrss/1024.0);
  }
  
  for(j=0; j<4; ++j)
    free(c[j]);

}

