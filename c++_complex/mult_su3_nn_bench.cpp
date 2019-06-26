#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <math.h>
#include <omp.h>
#include "su3.hpp"
#include "lattice.hpp"
#include "c99_su3_inline.hpp"

#define DEBUG

#ifndef ITERATIONS
#  define ITERATIONS 100
#endif

// Lattice dimension = N^4
#ifndef N
#  define N 7
#endif

int main(int argc, char *argv[])
{
  // initialize the lattice
  site *lattice;
  int total_sites;
  make_lattice(&lattice, (int)N, &total_sites);

  // initialize resultant lattice
  su3_matrix b[4], *c[4];
  for(int j=0;j<4;++j) for(int k=0;k<3;++k) for(int l=0;l<3;++l) {
    b[j].e[k][l] = Complx((1.0/3.0), 0.0);
  }
  for(int j=0; j<4; ++j) {
    int errval;
    if ((errval = posix_memalign((void **)&c[j], ALIGN_N, total_sites*sizeof(su3_matrix))) != 0) {
      printf("ERROR: Insufficient memory for resultant allocation\n");
      exit(errval);
    }
  }

#ifdef DEBUG
  {
    printf("Sizeof site is %lu bytes\n", sizeof(site));
    printf("Sizeof matrix is %lu bytes\n", sizeof(su3_matrix));
    // check alignment
    for (int i=0; i<3; ++i) {
      site *s = &lattice[i];
      printf("Address of site %d is 0x%lx, modulo %d = %ld\n", 
             i, (unsigned long)s, ALIGN_N, (unsigned long)s % ALIGN_N);
    }
  }
#endif

  // Test loop
  printf("Number of sites = %d^4\n", N);
  printf("Executing %d iterations\n", ITERATIONS);
  double tstart = omp_get_wtime();
  for (int iters=0; iters<ITERATIONS; ++iters) {
    #pragma omp parallel for
    for(int i=0;i<total_sites;++i) for(int j=0;j<4;++j)
        mult_su3_nn( &lattice[i].link[j], &b[j], &c[j][i]);
  }
  double ttotal = omp_get_wtime() - tstart;
  printf("Total execution time = %.2f secs\n", ttotal);

  // each iter of above loop is (3*3)*(12 mult + 10 add) = 108 mult + 90 add = 198 ops
  double tflop = (double)ITERATIONS * total_sites * 4.0 * 198.0;
  printf("Total GFLOP/s = %f\n", tflop / ttotal / 1.0e9);
  
  // calculate a checksum
  Complx sum = Complx(0.0, 0.0);
  #pragma omp parallel for reduction(+:sum)
  for (int i=0;i<total_sites;++i) for(int j=0;j<4;++j) for(int k=0;k<3;++k) for(int l=0;l<3;++l)
    sum += c[j][i].e[k][l];
  sum /= Complx((Real)total_sites, 0.0);

  if ( round(real(sum)) != (4.0*sizeof(su3_matrix)/(sizeof(Complx))))
    printf("Checksum FAILED: Sum = (%lf + %lfi)\n", real(sum), imag(sum));

  // check memory usage
  printf("Total allocation for matrices = %.3f MB\n", total_sites*(sizeof(site)+4*sizeof(su3_matrix))/1048576.0);
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) == 0)
    printf("Approximate memory usage = %.3f MB\n", (float)usage.ru_maxrss/1024.0);

  // clean up and exit
  for(int j=0; j<4; ++j)
    free(c[j]);
  free_lattice(lattice);
}

