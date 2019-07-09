#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <math.h>
#include <omp.h>

#include <vector>

#include "su3.hpp"
#include "lattice.hpp"
#include "c99_su3_inline.hpp"

#undef DEBUG

#ifndef ITERATIONS
#  define ITERATIONS 100
#endif

// Lattice size = LDIM^4
#ifndef LDIM
#  define LDIM 7
#endif

int main(int argc, char *argv[])
{
  // allocate initialize the lattice and working matrices
  int total_sites = LDIM*LDIM*LDIM*LDIM;

  std::vector<site> lattice(total_sites);
  make_lattice(&lattice[0], LDIM);

  std::vector<su3_matrix> b(4);
  for(int j=0;j<4;++j) for(int k=0;k<3;++k) for(int l=0;l<3;++l) {
    b[j].e[k][l] = Complx((1.0/3.0), 0.0);
  }

  std::vector<su3_matrix> c(4*total_sites);

#ifdef DEBUG
  {
    printf("Total number of sites = %d\n", total_sites);
    printf("Sizeof site is %lu bytes\n", sizeof(site));
    printf("Sizeof matrix is %lu bytes\n", sizeof(su3_matrix));
    printf("Sizeof lattice is %f MB\n", (float)sizeof(site)*lattice.size()/1048576.0);
    printf("Sizeof b[] is %f MB\n", (float)sizeof(su3_matrix)*b.size()/1048576.0);
    printf("Sizeof c[] is %f MB\n", (float)sizeof(su3_matrix)*c.size()/1048576.0);
    // check alignment
    for (int i=0; i<3; ++i) {
      site *s = &lattice[i];
      printf("Address of site %d is 0x%lx, modulo %d = %ld\n", 
             i, (unsigned long)s, ALIGN_N, (unsigned long)s % ALIGN_N);
    }
  }
#endif

  printf("Number of sites = %d^4\n", LDIM);
  printf("Executing %d iterations\n", ITERATIONS);
  // benchmark loop
  double tstart = omp_get_wtime();
  for (int iters=0; iters<ITERATIONS; ++iters) {
    #pragma omp parallel for
    for(int i=0;i<total_sites;++i) for(int j=0;j<4;++j)
        mult_su3_nn( &lattice[i].link[j], &b[j], &c.at(j*total_sites+i));
  }
  double ttotal = omp_get_wtime() - tstart;
  printf("Total execution time = %.2f secs\n", ttotal);

  // each iter of above loop is (3*3)*(12 mult + 10 add) = 108 mult + 90 add = 198 ops
  double tflop = (double)ITERATIONS * total_sites * 4.0 * 198.0;
  printf("Total GFLOP/s = %f\n", tflop / ttotal / 1.0e9);
  
  // calculate a checksum
  double sum = 0.0;
  #pragma omp parallel for reduction(+:sum)
  for (int i=0;i<total_sites;++i) for(int j=0;j<4;++j) for(int k=0;k<3;++k) for(int l=0;l<3;++l)
    sum += real(c.at(j*total_sites+i).e[k][l]);
  sum /= (double)total_sites;

  if ( round(sum) != (4.0*sizeof(su3_matrix)/(sizeof(Complx))))
    printf("Checksum FAILED: Sum = %lf\n", sum);

  // check memory usage
  printf("Total allocation for matrices = %.3f MB\n", total_sites*(sizeof(site)+4*sizeof(su3_matrix))/1048576.0);
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) == 0)
    printf("Approximate memory usage = %.3f MB\n", (float)usage.ru_maxrss/1024.0);

}

