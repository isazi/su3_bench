#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <math.h>
#include <omp.h>
#include "su3.h"
#include "lattice.h"

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
site *lattice;		// lattice store
int sites_on_node;	// number of sites in lattice
int nx,ny,nz,nt;	// lattice dimensions

int main(int argc, char *argv[])
{
  int i, j, k, l;
  int iters;
  double tstart, ttotal, tflop;
  double sum;
  struct rusage usage;

  su3_matrix b[4];
  site *c;

  // initialize
  nx=ny=nz=nt=N;
  sites_on_node=nx*ny*nz*nt;
  make_lattice();
  for(j=0;j<4;++j) for(k=0;k<3;++k) for(l=0;l<3;++l) {
    b[j].e[k][l] = (1.0/3.0) + 0.0*I;
  }
  if ((c = (site *)malloc(sites_on_node*sizeof(site))) == NULL) {
    printf("ERROR: no room for c vectors\n");
    exit(1);
  }

#ifdef DEBUG
  {
    printf("Sizeof site is %lu bytes\n", sizeof(site));
    printf("Address of link is %lx\n", (void *)&lattice->link - (void *)a);
    printf("Sizeof matrix is %d bytes\n", sizeof(su3_matrix));
    printf("Sizeof link is %d bytes\n", sizeof(lattice[0].link));
    printf("Sizeof site is %d bytes\n", sizeof(site));
    printf("Address of x, y, z, t is %x %x %x %x\n",
        (void *)&lattice->x - (void *)lattice, (void *)&lattice->y - (void *)lattice,
        (void *)&lattice->z - (void *)lattice, (void *)&lattice->t - (void *)lattice);
    su3_matrix *su3;
    // check alignment
    for(i=0;i<sites_on_node;++i) {
      su3 = &lattice[i].link[0];
      printf("Address of site %d is %lx, modulo 64 = %lx\n", i, su3, (unsigned long)su3 % 64);
    }
  }
#endif

  // Test loop
  printf("Number of sites = %d^4\n", N);
  printf("Executing %d iterations\n", ITERATIONS);
# ifdef OMP_TARGET
  #pragma omp target enter data map(to: lattice[0:sites_on_node], b[0:4], c[0:sites_on_node])
  tstart = omp_get_wtime();
  for (iters=0; iters<ITERATIONS; ++iters) {
    #pragma omp target teams distribute parallel for
#else
  tstart = omp_get_wtime();
  for (iters=0; iters<ITERATIONS; ++iters) {
    #pragma omp parallel for
# endif
    for(int i=0;i<sites_on_node;++i) {
      for (int j=0; j<4; ++j) {
        for(int k=0;k<3;k++) {
          for(int l=0;l<3;l++) {
            c[i].link[j].e[k][l]=0.0+0.0*I;
            for(int m=0;m<3;m++) {
              c[i].link[j].e[k][l] += lattice[i].link[j].e[k][m] * b[j].e[m][l];
            }
          }
        }
      }
    }
  }
  ttotal = omp_get_wtime() - tstart;
  printf("Total execution time = %.2f secs\n", ttotal);

# ifdef OMP_TARGET
  #pragma omp target exit data map(from: c[0:sites_on_node])
#endif
  // each iter of above loop is (3*3)*(12 mult + 10 add) = 108 mult + 90 add = 198 ops
  tflop = (double)iters * sites_on_node * 4.0 * 198.0;
  printf("Total GFLOP/s = %f\n", tflop / ttotal / 1.0e9);
  
  // calculate a checksum
  sum = 0.0;
  for (i=0;i<sites_on_node;++i) for(j=0;j<4;++j) for(k=0;k<3;++k) for(l=0;l<3;++l)
    sum += creal(c[i].link[j].e[k][l]);
  sum /= (double)sites_on_node;

  if ( sum != (double)(4.0*sizeof(su3_matrix)/(sizeof(Complx))))
    printf("Checksum FAILED: Sum = %lf\n", sum);

  printf("Total allocation for matrices = %.3f MiB\n", (2*sites_on_node*sizeof(site)+sizeof(b))/1048576.0);
  if (getrusage(RUSAGE_SELF, &usage) == 0)
    printf("Approximate memory usage = %.3f MiB\n", (float)usage.ru_maxrss/1024.0);

  free(c);
  free_lattice();
}

