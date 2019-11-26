#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/resource.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <chrono>
typedef std::chrono::system_clock Clock;

#ifndef ITERATIONS
#  define ITERATIONS 100
#endif
#ifndef LDIM
#  define LDIM 7       // Lattice size = LDIM^4
#endif
#ifndef PRECISION
#  define PRECISION 2  // 1->single, 2->double
#endif

// Global variables
unsigned int verbose=1;

// OpenCL 1.2 doesn't support complex data types
#if defined(USE_OPENCL)
  #define MILC_COMPLEX
#endif

#include "lattice.hpp"

#ifdef USE_CUDA
  #include <thrust/host_vector.h>
  #include "mat_nn_cuda.hpp"
#elif  USE_OPENMP
  #include "mat_nn_openmp.hpp"
#elif  USE_OPENCL
  #define MILC_COMPLEX
  #include "mat_nn_opencl.hpp"
#elif USE_SYCL
  #include "mat_nn_sycl.hpp"
#else
  #error Unknown programming model
#endif

// initializes su3_matrix to a given value
void init_link(su3_matrix *s, Complx val) {
  for(int j=0; j<4; ++j) for(int k=0; k<3; ++k) for(int l=0; l<3; ++l) {
#ifdef MILC_COMPLEX
    s[j].e[k][l].real=val.real;
    s[j].e[k][l].imag=val.imag;
#else
    s[j].e[k][l]=val;
#endif
  }
}

// initializes a lattice site 
void make_lattice(site *s, size_t n, Complx val) {
  int nx=n;
  int ny=n;
  int nz=n;
  int nt=n;
  for(int t=0;t<nt;t++) {
    int i=t*nz*ny*nx;
    for(int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++,i++){
      s[i].x=x; s[i].y=y; s[i].z=z; s[i].t=t;
      s[i].index = x+nx*(y+ny*(z+nz*t));
      if( (x+y+z+t)%2 == 0)
        s[i].parity=EVEN;
      else
        s[i].parity=ODD;
      init_link(&s[i].link[0], val);
    }
  }
}

// Main
int main(int argc, char *argv[])
{
  int opt;
  size_t iterations = ITERATIONS;
  size_t ldim = LDIM;
  size_t threads_per_group = 0;
#ifdef USE_SYCL
  int device = 1;   // ComputeCpp assigns 0 to Host Device
#else
  int device = 0;
#endif

  // parse command line for parameters
  while ((opt=getopt(argc, argv, "hi:l:t:v:d:")) != -1) {
    switch (opt) {
    case 'i':
      iterations = atoi(optarg);
      break;
    case 'l':
      ldim = atoi(optarg);
      break;
    case 't':
      threads_per_group = atoi(optarg);
      break;
    case 'd':
      device = atoi(optarg);
      break;
    case 'v':
      verbose = atoi(optarg);
      break;
    case 'h':
    default: 
      fprintf(stderr, "Usage: %s [-i iterations] [-l lattice dimension] \
[-t threads per workgroup] [-v verbosity level [0,1,2,3]]\n", argv[0]);
      exit (1);
    }
  }

  // allocate and initialize the working lattices and B su3 matrices
  size_t total_sites = ldim*ldim*ldim*ldim;
  Complx val;
#ifndef USE_CUDA
  std::vector<site> a(total_sites);
  std::vector<su3_matrix> b(4);
  std::vector<site> c(total_sites);
#else
  thrust::host_vector<site> a(total_sites);
  thrust::host_vector<su3_matrix> b(4);
  thrust::host_vector<site> c(total_sites);
#endif

  // initialize the lattices
  make_lattice(a.data(), ldim, Complx{1.0,0.0});
  init_link(b.data(), Complx{1.0/3.0,0.0});

  if (verbose >= 1) {
    printf("Number of sites = %zu^4\n", ldim);
    printf("Executing %zu iterations\n", iterations);
    if (threads_per_group != 0)
      printf("Threads per group = %zu\n", threads_per_group);
  }

  // benchmark call
  double ttotal = su3_mat_nn(a, b, c, total_sites, iterations, threads_per_group, device);
  if (verbose >= 1)
    printf("Total execution time = %.3f secs\n", ttotal);

  // calculate flops/s, etc.
  // each iter of above kernel is (3*3)*(12 mult + 12 add) = 108 mult + 108 add = 216 ops
  double tflop = (double)iterations * total_sites * 4.0 * 216.0;
  printf("Total GFLOP/s = %.3f\n", tflop / ttotal / 1.0e9);

  // calculate a checksum
  double sum = 0.0;
  #pragma omp parallel for reduction(+:sum)
  for (int i=0;i<total_sites;++i) for(int j=0;j<4;++j) for(int k=0;k<3;++k) for(int l=0;l<3;++l) {
#ifdef MILC_COMPLEX
    sum += c[i].link[j].e[k][l].real;
#else
    sum += c[i].link[j].e[k][l].real();
#endif
  }
  sum /= (double)total_sites;

  if ( round(sum) != (4.0*sizeof(su3_matrix)/(sizeof(Complx))))
    printf("Checksum FAILED: Sum = %lf\n", sum);

  // check memory usage
  if (verbose >= 2) {
    printf("Total allocation for matrices = %.3f MiB\n", 
           ((float)sizeof(site)*(a.capacity()+c.capacity())+sizeof(su3_matrix)*b.capacity())/1048576.0);
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0)
      printf("Approximate memory usage = %.3f MiB\n", (float)usage.ru_maxrss/1024.0);
  }
}

