#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/resource.h>
#include <time.h>

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "lattice.hpp"

#ifndef ITERATIONS
#  define ITERATIONS 100
#endif
#ifndef LDIM
#  define LDIM 7       // Lattice size = LDIM^4
#endif
#ifndef PRECISION
#  define PRECISION 2  // 1->single, 2->double
#endif
#ifndef VERBOSE
#  define VERBOSE 1    // valid values: 0, 1 or 2
#endif

#define CUCHECK(err, s) \
  if (err != cudaSuccess) { \
        printf("%s (error code %d:%s)!\n", s, err, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
  }

//*******************  m_mat_nn.c  (in su3.a) ****************************
//  void mult_su3_nn( su3_matrix *a,*b,*c )
//  matrix multiply, no adjoints 
//  C  <-  A*B	
__global__ void k_mat_nn(
  const site* __restrict__ a,
  const su3_matrix* __restrict__ b,
  site* __restrict__ c,
  int total_sites)
{
  int mysite = blockDim.x * blockIdx.x + threadIdx.x;
#ifdef DEBUG
  printf("mysite = %d\n", mysite);
#endif

  for (int j=0; j<4; ++j) {
    for (int k=0;k<3;k++) {
      for (int l=0;l<3;l++){
        Complx cc = (0.0,0.0);
        for (int m=0;m<3;m++) {
          Complx bb = b[j].e[m][l]; __syncthreads();
          cc += a[mysite].link[j].e[k][m] * bb;
#ifdef DEBUG
          if (mysite==0 && m==2)
          printf("a[%d][%d]->e[%d][%d]=%f b[%d][%d]->e[%d][%d]=%f c[%d][%d]->e[%d][%d]=%f\n",
                  j,mysite,k,m,a[mysite].link[j].e[k][m].real(),
                  j,mysite,m,l,b[j].e[m][l].real(),
                  j,mysite,k,l,c[mysite].link[j].e[k][l].real());
#endif
        }
        c[mysite].link[j].e[k][l] = cc;
      }
    }
  }
}

int main(int argc, char *argv[])
{
  int opt;
  int threadsPerBlock=4;
  int blocksPerGrid;
  unsigned int iterations=ITERATIONS;
  unsigned int ldim=LDIM;
  unsigned int verbose=VERBOSE;

  // parse command line for parameters
  while ((opt=getopt(argc, argv, "i:n:t:v:")) != -1) {
    switch (opt) {
    case 'i':
      iterations = atoi(optarg);
      break;
    case 'n':
      ldim = atoi(optarg);
      break;
    case 't':
      threadsPerBlock = atoi(optarg);
      break;
    case 'v':
      verbose = atoi(optarg);
      break;
    default: 
      fprintf(stderr, "Usage: %s [-i iterations] [-n lattice dimension] \
[-t threads per block] [-v verbosity]\n", argv[0]);
      exit (1);
    }
  }

  // allocate and initialize the working lattices and B link matrix
  int total_sites = ldim*ldim*ldim*ldim;
  // A
  thrust::host_vector<site> a(total_sites);
  make_lattice(&a[0], ldim);
  int size_a = a.size()*sizeof(site);
  // B
  thrust::host_vector<su3_matrix> b(4);
  init_link(&b[0], Complx(1.0/3.0, 0.0));
  int size_b = b.size()*sizeof(su3_matrix);
  // C
  thrust::host_vector<site> c(total_sites);
  int size_c = c.size()*sizeof(site);

  // Device initialization
  int device;
  cudaError_t cuErr;
  CUCHECK(cudaGetDevice(&device), "Unable to find a device");;
  if (verbose >= 2) {
    struct cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device);
    printf("Using device: %s\n", device_prop.name);
  }

  // Declare target storage and copy A and B
  site *d_a, *d_c;
  su3_matrix *d_b;
  cuErr = cudaMalloc((void **)&d_a, total_sites*sizeof(site));
  CUCHECK(cuErr, "Unable to allocate array d_a");
  cuErr = cudaMalloc((void **)&d_b, 4*sizeof(su3_matrix));
  CUCHECK(cuErr, "Unable to allocate array d_b");
  cuErr = cudaMalloc((void **)&d_c, total_sites*sizeof(site));
  CUCHECK(cuErr, "Unable to allocate array d_c");
  cudaMemcpy(d_a, a.data(), size_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), size_b, cudaMemcpyHostToDevice);

  if (verbose >= 1) {
    printf("Number of sites = %d^4\n", ldim);
    printf("Executing %d iterations\n", iterations);
    printf("Threads per block set to %d\n", threadsPerBlock);
  }

  // benchmark loop
  blocksPerGrid = (total_sites + threadsPerBlock - 1)/threadsPerBlock;
  clock_t tstart = clock();
  for (int iters=0; iters<iterations; ++iters) {
      k_mat_nn<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, total_sites);
  }
  cudaDeviceSynchronize();
  CUCHECK(cudaGetLastError(), "k_mat_nn kernel Failed");

  double ttotal = (double)(clock()-tstart)/CLOCKS_PER_SEC;
  if (verbose >= 1)
    printf("Total execution time = %.3f secs\n", ttotal);

  // each iter of above loop is (3*3)*(12 mult + 10 add) = 108 mult + 90 add = 198 ops
  double tflop = (double)iterations * total_sites * 4.0 * 198.0;
  printf("Total GFLOP/s = %.3f\n", tflop / ttotal / 1.0e9);
  
  // copy data back from device
  cudaMemcpy(c.data(), d_c, size_c, cudaMemcpyDeviceToHost);

  // calculate a checksum
  double sum = 0.0;
  #pragma omp parallel for reduction(+:sum)
  for (int i=0;i<total_sites;++i) for(int j=0;j<4;++j) for(int k=0;k<3;++k) for(int l=0;l<3;++l) {
    sum += c[i].link[j].e[k][l].real();
#ifdef DEBUG
    if (i == 0)
      printf("c[%d][%d]->e[%d][%d]=%f, sum = %f\n",j,i,k,l,c[i].link[j].e[k][l].real(),sum);
#endif
  }
  sum /= (double)total_sites;

  if ( round(sum) != (4.0*sizeof(su3_matrix)/(sizeof(Complx))))
    printf("Checksum FAILED: Sum = %lf\n", sum);

  if (verbose >= 2) {
    // check memory usage
    printf("Total allocation for matrices = %.3f MiB\n", 
           ((float)sizeof(site)*(a.capacity()+c.capacity())+sizeof(su3_matrix)*b.capacity())/1048576.0);
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0)
      printf("Approximate memory usage = %.3f MiB\n", (float)usage.ru_maxrss/1024.0);
  }

  // Deallocate
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

}

