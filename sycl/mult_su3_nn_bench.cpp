#include <CL/sycl.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/resource.h>
#include <math.h>

#include <vector>
#include <iostream>
#include <chrono>
typedef std::chrono::system_clock Clock;

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

// Global variables
class my_kernel;
unsigned int verbose=VERBOSE;

double k_mat_nn(const std::vector<site> &a, const std::vector<su3_matrix> &b, std::vector<site> &c, 
              const unsigned int total_sites, const unsigned int iterations)
{ 
  // Create a SYCL queue
  cl::sycl::queue queue;
  if (verbose >= 2)
    std::cout << "Using device " << queue.get_device().get_info<cl::sycl::info::device::name>() << "\n";

  // wrap arrays in SYCL buffers
  // since buffers are inside the SYCL block, c_buf gets copied back to the host when it's destroyed
  cl::sycl::buffer<site, 1>       a_buf {a.data(), cl::sycl::range<1> {total_sites}};
  cl::sycl::buffer<su3_matrix, 1> b_buf {b.data(), cl::sycl::range<1> {4}};
  cl::sycl::buffer<site, 1>       c_buf {c.data(), cl::sycl::range<1> {total_sites}};

  // benchmark loop
  auto tstart = Clock::now();
  for (int iters=0; iters<iterations; ++iters) {
    // create a command_group to issue commands
    queue.submit([&](cl::sycl::handler& cgh) {
      // request access to the host buffers
      auto d_a = a_buf.get_access<cl::sycl::access::mode::read>(cgh);
      auto d_b = b_buf.get_access<cl::sycl::access::mode::read>(cgh);
      auto d_c = c_buf.get_access<cl::sycl::access::mode::read_write>(cgh);

      // Lambda function defines the kernel scope
      cgh.parallel_for<class my_kernel>(cl::sycl::range<1> {total_sites}, [=](cl::sycl::id<1> idx) { 
        for (int j=0; j<4; ++j) {
          for (int k=0;k<3;k++) {
            for (int l=0;l<3;l++){
              d_c[idx].link[j].e[k][l].real=0.0;
              d_c[idx].link[j].e[k][l].imag=0.0;
              for (int m=0;m<3;m++) {
                CMULSUM(d_a[idx].link[j].e[k][m], d_b[j].e[m][l], d_c[idx].link[j].e[k][l]);
              }
            }
          }
        }
      }); // end of the kernel function
    });   // end of our commands for this queue
  } // end of iteration loop

  // wait for all queue submissions to complete
  queue.wait();
  double ttotal = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tstart).count();
  return (ttotal /= 1.0e6);

} // end of SYCL block, all objects are destroyed, and c_buf data copied back to the host

int main(int argc, char *argv[])
{
  int flags, opt;
  unsigned int iterations=ITERATIONS;
  unsigned int ldim=LDIM;
  unsigned int sites_per_wi = 1;
  unsigned int wgsize = 0;

  // parse command line for parameters
  while ((opt=getopt(argc, argv, "i:l:s:g:v:")) != -1) {
    switch (opt) {
    case 'i':
      iterations = atoi(optarg);
      break;
    case 'l':
      ldim = atoi(optarg);
      break;
    case 's':
      sites_per_wi = atoi(optarg);
      break;
    case 'g':
      wgsize = atoi(optarg);
      break;
    case 'v':
      verbose = atoi(optarg);
      break;
    default: 
      fprintf(stderr, "Usage: %s [-i iterations] [-l lattice dimension] \
[-s sites per work item] [-g workgroup size] [-v verbosity]\n", argv[0]);
      exit (1);
    }
  }

  // allocate and initialize the working lattices and B link matrix
  unsigned int total_sites = ldim*ldim*ldim*ldim;
  unsigned int total_wi = total_sites / sites_per_wi;
  if (total_wi > total_sites) {
    fprintf(stderr, "ERROR: total work items %d > total sites %d\n", total_wi, total_sites);
    exit (1);
  }
  if (total_wi > 0 && total_wi < wgsize) {
    fprintf(stderr, "ERROR: total work items %d < work group size %d\n", total_wi, wgsize);
    exit (1);
  }
  // A
  std::vector<site> a(total_sites);
  make_lattice(a.data(), ldim);
  // B
  std::vector<su3_matrix> b(4);
  Complx val = {1.0/3.0,0.0};
  init_link(b.data(), val);
  // C
  std::vector<site> c(total_sites);

  if (verbose >= 1) {
    printf("Number of sites = %d^4\n", ldim);
    printf("Executing %d iterations\n", iterations);
    printf("Total work items = %d\n", total_wi);
    if (wgsize != 0)
      printf("Workgroup size = %d\n", wgsize);
  }

  // initial call to force kernel compile
  k_mat_nn(a, b, c, total_sites, 1);

  // benchmark call
  double ttotal = k_mat_nn(a, b, c, total_sites, iterations);

  // calculate flops/s, etc.
  if (verbose >= 1)
    printf("Total execution time = %.3f secs\n", ttotal);
  // each iter of above loop is (3*3)*(12 mult + 10 add) = 108 mult + 90 add = 198 ops
  double tflop = (double)iterations * total_sites * 4.0 * 198.0;
  printf("Total GFLOP/s = %.3f\n", tflop / ttotal / 1.0e9);

  // calculate a checksum
  double sum = 0.0;
  #pragma omp parallel for reduction(+:sum)
  for (int i=0;i<total_sites;++i) for(int j=0;j<4;++j) for(int k=0;k<3;++k) for(int l=0;l<3;++l) {
    sum += c[i].link[j].e[k][l].real;
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

