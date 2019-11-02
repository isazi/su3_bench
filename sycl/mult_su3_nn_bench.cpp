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

#define USE_ND_ITEM

// Global variables
class my_kernel;
unsigned int verbose=VERBOSE;

double k_mat_nn(const std::vector<site> &a, const std::vector<su3_matrix> &b, std::vector<site> &c, 
              const size_t total_sites, size_t wgsize, const size_t iterations)
{ 
  // Create a SYCL queue
  cl::sycl::queue queue;
  if (verbose >= 2)
    std::cout << "Using device " << queue.get_device().get_info<cl::sycl::info::device::name>() << "\n";

  // Pre-build the kernel
  auto build_start = Clock::now();
  cl::sycl::program program = cl::sycl::program(queue.get_context());
  program.build_with_kernel_type<my_kernel>();
  double build_time = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-build_start).count();
  if (verbose >= 2)
    std::cout << "Time to build kernel = " << build_time/1.0e6 << " secs\n";

#ifdef USE_ND_ITEM
  if (wgsize == 0) {
    wgsize = queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
    if (verbose >= 1)
      std::cout << "Setting workgroup size to " << wgsize << "\n";
    if (verbose >= 2) {
      std::cout << "max compute units = " 
	        << queue.get_device().get_info<cl::sycl::info::device::max_compute_units>() << "\n";
      std::cout << "max workgroup size = " 
	        << queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>() << "\n";
    }
  }
#endif

  // wrap arrays in SYCL buffers, suppling global memory pointer implicitly copies the data to the device
  cl::sycl::buffer<site, 1>       a_buf {a.data(), cl::sycl::range<1> {total_sites}};
  cl::sycl::buffer<su3_matrix, 1> b_buf {b.data(), cl::sycl::range<1> {4}};
  // just create the c buffer on the device, no copy necessary
  cl::sycl::buffer<site, 1>       c_buf {cl::sycl::range<1> {total_sites}};

  // benchmark loop
  auto tstart = Clock::now();
  for (int iters=0; iters<iterations; ++iters) {
    // create a command_group to issue commands
    queue.submit([&](cl::sycl::handler& cgh) {
      // request access to the host buffers
      auto d_a = a_buf.get_access<cl::sycl::access::mode::read>(cgh);
      auto d_b = b_buf.get_access<cl::sycl::access::mode::read>(cgh);
      auto d_c = c_buf.get_access<cl::sycl::access::mode::write>(cgh);

      // Lambda function defines the kernel scope
#ifdef USE_ND_ITEM
      cgh.parallel_for<class my_kernel>(program.get_kernel<my_kernel>(), 
        cl::sycl::nd_range<1> {total_sites, wgsize}, 
	[=](cl::sycl::nd_item<1> item) { 
        size_t idx = item.get_global_id(0);
#else
      cgh.parallel_for<class my_kernel>(program.get_kernel<my_kernel>(), 
        cl::sycl::range<1> {total_sites}, 
	[=](cl::sycl::id<1> idx) { 
#endif
        for (int j=0; j<4; ++j) {
          for (int k=0;k<3;k++) {
            for (int l=0;l<3;l++){
#if 1
              d_c[idx].link[j].e[k][l].real=0.0;
              d_c[idx].link[j].e[k][l].imag=0.0;
              for (int m=0;m<3;m++) {
                CMULSUM(d_a[idx].link[j].e[k][m], d_b[j].e[m][l], d_c[idx].link[j].e[k][l]);
		//d_c[idx].link[j].e[k][l].real += d_a[idx].link[j].e[k][m].real * d_b[j].e[m][l].real;
              }
#else
	      ;
#endif
            }
          }
        }
      }); // end of the kernel lambda function
    });   // end of commands group
  } // end of iteration loop
  queue.wait();

  double ttotal = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tstart).count();

  // Copy the C buffer back to host memory
  auto d_c = c_buf.get_access<cl::sycl::access::mode::read>();
  for (size_t i = 0; i < total_sites; ++i)
	  c[i] = d_c[i];

  return (ttotal /= 1.0e6);
} // end of SYCL block

int main(int argc, char *argv[])
{
  int flags, opt;
  size_t iterations=ITERATIONS;
  size_t ldim=LDIM;
  size_t wgsize = 32;

  // parse command line for parameters
  while ((opt=getopt(argc, argv, "i:l:s:g:v:")) != -1) {
    switch (opt) {
    case 'i':
      iterations = atoi(optarg);
      break;
    case 'l':
      ldim = atoi(optarg);
      break;
    case 'g':
      wgsize = atoi(optarg);
      break;
    case 'v':
      verbose = atoi(optarg);
      break;
    default: 
      fprintf(stderr, "Usage: %s [-i iterations] [-l lattice dimension] \
[-g workgroup size] [-v verbosity]\n", argv[0]);
      exit (1);
    }
  }

  // allocate and initialize the working lattices and B link matrix
  size_t total_sites = ldim*ldim*ldim*ldim;
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
    printf("Number of sites = %zu^4\n", ldim);
    printf("Executing %zu iterations\n", iterations);
    if (wgsize != 0)
      printf("Workgroup size = %zu\n", wgsize);
  }

  // benchmark call
  double ttotal = k_mat_nn(a, b, c, total_sites, wgsize, iterations);
  if (verbose >= 1)
    printf("Total execution time = %.3f secs\n", ttotal);

  // calculate flops/s, etc.
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

