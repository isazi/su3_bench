#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <math.h>
#include <omp.h>

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
typedef std::chrono::system_clock Clock;

#include <CL/cl.hpp>
#ifndef DEVICE
#  define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

#include "lattice.hpp"

#ifndef ITERATIONS
#  define ITERATIONS 100
#endif
#ifndef LDIM
#  define LDIM 7  // Lattice size = LDIM^4
#endif
#ifndef VERBOSE
#  define VERBOSE 1  // valid values 0, 1 or 2
#endif
#ifndef DEBIG
#  undef DEBUG
#endif

static inline std::string loadProgram(std::string input)
{
    std::ifstream stream(input.c_str());
    if (!stream.is_open()) {
        std::cout << "Cannot open file: " << input << std::endl;
        exit(1);
    }

     return std::string(
        std::istreambuf_iterator<char>(stream),
        (std::istreambuf_iterator<char>()));
}

int main(int argc, char *argv[])
{
  // allocate and initialize the working lattices and B link matrix
  int total_sites = LDIM*LDIM*LDIM*LDIM;
  // A
  std::vector<site> a(total_sites);
  make_lattice(&a[0], LDIM);
  // B
  std::vector<su3_matrix> b(4);
  init_link(&b[0], Complx((1.0/3.0), 0.0));
  // C
  std::vector<site> c(total_sites);

#ifdef DEBUG
  {
    printf("Total number of sites = %d\n", total_sites);
    printf("Sizeof site is %lu bytes\n", sizeof(site));
    printf("Sizeof matrix is %lu bytes\n", sizeof(su3_matrix));
    printf("Sizeof lattice is %f MB\n", (float)sizeof(site)*a.size()/1048576.0);
    printf("Sizeof b[] is %f MB\n", (float)sizeof(su3_matrix)*b.size()/1048576.0);
    printf("Sizeof c[] is %f MB\n", (float)sizeof(su3_matrix)*c.size()/1048576.0);
    // check alignment
    for (int i=0; i<3; ++i) {
      site *s = &a[i];
      printf("Address of site %d is 0x%lx, modulo %d = %ld\n", 
             i, (unsigned long)s, ALIGN_N, (unsigned long)s % ALIGN_N);
    }
  }
#endif

  // Setup OpenCL context and devices
  cl::Context context(DEVICE);
  cl::CommandQueue queue(context);
  std::vector<cl::Device> devices;
  context.getInfo(CL_CONTEXT_DEVICES, &devices);
  cl::Device device=devices[0];
  // make the kernel
#if defined VERBOSE && VERBOSE >= 2
  std::cout << "Building Kernel" << std::endl;
#endif
  cl::Program program(context, loadProgram("m_mat_nn.cl"), false);
  if (program.build("-I .") != CL_SUCCESS) {
    std::cout << "ERROR: OpenCL kernel failed to build" << std::endl;
    exit(-1);
  }
  auto k_mat_nn = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer>(program, "k_mat_nn");
#if defined VERBOSE && VERBOSE >= 2
  {
    std::string s, v;
    device.getInfo(CL_DEVICE_VENDOR, &v);
    device.getInfo(CL_DEVICE_NAME, &s);
    std::cout << "Using device: " << v << ": " << s << std::endl;
  }
#endif

  // Declare target storage and copy A and B
  auto d_a = cl::Buffer(context, begin(a), end(a), true);
  auto d_b = cl::Buffer(context, begin(b), end(b), true);
  auto d_c = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(site)*c.size());

#if defined VERBOSE && VERBOSE >= 1
  printf("Number of sites = %d^4\n", LDIM);
  printf("Executing %d iterations\n", ITERATIONS);
#endif
  // benchmark loop
  auto tstart = Clock::now();
  for (int iters=0; iters<ITERATIONS; ++iters) {
    k_mat_nn(cl::EnqueueArgs(queue, cl::NDRange(total_sites)), d_a, d_b, d_c);
  }
  queue.finish(); 
  double ttotal = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tstart).count();
  ttotal /= 1.0e6;
#if defined VERBOSE && VERBOSE >= 1
  printf("Total execution time = %.3f secs\n", ttotal);
#endif

  // each iter of above loop is (3*3)*(12 mult + 10 add) = 108 mult + 90 add = 198 ops
  double tflop = (double)ITERATIONS * total_sites * 4.0 * 198.0;
  printf("Total GFLOP/s = %.3f\n", tflop / ttotal / 1.0e9);
  
  // calculate a checksum
  double sum = 0.0;
  cl::copy(queue, d_c, begin(c), end(c));  // copy data back from device
  #pragma omp parallel for reduction(+:sum)
  for (int i=0;i<total_sites;++i) for(int j=0;j<4;++j) for(int k=0;k<3;++k) for(int l=0;l<3;++l) {
    sum += real(c[i].link[j].e[k][l]);
#ifdef DEBUG
    if (i == 0)
      printf("c[%d][%d]->e[%d][%d]=%f, sum = %f\n",j,i,k,l,real(c[i].link[j].e[k][l]),sum);
#endif
  }
  sum /= (double)total_sites;

  if ( round(sum) != (4.0*sizeof(su3_matrix)/(sizeof(Complx))))
    printf("Checksum FAILED: Sum = %lf\n", sum);

#if defined VERBOSE && VERBOSE >= 2
  // check memory usage
  printf("Total allocation for matrices = %.3f MiB\n", 
         ((float)sizeof(site)*(a.capacity()+c.capacity())+sizeof(su3_matrix)*b.capacity())/1048576.0);
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) == 0)
    printf("Approximate memory usage = %.3f MiB\n", (float)usage.ru_maxrss/1024.0);
#endif

}
