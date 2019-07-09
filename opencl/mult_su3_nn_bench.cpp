#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <math.h>
#include <omp.h>

#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include <CL/cl.hpp>
#ifndef DEVICE
#  define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

#include "su3.hpp"
#include "lattice.hpp"

#ifndef ITERATIONS
#  define ITERATIONS 100
#endif

// Lattice size = LDIM^4
#ifndef LDIM
#  define LDIM 7
#endif

#undef DEBUG
#define VERBOSE

inline std::string loadProgram(std::string input)
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

  // Setup OpenCL context and devices
  cl::Context context(DEVICE);
  cl::CommandQueue queue(context);
  std::vector<cl::Device> devices;
  context.getInfo(CL_CONTEXT_DEVICES, &devices);
  cl::Device device=devices[0];
  // build the kernel
#ifdef VERBOSE
  std::cout << "Building Kernel" << std::endl;
#endif
  cl::Program program(context, loadProgram("m_mat_nn.cl"), false);
  if (program.build("-I .") != CL_SUCCESS) {
    fprintf(stderr, "ERROR: OpenCL kernel failed to build\n");
    exit(-1);
  }
  auto k_mat_nn = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer>(program, "k_mat_nn");
#ifdef VERBOSE
  {
    std::string s, v;
    device.getInfo(CL_DEVICE_VENDOR, &v);
    device.getInfo(CL_DEVICE_NAME, &s);
    std::cout << "Using device: " << v << ": " << s << std::endl;
  }
#endif

  cl::Buffer d_lattice, d_b, d_c;
//  d_lattice = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(site) * total_sites);
//  d_b = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(b));
//  d_c = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(su3_matrix) * 4 * total_sites);
  d_lattice = cl::Buffer(context, begin(lattice), end(lattice), true);
  d_b = cl::Buffer(context, begin(b), end(b), true);
  d_c = cl::Buffer(context, begin(c), end(c), true);

  // move lattice and b matrix to the device
//  d_lattice = cl::Buffer(context, begin(lattice), end(lattice), true);
//  cl::copy(queue, begin(lattice), end(lattice), d_lattice);
//  d_b = cl::Buffer(context, begin(b), end(b), true);
//  cl::copy(queue, begin(b), end(b), d_b);

  printf("Number of sites = %d^4\n", LDIM);
  printf("Executing %d iterations\n", ITERATIONS);
  // benchmark loop
  double tstart = omp_get_wtime();
  for (int iters=0; iters<ITERATIONS; ++iters) {
//    #pragma omp parallel for
//    for(int i=0;i<total_sites;++i) for(int j=0;j<4;++j)
//        mult_su3_nn( &lattice[i].link[j], &b[j], &c.at(j*total_sites+i));
    k_mat_nn(cl::EnqueueArgs(queue, cl::NDRange(total_sites)), d_lattice, d_b, d_c);
    queue.finish(); 
  }
  double ttotal = omp_get_wtime() - tstart;
  printf("Total execution time = %.2f secs\n", ttotal);

  // each iter of above loop is (3*3)*(12 mult + 10 add) = 108 mult + 90 add = 198 ops
  double tflop = (double)ITERATIONS * total_sites * 4.0 * 198.0;
  printf("Total GFLOP/s = %f\n", tflop / ttotal / 1.0e9);
  
  // calculate a checksum
  double sum = 0.0;
  cl::copy(queue, d_c, begin(c), end(c));  // copy data back from device
  #pragma omp parallel for reduction(+:sum)
  for (int i=0;i<total_sites;++i) for(int j=0;j<4;++j) for(int k=0;k<3;++k) for(int l=0;l<3;++l) {
    sum += real(c.at(j*total_sites+i).e[k][l]);
#ifdef DEBUG
    printf("c[%d][%d]->e[%d][%d]=%f, sum = %f\n",j,i,k,l,real(c.at(j*total_sites+i).e[k][l]),sum);
#endif
  }
  sum /= (double)total_sites;

  if ( round(sum) != (4.0*sizeof(su3_matrix)/(sizeof(Complx))))
    printf("Checksum FAILED: Sum = %lf\n", sum);

  // check memory usage
  printf("Total allocation for matrices = %.3f MB\n", total_sites*(sizeof(site)+4*sizeof(su3_matrix))/1048576.0);
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) == 0)
    printf("Approximate memory usage = %.3f MB\n", (float)usage.ru_maxrss/1024.0);

}

