// KMM+CUDA implementation
#include <kmm/array.hpp>
#include <kmm/cuda/cuda.hpp>
#include <kmm/runtime_handle.hpp>
#include "mat_nn_cuda_kernel.cu"


#define THREADS_PER_SITE 36

double su3_mat_nn(std::vector<site> &a, std::vector<su3_matrix> &b, std::vector<site> &c,
                  size_t total_sites, size_t iterations, size_t threadsPerBlock, int use_device, Profile *profile)
{
  int blocksPerGrid;

  if (threadsPerBlock == 0) {
    threadsPerBlock = THREADS_PER_SITE;
  }

  // initialize KMM manager
  auto manager = kmm::build_runtime();

  // declare target storage and copy A and B
  auto d_a = manager.allocate(a);
  auto d_b = manager.allocate(b);
  auto d_c = kmm::Array<site>(total_sites);

  double sitesPerBlock = (double)threadsPerBlock / THREADS_PER_SITE;
  blocksPerGrid = total_sites/sitesPerBlock + 0.999999;

  if (verbose >= 1) {
    printf("Number of blocks set to %d\n", blocksPerGrid);
    printf("Threads per block set to %d\n", threadsPerBlock);
  }

  // benchmark loop
  auto tstart = Clock::now();

  for (int iters=0; iters<iterations+warmups; ++iters) {
    if (iters == warmups) {
      manager.synchronize();
      tstart = Clock::now();
    }
    manager.submit(kmm::CudaKernel(blocksPerGrid, threadsPerBlock), k_mat_nn, d_a, d_b, write(d_c), total_sites);
  }
  manager.synchronize();
  double ttotal = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tstart).count();

  // copy data back from device
  d_c.read(c.data(), c.size());

  return (ttotal /= 1.0e6);
}
