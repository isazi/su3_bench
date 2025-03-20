// KMM+CUDA implementation
#include <kmm/kmm.hpp>
#include "mat_nn_cuda_kernel_kmm.cu"


#define THREADS_PER_SITE 36

double su3_mat_nn(std::vector<site> &a, std::vector<su3_matrix> &b, std::vector<site> &c,
                  size_t total_sites, size_t iterations, size_t threadsPerBlock, int use_device, Profile *profile)
{
  int blocksPerGrid;
  int chunk_size = total_sites / 10;

  if (threadsPerBlock == 0) {
    threadsPerBlock = THREADS_PER_SITE;
  }

  // initialize KMM runtime
  auto rt = kmm::make_runtime();

  auto d_a = kmm::Array<site> {total_sites};
  auto d_b = kmm::Array<su3_matrix> {4};
  auto d_c = kmm::Array<site> {total_sites};
  auto domain = kmm::TileDomain(total_sites, chunk_size);

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
      rt.synchronize();
      tstart = Clock::now();
    }
    rt.submit(
        domain,
        kmm::GPUKernel(k_mat_nn, threadsPerBlock),
        d_a,
        d_b,
        write(d_c),
        total_sites
    );
  }
  rt.synchronize();
  double ttotal = (std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tstart).count()) / 1.0e6;

  // copy data back from device
  d_c.copy_to(c);

  return (ttotal /= 1.0e6);
}
