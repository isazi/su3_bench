// KMM+CUDA implementation
#include <kmm/kmm.hpp>
#include "mat_nn_cuda_kernel_kmm.cu"


#define THREADS_PER_SITE 36

double su3_mat_nn(std::vector<site> &a, std::vector<su3_matrix> &b, std::vector<site> &c,
                  size_t total_sites, size_t iterations, size_t threadsPerBlock, int use_device, Profile *profile)
{
  using namespace kmm::placeholders;
  int blocksPerGrid;
  int chunk_size = total_sites / 10;

  if (threadsPerBlock == 0) {
    threadsPerBlock = THREADS_PER_SITE;
  }

  // initialize KMM runtime
  auto rt = kmm::make_runtime();
  auto domain = kmm::TileDomain(total_sites, chunk_size);
  auto _x = kmm::Axis(0);

  auto tprofiling = Clock::now();

  // Declare target storage and copy A and B
  auto d_a = rt.allocate(a);
  auto d_b = rt.allocate(b);
  auto d_c = rt.allocate(c);

  d_a.copy_from(a.data(), a.size());
  d_b.copy_from(b.data(), b.size());
  rt.synchronize();
  profile->host_to_device_time = (std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tprofiling).count())/1.0e6;

  double sitesPerBlock = (double)threadsPerBlock / THREADS_PER_SITE;
  blocksPerGrid = total_sites/sitesPerBlock + 0.999999;

  if (verbose >= 1) {
    printf("Number of blocks set to %d\n", blocksPerGrid);
    printf("Threads per block set to %d\n", threadsPerBlock);
  }

  // benchmark loop
  auto tstart = Clock::now();
  tprofiling = tstart;

  for (int iters=0; iters<iterations+warmups; ++iters) {
    if (iters == warmups) {
      rt.synchronize();
      tstart = Clock::now();
      tprofiling = tstart;
    }
    rt.parallel_submit(
        domain,
        kmm::GPUKernel(k_mat_nn, threadsPerBlock),
        _x,
        d_a[_x],
        d_b[_],
        write(d_c[_x]),
        total_sites
    );
  }
  rt.synchronize();
  profile->kernel_time = (std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tprofiling).count())/1.0e6;
  double ttotal = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tstart).count();

  // copy data back from device
  tprofiling = Clock::now();
  d_c.copy_to(c.data(), c.size());
  rt.synchronize();
  profile->device_to_host_time= (std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tprofiling).count())/1.0e6;

  return (ttotal /= 1.0e6);
}
