// HIP implementation
#include <hip/hip_runtime.h>

#define CUCHECK(err, s) \
  if (err != hipSuccess) { \
        printf("%s (error code %d:%s)!\n", s, err, hipGetErrorString(err)); \
        exit(EXIT_FAILURE); \
  }

#define THREADS_PER_SITE 36

#include "mat_nn_hip_kernel.cpp"

double su3_mat_nn(std::vector<site> &a, std::vector<su3_matrix> &b, std::vector<site> &c, 
		  size_t total_sites, size_t iterations, size_t threadsPerBlock, int use_device, Profile* profile)
{
  int blocksPerGrid;
  int size_a = sizeof(site) * total_sites;
  int size_b = sizeof(su3_matrix) * 4;
  int size_c = sizeof(site) * total_sites;

  if (threadsPerBlock == 0)
    threadsPerBlock = THREADS_PER_SITE;

  // Device initialization
  int deviceCount;
  hipGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    printf("ERROR: No devices found\n");
    exit(1);
  }

  struct hipDeviceProp_t device_prop;
  if (verbose >= 3) {
    for (int i = 0; i < deviceCount; ++i) {
      hipGetDeviceProperties(&device_prop, i);
      printf("Located device %d: %s\n", i, device_prop.name);
    }
  }
  if (use_device == -1)
    use_device = 0;
  else if (use_device >= deviceCount) {
    printf("ERROR: Device %d not found\n", use_device);
    exit(1);
  }
  hipSetDevice(use_device);
  if (verbose >= 2) {
    hipGetDeviceProperties(&device_prop, use_device);
    printf("Using device %d: %s\n", use_device, device_prop.name);
  }

  auto tprofiling = Clock::now();

  // Declare target storage and copy A and B
  hipError_t cuErr;
  site *d_a, *d_c;
  su3_matrix *d_b;
  cuErr = hipMalloc((void **)&d_a, size_a);
  CUCHECK(cuErr, "Unable to allocate array d_a");
  cuErr = hipMalloc((void **)&d_b, size_b);
  CUCHECK(cuErr, "Unable to allocate array d_b");
  cuErr = hipMalloc((void **)&d_c, size_c);
  CUCHECK(cuErr, "Unable to allocate array d_c");
  hipMemcpy(d_a, a.data(), size_a, hipMemcpyHostToDevice);
  hipMemcpy(d_b, b.data(), size_b, hipMemcpyHostToDevice);

  profile->host_to_device_time = (std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tprofiling).count())/1.0e6;

  double sitesPerBlock = (double)threadsPerBlock / THREADS_PER_SITE;
  blocksPerGrid = total_sites/sitesPerBlock + 0.999999;

  if (verbose >= 1) {
    printf("Number of blocks set to %d\n", blocksPerGrid);
    printf("Threads per block set to %zu\n", threadsPerBlock);
  }

  // benchmark loop
  auto tstart = Clock::now();
  tprofiling = tstart;

  for (int iters=0; iters<iterations+warmups; ++iters) {
    if (iters == warmups) {
      hipDeviceSynchronize();
      tstart = Clock::now();
      tprofiling = tstart;
    }
    hipLaunchKernelGGL(k_mat_nn, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, d_a, d_b, d_c, total_sites);
  }
  hipDeviceSynchronize();
  profile->kernel_time = (std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tprofiling).count())/1.0e6;
  double ttotal = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tstart).count();
  CUCHECK(hipGetLastError(), "k_mat_nn kernel Failed");

  // copy data back from device
  tprofiling = Clock::now();
  hipMemcpy(c.data(), d_c, size_c, hipMemcpyDeviceToHost);
  profile->device_to_host_time= (std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tprofiling).count())/1.0e6;

  // Deallocate
  hipFree(d_a);
  hipFree(d_b);
  hipFree(d_c);

  return (ttotal /= 1.0e6);
}
