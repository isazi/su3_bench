// OpenCL implementation
#include <CL/cl.hpp>
#include <fstream>
#include <string>

#ifndef DEVICE
#  define DEVICE CL_DEVICE_TYPE_ALL
#endif

#define THREADS_PER_SITE 36

// loads an opencl kernel source file into a string
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


double su3_mat_nn(std::vector<site> &a, std::vector<su3_matrix> &b, std::vector<site> &c, 
              size_t total_sites, size_t iterations, size_t wgsize, int use_device)
{ 

  if (wgsize < THREADS_PER_SITE)
    wgsize = THREADS_PER_SITE;


  // Setup OpenCL context and devices
  std::vector<cl::Device> devices;
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  for (int i=0, d=0; i< platforms.size(); ++i) {
    std::vector<cl::Device> pdevices;
    platforms[i].getDevices(DEVICE, &pdevices);
    for (int j = 0; j < pdevices.size(); ++j, ++d) {
      devices.insert(devices.end(), pdevices[j]);
      if (verbose >= 3) {
        std::string s;
        pdevices[j].getInfo(CL_DEVICE_NAME, &s);
        std::cout << "Appending device " << d << ": " << s << std::endl;
      }
    }
  }
  if (use_device >= devices.size()) {
    std::cout << "ERROR: Device " << use_device << " not found\n" << std::endl;
    exit(1);
  }
  cl::Device device=devices[use_device];

  cl::Context context(device);
  cl::CommandQueue queue(context);

  // make the kernel
  char build_args[80];
  sprintf(build_args, "-I. -DPRECISION=%d -DUSE_OPENCL", PRECISION);
  if (verbose >= 2)
    std::cout << "Building Kernel with: " << build_args << std::endl;
  cl::Program program(context, loadProgram("k_mat_nn.cl"), false);
  if (program.build(build_args) != CL_SUCCESS) {
    std::cout << "ERROR: OpenCL kernel failed to build" << std::endl;
    exit(1);
  }
  auto k_mat_nn = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int>(program, "k_mat_nn");
  if (verbose >= 2) {
    std::string s;
    device.getInfo(CL_DEVICE_NAME, &s);
    std::cout << "Using device: " << s << std::endl;
  }

  // Declare target storage and copy A and B
  auto d_a = cl::Buffer(context, begin(a), end(a), true);
  auto d_b = cl::Buffer(context, begin(b), end(b), true);
  auto d_c = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(site)*c.size());

  if (verbose >= 1)
    std::cout << "Setting workgroup size to " << wgsize << std::endl;

  // benchmark loop
  size_t total_wi = total_sites * wgsize;
  auto tstart = Clock::now();
  for (int iters=0; iters<iterations; ++iters) {
    k_mat_nn(cl::EnqueueArgs(queue, cl::NDRange(total_wi), cl::NDRange(wgsize)), d_a, d_b, d_c, total_sites);
  }
  queue.finish(); 
  double ttotal = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tstart).count();

  // copy data back from device
  cl::copy(queue, d_c, begin(c), end(c));

  return (ttotal /= 1.0e6);
}

