// SYCL implementation
#include <CL/sycl.hpp>

#define THREADS_PER_SITE 36

// Sycl requires that kernels be named
class k_mat_nn;

double su3_mat_nn(const std::vector<site> &a, const std::vector<su3_matrix> &b, std::vector<site> &c, 
              const size_t total_sites, const size_t iterations, size_t wgsize, const int target)
{ 
  // build a list of devices
  std::vector<cl::sycl::platform> platforms = cl::sycl::platform::get_platforms();
  std::vector<cl::sycl::device> devices;
  for (int i=0, d=0; i < platforms.size(); ++i) {
    std::vector<cl::sycl::device> pdevices = platforms[i].get_devices();
    for (int j=0; j < pdevices.size(); ++j, ++d) {
      devices.insert(devices.end(), pdevices[j]);
      if (verbose >= 3)
        std::cout << "Appending device " << d << ": " << pdevices[j].get_info<cl::sycl::info::device::name>() << std::endl;
    }
  }

  // Create a SYCL queue
  cl::sycl::device target_device;
  if (target < 0) {
    cl::sycl::default_selector selector;
    target_device = selector.select_device();
  } 
  else if (target < devices.size()) {
    target_device = devices[target];
  }
  else {
    std::cout << "Invalid device specified: " << target << std::endl;
    exit(1);
  }
  cl::sycl::queue queue(target_device);
  if (verbose >= 2)
    std::cout << "Using device: " << queue.get_device().get_info<cl::sycl::info::device::name>() << "\n";

#ifndef HIPSYCL
  // Pre-build the kernel
  auto build_start = Clock::now();
  cl::sycl::program program = cl::sycl::program(queue.get_context());
  program.build_with_kernel_type<k_mat_nn>();
  double build_time = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-build_start).count();
  if (verbose >= 3)
    std::cout << "Time to build kernel = " << build_time/1.0e6 << " secs\n";
#endif

  if (wgsize < THREADS_PER_SITE)
    wgsize = THREADS_PER_SITE;

  size_t total_wi = total_sites * wgsize;
  if (verbose >= 1) {
    std::cout << "Setting number of work items " << total_wi << std::endl;
    std::cout << "Setting workgroup size to " << wgsize << std::endl;
  }

  if (verbose >= 3) {
    std::cout << "max compute units = " 
       << queue.get_device().get_info<cl::sycl::info::device::max_compute_units>() << "\n";
    std::cout << "max workgroup size = " 
       << queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>() << "\n";
  }

  double ttotal;

  {
  // wrap arrays in SYCL buffers, suppling global memory pointer implicitly copies the data to the device when needed
  cl::sycl::buffer<site, 1>       a_buf {a.data(), cl::sycl::range<1> {total_sites}};
  cl::sycl::buffer<su3_matrix, 1> b_buf {b.data(), cl::sycl::range<1> {4}};
  // The C array  will never be copy from the Host to the Device. Indeed the first aceesor used it a discard_write
  // The Copy Deive -> Host will occur when the destructor will be called (at the end of the scope)
  cl::sycl::buffer<site, 1>       c_buf {c.data(), cl::sycl::range<1> {total_sites}};

  // benchmark loop
  auto tstart = Clock::now();
  for (int iters=0; iters<iterations+warmups; ++iters) {
    if (iters == warmups)
      tstart = Clock::now();

    // create a command_group to issue commands
    queue.submit([&](cl::sycl::handler& cgh) {
      // request access to the host buffers
      auto d_a = a_buf.get_access<cl::sycl::access::mode::read>(cgh);
      auto d_b = b_buf.get_access<cl::sycl::access::mode::read>(cgh);
      auto d_c = c_buf.get_access<cl::sycl::access::mode::discard_write>(cgh);

      // Lambda function defines the kernel scope

#ifndef HIPSYCL
      cgh.parallel_for<class k_mat_nn>(program.get_kernel<k_mat_nn>(),
#else
      cgh.parallel_for<class k_mat_nn>(
#endif
      cl::sycl::nd_range<1> {total_wi, wgsize}, [=](cl::sycl::nd_item<1> item) {
        size_t myThread = item.get_global_id(0);
        size_t mySite = myThread/36;
        if (mySite < total_sites) {
          int j = (myThread%36)/9;
          int k = (myThread%9)/3;
          int l = myThread%3;
          Complx cc = {0.0, 0.0};
#ifndef LAT_CHECK
          for (int m=0;m<3;m++) {
    #ifndef MILC_COMPLEX
            cc += d_a[mySite].link[j].e[k][m] * d_b[j].e[m][l];
          }
          d_c[mySite].link[j].e[k][l] = cc;
    #else
            CMULSUM(d_a[mySite].link[j].e[k][m], d_b[j].e[m][l], cc);
          }
          d_c[mySite].link[j].e[k][l].real = cc.real;
          d_c[mySite].link[j].e[k][l].imag = cc.imag;
    #endif
#endif
        }
      }); // end of the kernel lambda function
    });   // end of command group
  } // end of iteration loop
  queue.wait();

  ttotal = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tstart).count();

  }

  return (ttotal /= 1.0e6);
} // end of SYCL block

