# Lattice QCD su3 microbenchmarks

Kernels are based on the MILC code and wrapped with drivers to form microbenchmarks.

The directory "c99" is an OpenMP target offload implementation using C99 complex datatypes. This is the initial implementation.

The directory "c++" is an OpenMP target offload implementation using C++11 complex, and other C++ features.

The directory "opencl" is an OpenCL version using C++ OpenCL 1.2 bindings. 
Since OpenCL 1.2 doesn't support complex data types, MILC complex data structures are used and math routines are used.

The directory "sycl" is an immature SYCL version. Still needs features to enable tuning. 

The directory "cuda", is an NVIDIA Cuda implementation.

At this point in time, the OpenCL implementation is the most mature and has been tested on NVIDIA GPUs, AMD GCN GPUs and Intel CPUs and GPUs.


