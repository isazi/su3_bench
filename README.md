# Lattice QCD su3 microbenchmarks

Kernels are based on the MILC code and wrapped with drivers to form microbenchmarks.

The directory "c99" is an implementation using C99 complex datatypes. This is the initial implementation.

The directory "c++" is a rewrite using C++11 complex, and other C++ features.

The directory "opencl" is an OpenCL version using C++ OpenCL 1.2 bindings. 
Since OpenCL is basically C99, complex data types are not supported so MILC complex data structures were used.

