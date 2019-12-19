## su3_bench: Lattice QCD SU(3) matrix-matrix multiply microbenchmark  
The purpose of this microbenchmark is to provide a means to explore different programming methodologies using a simple, but not trivial, mathematical kernel. The kernel is based on the mult_su3_nn() SU(3) matrix-matrix multiply routine in the MILC Lattice Quantum Chromodynamics(LQCD) code. Matrix-matrix (and matrix-vector) SU(3) operations is a fundamental building blocks of LQCD applications. Most LQCD applications use custom implementations of these kernels, and they are usually written in machine specific languages and/or  intrinsics. 

### Design
The code is written in standard C and C++. The main driver routine is used for all programming model implementations, with programming model specific implementations self contained in respective C++ include files. Programming methods implemented to date include: OpenCL, OpenMP, OpenACC, SYCL, and Cuda.

The code is the documentation. It's simple enough, so dive in.

Various makefiles are also included, one for each of the respective compile environments I've tried so far.

### Usage
bench_xxx.exe -h

### Contact info
Doug Doerfler
dwdoerf@lbl.gov
