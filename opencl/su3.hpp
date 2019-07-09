#ifndef _SU3_HPP
#define _SU3_HPP
// Adapted from su3.h in MILC version 7

#include <complex>

typedef struct { std::complex<float> e[3][3]; } fsu3_matrix;
typedef struct { std::complex<float> c[3]; } fsu3_vector;

typedef struct { std::complex<double> e[3][3]; } dsu3_matrix;
typedef struct { std::complex<double> c[3]; } dsu3_vector;

#if (PRECISION==1)
  #define su3_matrix    fsu3_matrix
  #define su3_vector    fsu3_vector
  #define Real          float
  #define Complx        std::complex<float>
#else
  #define su3_matrix    dsu3_matrix
  #define su3_vector    dsu3_vector
  #define Real          double
  #define Complx        std::complex<double>
#endif  // PRECISION

#endif  // _SU3_HPP

