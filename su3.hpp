#ifndef _SU3_HPP
#define _SU3_HPP
// Adapted from su3.h in MILC version 7

#ifdef USE_THRUST
  #include <thrust/complex.h>
  struct fsu3_matrix { thrust::complex<float> e[3][3]; } ;
  struct fsu3_vector { thrust::complex<float> c[3]; } ;
  struct dsu3_matrix { thrust::complex<double> e[3][3]; } ;
  struct dsu3_vector { thrust::complex<double> c[3]; } ;
#else
  #include <complex>
  struct fsu3_matrix { std::complex<float> e[3][3]; } ;
  struct fsu3_vector { std::complex<float> c[3]; } ;
  struct dsu3_matrix { std::complex<double> e[3][3]; } ;
  struct dsu3_vector { std::complex<double> c[3]; } ;
#endif


#if (PRECISION==1)
  #define su3_matrix    fsu3_matrix
  #define su3_vector    fsu3_vector
  #define Real          float
#ifdef USE_THRUST
  #define Complx        thrust::complex<float>
#else
  #define Complx        std::complex<float>
#endif
#else
  #define su3_matrix    dsu3_matrix
  #define su3_vector    dsu3_vector
  #define Real          double
#ifdef USE_THRUST
  #define Complx        thrust::complex<double>
#else
  #define Complx        std::complex<double>
#endif
#endif  // PRECISION

#endif  // _SU3_HPP

