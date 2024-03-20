#ifdef kernel_tuner
#pragma tuner initialize
#include <lattice.hpp>
#pragma tuner stop
#else kernel_tuner
#define NTHREADS 128
#define COLLAPSE_FACTOR 3
#endif

// OpenACC implementation
double su3_mat_nn(std::vector<site> &a, std::vector<su3_matrix> &b, std::vector<site> &c,
		  size_t total_sites, size_t iterations, size_t threads_per_team, int use_device, Profile* profile)
{
  site *d_a, *d_c;
  su3_matrix *d_b;
  size_t len_a, len_b, len_c;
  d_a = a.data(); len_a = a.size();
  d_b = b.data(); len_b = b.size();
  d_c = c.data(); len_c = c.size();

  auto tprofiling = Clock::now();

  // Move A, B and C vectors to the device
  #pragma acc enter data copyin(d_a[0:len_a], d_b[0:len_b], d_c[0:len_c])

  profile->host_to_device_time = (std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tprofiling).count())/1.0e6;

  // benchmark loop
  auto tstart = Clock::now();
  tprofiling = tstart;
  for (int iters=0; iters<iterations+warmups; ++iters) {
    if (iters == warmups) {
      tstart = Clock::now();
      tprofiling = tstart;
    }
    #pragma tuner start k_mat_nn d_a(site*:LEN_A) d_b(su3_matrix*:LEN_B) d_c(site*:LEN_C) total_sites(int:SITES)
    #pragma acc parallel vector_length(NTHREADS) present(d_a[0:len_a], d_b[0:len_b], d_c[0:len_c])
    #pragma acc loop gang
    for(int i=0;i<total_sites;++i) {
      #pragma acc loop worker vector collapse(COLLAPSE_FACTOR)
      for (int j=0; j<4; ++j) {
        for(int k=0;k<3;k++) {
          for(int l=0;l<3;l++){
	    Complx cc = {0.0, 0.0};
#ifndef MILC_COMPLEX
            #pragma acc loop seq
            for(int m=0;m<3;m++) {
               cc += d_a[i].link[j].e[k][m] * d_b[j].e[m][l];
            }
            d_c[i].link[j].e[k][l] = cc;
#else
            #pragma acc loop seq
            for(int m=0;m<3;m++) {
               CMULSUM(d_a[i].link[j].e[k][m], d_b[j].e[m][l], cc);
            }
            d_c[i].link[j].e[k][l].real = cc.real;
            d_c[i].link[j].e[k][l].imag = cc.imag;
#endif
          }
        }
      }
    }
    #pragma tuner stop
  }
  double ttotal = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tstart).count();
  profile->kernel_time = (std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tprofiling).count())/1.0e6;

  // move the result back
  tprofiling = Clock::now();
  #pragma acc exit data copyout(d_c[0:len_c])
  profile->device_to_host_time = (std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tprofiling).count())/1.0e6;

  return (ttotal /= 1.0e6);
}
