#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <math.h>
#include <omp.h>

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
typedef std::chrono::system_clock Clock;

#include "lattice.hpp"

#ifndef ITERATIONS
#  define ITERATIONS 100
#endif
#ifndef LDIM
#  define LDIM 7  // Lattice size = LDIM^4
#endif
#ifndef VERBOSE
#  define VERBOSE 1  // valid values 0, 1 or 2
#endif

int main(int argc, char *argv[])
{
  // allocate and initialize the working lattices and B link matrix
  int total_sites = LDIM*LDIM*LDIM*LDIM;
  // A
  std::vector<site> a(total_sites);
  make_lattice(&a[0], LDIM);
  // B
  std::vector<su3_matrix> b(4);
  init_link(&b[0], Complx((1.0/3.0), 0.0));
  // C
  std::vector<site> c(total_sites);

#if defined VERBOSE && VERBOSE >= 1
  printf("Number of sites = %d^4\n", LDIM);
  printf("Executing %d iterations\n", ITERATIONS);
#endif

  site *p_a, *p_c;
  su3_matrix *p_b;
  size_t len_a, len_b, len_c;
  p_a = a.data(); len_a = a.size();
  p_b = b.data(); len_b = b.size();
  p_c = c.data(); len_c = c.size();
 
  // benchmark loop
#define WORKAROUND
#ifdef OMP_TARGET
  #pragma omp target enter data map(to: p_a[0:len_a], p_b[0:len_b], p_c[0:len_c])
  auto tstart = Clock::now();
  for (int iters=0; iters<ITERATIONS; ++iters) {
#ifndef WORKAROUND
    //#pragma omp target teams distribute
    #pragma omp target teams distribute thread_limit(32)
    for(int i=0;i<total_sites;++i) {
      int total_teams = omp_get_num_teams();
      int team_id = omp_get_team_num();
      int sites_per_team = (total_sites + total_teams - 1) / total_teams;
if (iters == 0 && i == 0 && team_id == 0) printf("total teams = %d, sites/team = %d\n", total_teams, sites_per_team);
      #pragma omp parallel for collapse(3)
#else
    //#pragma omp target teams
    #pragma omp target teams num_teams(4096)
    {
      int total_teams = omp_get_num_teams();
      int team_id = omp_get_team_num();
      int sites_per_team = (total_sites + total_teams - 1) / total_teams;
      int istart = team_id * sites_per_team;
      if (istart > total_sites) istart = total_sites;
      int iend = istart + sites_per_team;
      if (iend > total_sites) iend = total_sites;

      #pragma omp parallel
      for (int i = istart; i < iend; ++i)
      #pragma omp for collapse(3)
#endif
#else
  auto tstart = Clock::now();
  for (int iters=0; iters<ITERATIONS; ++iters) {
    #pragma omp parallel for
    for(int i=0;i<total_sites;++i) {
#endif
      for (int j=0; j<4; ++j) {
        for(int k=0;k<3;k++) {
          for(int l=0;l<3;l++){
            Complx cc;
            for(int m=0;m<3;m++) {
              cc += p_a[i].link[j].e[k][m] * p_b[j].e[m][l];
            }
            p_c[i].link[j].e[k][l] = cc;
          }
        }
      }
    }
  }
  double ttotal = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tstart).count() / 1.0e6;
#if defined VERBOSE && VERBOSE >= 1
  printf("Total execution time = %.3f secs\n", ttotal);
#endif

#ifdef OMP_TARGET
  #pragma omp target exit data map(from: p_c[0:len_c])
#endif

  // each iter of above loop is (3*3)*(12 mult + 10 add) = 108 mult + 90 add = 198 ops
  double tflop = (double)ITERATIONS * total_sites * 4.0 * 198.0;
  printf("Total GFLOP/s = %.3f\n", tflop / ttotal / 1.0e9);
  
  // calculate a checksum
  double sum = 0.0;
  #pragma omp parallel for reduction(+:sum)
  for (int i=0;i<total_sites;++i) for(int j=0;j<4;++j) for(int k=0;k<3;++k) for(int l=0;l<3;++l) {
    sum += real(c[i].link[j].e[k][l]);
#ifdef DEBUG
    if (i == 0)
      printf("c[%d][%d]->e[%d][%d]=%f, sum = %f\n",j,i,k,l,real(c[i].link[j].e[k][l]),sum);
#endif
  }
  sum /= (double)total_sites;

  if ( round(sum) != (4.0*sizeof(su3_matrix)/(sizeof(Complx))))
    printf("Checksum FAILED: Sum = %lf\n", sum);

#if defined VERBOSE && VERBOSE >= 2
  // check memory usage
  printf("Total allocation for matrices = %.3f MiB\n", 
         ((float)sizeof(site)*(a.capacity()+c.capacity())+sizeof(su3_matrix)*b.capacity())/1048576.0);
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) == 0)
    printf("Approximate memory usage = %.3f MiB\n", (float)usage.ru_maxrss/1024.0);
#endif

}

