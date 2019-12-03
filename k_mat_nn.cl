#include <lattice.hpp>

//*******************  m_mat_nn.c  (in su3.a) ****************************
//  void mult_su3_nn( su3_matrix *a,*b,*c )
//  matrix multiply, no adjoints 
//  C  <-  A*B	
__kernel void k_mat_nn(
  __global const site*       restrict a,
  __global const su3_matrix* restrict b,
  __global       site*       restrict c,
           const int         total_sites)
{
  int myThread = get_global_id(0);
  int mySite = myThread/36;

  if (mySite < total_sites) {
    int j = (myThread%36)/9;
    int k = (myThread%9)/3;
    int l = myThread%3;
    Complx cc = {0.0, 0.0};
#ifndef LAT_CHECK
    for (int m=0;m<3;m++)
      CMULSUM(a[mySite].link[j].e[k][m], b[j].e[m][l], cc);
    c[mySite].link[j].e[k][l].real = cc.real;
    c[mySite].link[j].e[k][l].imag = cc.imag;
#else
    ;
#endif
  }
}

