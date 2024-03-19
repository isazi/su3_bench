#ifdef kernel_tuner
#include <lattice.hpp>
#endif

//*******************  m_mat_nn.c  (in su3.a) ****************************
//  void mult_su3_nn( su3_matrix *a,*b,*c )
//  matrix multiply, no adjoints
//  C  <-  A*B
__global__ void k_mat_nn(
    const site*       __restrict__ a,
    const su3_matrix* __restrict__ b,
    site*       __restrict__ c,
    int               total_sites)
{
  int myThread = blockDim.x * blockIdx.x + threadIdx.x;
  int mySite = myThread/36;

  if (mySite < total_sites) {
    int j = (myThread%36)/9;
    int k = (myThread%9)/3;
    int l = myThread%3;
    Complx cc = {0.0,0.0};
    for (int m=0;m<3;m++)
#ifdef MILC_COMPLEX
      CMULSUM(a[mySite].link[j].e[k][m], b[j].e[m][l], cc);
      //c[mySite].link[j].e[k][l].real = cc.real;
      //c[mySite].link[j].e[k][l].imag = cc.imag;
#else
      cc += a[mySite].link[j].e[k][m] * b[j].e[m][l];
      //c[mySite].link[j].e[k][l] = cc;
#endif
    c[mySite].link[j].e[k][l] = cc;
  }
}