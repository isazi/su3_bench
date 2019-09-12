#include <lattice.hpp>

//*******************  m_mat_nn.c  (in su3.a) ****************************
//  void mult_su3_nn( su3_matrix *a,*b,*c )
//  matrix multiply, no adjoints 
//  C  <-  A*B	
__kernel void k_mat_nn(
  __global const site*       restrict a,
  __global const su3_matrix* restrict b,
  __global       site*       restrict c )
{
  size_t mysite = get_global_id(0);
#ifdef DEBUG
  printf("mysite = %d\n", (int)mysite);
#endif

  for (int j=0; j<4; ++j) {
    for (int k=0;k<3;k++) {
      for (int l=0;l<3;l++){
        c[mysite].link[j].e[k][l].real=0.0;
        c[mysite].link[j].e[k][l].imag=0.0;
        for (int m=0;m<3;m++) {
          CMULSUM(a[mysite].link[j].e[k][m], b[j].e[m][l], c[mysite].link[j].e[k][l]);
#ifdef DEBUG
          if (mysite==0 && m==2)
          printf("a[%d][%d]->e[%d][%d]=%f b[%d][%d]->e[%d][%d]=%f c[%d][%d]->e[%d][%d]=%f\n",
                  j,(int)mysite,k,m,a[mysite].link[j].e[k][m].real,
                  j,(int)mysite,m,l,b[j].e[m][l].real,
                  j,(int)mysite,k,l,c[mysite].link[j].e[k][l].real);
#endif
        }
      }
    }
  }
}

