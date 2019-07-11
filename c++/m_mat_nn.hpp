#include <omp.h>
#include "lattice.hpp"

//*******************  m_mat_nn.c  (in su3.a) ****************************
//  void mult_su3_nn( su3_matrix *a,*b,*c )
//  matrix multiply, no adjoints 
//  C  <-  A*B	
void k_mat_nn(
            site *aa,
            su3_matrix *bb,
            site *cc )
{
  su3_matrix *a;
  su3_matrix *b;
  su3_matrix *c;

  for (int j=0; j<4; ++j) {
    a = &aa->link[j];
    b = &bb[j];
    c = &cc->link[j];
    for(int k=0;k<3;k++)for(int l=0;l<3;l++){
      c->e[k][l]=Complx(0.0,0.0);
      for(int m=0;m<3;m++) {
        c->e[k][l] += a->e[k][m] * b->e[m][l];
#ifdef DEBUG
        int mysite = omp_get_team_num()*omp_get_team_size(0)+omp_get_thread_num();
        if (mysite == 0 && m == 2)
        printf("a[%d][%d]->e[%d][%d]=%f b[%d][%d]->e[%d][%d]=%f c[%d][%d]->e[%d][%d]=%f\n",
                j,(int)mysite,k,m,real(a->e[k][m]),
                j,(int)mysite,m,l,real(b->e[m][l]),
                j,(int)mysite,k,l,real(c->e[k][l]));
#endif
      }
    }
  }
}

