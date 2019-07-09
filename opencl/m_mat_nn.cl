#include <su3.h>
#include <lattice.h>

//*******************  m_mat_nn.c  (in su3.a) ****************************
//  void mult_su3_nn( su3_matrix *a,*b,*c )
//  matrix multiply, no adjoints 
//  C  <-  A*B	
__kernel void k_mat_nn(
            __global site *aa,
            __global su3_matrix *bb,
            __global su3_matrix *cc )
{
  __global su3_matrix *a;
  __global su3_matrix *b;
  __global su3_matrix *c;

  size_t mysite = get_global_id(0);
  size_t size = get_global_size(0);
//  printf("mysite = %d\n", mysite);

  for (int j=0; j<4; ++j) {
    //a = aa+j*size+mysite;
    a = &aa[mysite].link[j];
    b = &bb[j];
    c = cc+j*size+mysite;
    for(int k=0;k<3;k++)for(int l=0;l<3;l++){
      c->e[k][l].real=0.0;
      c->e[k][l].imag=0.0;
      for(int m=0;m<3;m++) {
        CMULSUM(a->e[k][m], b->e[m][l], c->e[k][l]);
//if (mysite==0 && m==2)
//printf("a[%d][%d]->e[%d][%d]=%f b[%d][%d]->e[%d][%d]=%f c[%d][%d]->e[%d][%d]=%f\n",j,(int)mysite,k,m,a->e[k][m],j,(int)mysite,m,l,b->e[m][l],j,(int)mysite,k,l,c->e[k][l]);
      }
    }
  }

}

