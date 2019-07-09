#include <su3.h>

//*******************  m_mat_nn.c  (in su3.a) ****************************
//  void mult_su3_nn( su3_matrix *a,*b,*c )
//  matrix multiply, no adjoints 
//  C  <-  A*B	
__kernel void k_mat_nn(
            __global su3_matrix *aa,
            __global su3_matrix *bb,
            __global su3_matrix *cc,
            __global int* params)
{
  __global su3_matrix *a;
  __global su3_matrix *b;
  __global su3_matrix *c;

  size_t site = get_global_id(0);
  size_t size = get_global_size(0);
  printf("site = %d\n", site);

  for (int j=0; j<4; ++j) {
    a = aa+j*size+site;
    b = bb+j*size+site;
    c = cc+j*size+site;
    for(int k=0;k<3;k++)for(int l=0;l<3;l++){
      c->e[k][l].real=0.0;
      c->e[k][l].imag=0.0;
      for(int m=0;m<3;m++) {
        CMULSUM(a->e[k][m], b->e[m][l], c->e[k][l]);
if (site == 0)
printf("a[%d][%d]->e[%d][%d]=%f b[%d][%d]->e[%d][%d]=%f c[%d][%d]->e[%d][%d]=%f\n",j,(int)site,k,m,a->e[k][m],j,(int)site,m,l,b->e[m][l],j,(int)site,k,l,c->e[k][l]);
      }
    }
  }

  if (site == 0) printf("global size = %d\n", size);
  params[0]=(int)size;
}

