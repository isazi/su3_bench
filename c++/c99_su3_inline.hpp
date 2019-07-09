
//*******************  m_mat_nn.c  (in su3.a) ****************************
//  void mult_su3_nn( su3_matrix *a,*b,*c )
//  matrix multiply, no adjoints 
//  C  <-  A*B	
static inline void mult_su3_nn( su3_matrix *a, su3_matrix *b, su3_matrix *c ){
  for(int k=0;k<3;k++)for(int l=0;l<3;l++){
    c->e[k][l] = Complx(0.0, 0.0);
    for(int m=0;m<3;m++)
      c->e[k][l] += a->e[k][m] * b->e[m][l];
  }
}

