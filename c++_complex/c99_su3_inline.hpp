
//*******************  m_mat_nn.c  (in su3.a) ****************************
//  void mult_su3_nn( su3_matrix *a,*b,*c )
//  matrix multiply, no adjoints 
//  C  <-  A*B	
static inline void mult_su3_nn( su3_matrix *a, su3_matrix *b, su3_matrix *c ){
  for(int i=0;i<3;i++)for(int j=0;j<3;j++){
    c->e[i][j] = Complx(0.0, 0.0);
    for(int k=0;k<3;k++)
      c->e[i][j] += a->e[i][k] * b->e[k][j];
  }
}

