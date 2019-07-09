
//*******************  m_mat_nn.c  (in su3.a) ****************************
//  void mult_su3_nn( su3_matrix *a,*b,*c )
//  matrix multiply, no adjoints 
//  C  <-  A*B	
static inline void mult_su3_nn( su3_matrix * restrict a, su3_matrix * restrict b, 
su3_matrix * restrict c ){
  for(int i=0;i<3;i++)for(int j=0;j<3;j++){
    c->e[i][j] = 0.0+0.0*I;
    for(int k=0;k<3;k++)
      c->e[i][j] += a->e[i][k] * b->e[k][j];
  }
}

/****************  m_matvec.c  (in su3.a) *******************************
*                                                                       *
* void mult_su3_mat_vec( su3_matrix *a, su3_vector *b,*c )              *
* matrix times vector multiply, no adjoints                             *
*  C  <-  A*B                                                           *
*/
static inline void mult_su3_mat_vec( su3_matrix * restrict a, su3_vector * restrict b, 
su3_vector * restrict c  ){
  for(int i=0;i<3;i++){
    c->c[i] = 0.0+0.0*I;
    for(int j=0;j<3;j++)
        c->c[i] += a->e[i][j] * b->c[j];
  }
}

/****************  m_matvec_s.c  (in su3.a) *****************************
*                                                                       *
* void mult_su3_mat_vec_sum( su3_matrix *a, su3_vector *b,*c )          *
* su3_matrix times su3_vector multiply and add to another su3_vector    *
* C  <-  C + A*B                                                        *
*/
static inline void mult_su3_mat_vec_sum( su3_matrix * restrict a, 
su3_vector * restrict b, su3_vector * restrict c ){
  for(int i=0;i<3;i++){
    for(int j=0;j<3;j++)
        c->c[i] += a->e[i][j] * b->c[j];
  }
}

/*****************  m_amatvec.c  (in su3.a) *****************************
*                                                                       *
*  void mult_adj_su3_mat_vec( su3_matrix *a, su3_vector *b,*c )         *
*  C  <-  A_adjoint * B                                                 *
*/
static inline void mult_adj_su3_mat_vec( su3_matrix * restrict a, 
su3_vector * restrict b, su3_vector * restrict c ){
  for(int i=0;i<3;i++){
    c->c[i] = 0.0+0.0*I;
    for(int j=0;j<3;j++)
      c->c[i] += conj(a->e[j][i]) * b->c[j];
  }
}

/****************  s_m_a_vec.c  (in su3.a) ******************************
*                                                                       *
* void scalar_mult_add_su3_vector( su3_vector *a, su3_vector *b,        *
*       Real s, su3_vector *c)                                          *
* C <- A + s*B,   A,B and C vectors                                     *
*/
static inline void scalar_mult_add_su3_vector(su3_vector * restrict a, 
su3_vector * restrict b, Real s, su3_vector * restrict c){
  int i;
  for(int i=0;i<3;i++)
    c->c[i] = a->c[i] + s * b->c[i];
}

/****************  m_mv_s_4dir.c  (in su3.a) *****************************
*                                                                       *
* void mult_su3_mat_vec_sum_4dir( su3_matrix *a, su3_vector *b[0123],*c )*
* Multiply the elements of an array of four su3_matrices by the         *
* four su3_vectors, and add the results to                              *
* produce a single su3_vector.                                          *
* C  <-  A[0]*B[0]+A[1]*B[1]+A[2]*B[2]+A[3]*B[3]                        *
*/
static inline void mult_su3_mat_vec_sum_4dir(  su3_matrix * restrict a, 
su3_vector * restrict b0, su3_vector * restrict b1, su3_vector * restrict b2, 
su3_vector * restrict b3, su3_vector * restrict c  ){
    mult_su3_mat_vec( a+0,b0,c );
    mult_su3_mat_vec_sum( a+1,b1,c );
    mult_su3_mat_vec_sum( a+2,b2,c );
    mult_su3_mat_vec_sum( a+3,b3,c );
}

/*****************  m_amv_4dir.c  (in su3.a) *****************************
*                                                                       *
*  void mult_adj_su3_mat_vec_4dir( su3_matrix *mat,                     *
*  su3_vector *src, su3_vector *dest )                                  *
*  Multiply an su3_vector by an array of four adjoint su3_matrices,     *
*  result in an array of four su3_vectors.                              *
*  dest[i]  <-  A_adjoint[i] * src                                      *
*/
static inline void mult_adj_su3_mat_vec_4dir( su3_matrix * restrict mat, 
su3_vector * restrict src, su3_vector * restrict dest ) {
    mult_adj_su3_mat_vec( mat+0, src, dest+0 );
    mult_adj_su3_mat_vec( mat+1, src, dest+1 );
    mult_adj_su3_mat_vec( mat+2, src, dest+2 );
    mult_adj_su3_mat_vec( mat+3, src, dest+3 );
}


