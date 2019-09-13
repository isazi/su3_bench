#include <lattice.hpp>

//*******************  m_mat_nn.c  (in su3.a) ****************************
//  void mult_su3_nn( su3_matrix *a,*b,*c )
//  matrix multiply, no adjoints 
//  C  <-  A*B	
__kernel void k_mat_nn(
           const int         num_wi,
  __global const site*       restrict a,
  __global const su3_matrix* restrict b,
  __global       site*       restrict c )
{
  size_t mysite = get_global_id(0);
#ifdef DEBUG
  printf("mysite = %d\n", (int)mysite);
#endif

  size_t total_wi = get_num_groups(0) * get_local_size(0);
  size_t sites_per_wi = (num_wi + (total_wi-1)) / total_wi;    

  size_t start  = get_group_id(0) * get_local_size(0) + get_local_id(0) ;
  size_t end    = start + sites_per_wi * total_wi;
  if (start > num_wi) start = num_wi;
  if (end > num_wi) end = num_wi;

#ifdef DEBUG
  if (get_group_id(0) == 0 && get_local_id(0) == 0)
  printf("kernel: group id %d local id %d: sites_per_wi %d, start %d, end %d, stride %d\n",
         (int)get_group_id(0), (int)get_local_id(0), (int)sites_per_wi, (int)start, (int)end, (int)total_wi);
#endif

  for (int i=start; i<end; i+=total_wi) {
    for (int j=0; j<4; ++j) {
      for (int k=0;k<3;k++) {
        for (int l=0;l<3;l++){
          c[i].link[j].e[k][l].real=0.0;
          c[i].link[j].e[k][l].imag=0.0;
          for (int m=0;m<3;m++) {
            CMULSUM(a[i].link[j].e[k][m], b[j].e[m][l], c[i].link[j].e[k][l]);
#ifdef DEBUG
            if (i==0 && m==2)
            printf("a[%d][%d]->e[%d][%d]=%f b[%d][%d]->e[%d][%d]=%f c[%d][%d]->e[%d][%d]=%f\n",
                    j,(int)i,k,m,a[i].link[j].e[k][m].real,
                    j,(int)i,m,l,b[j].e[m][l].real,
                    j,(int)i,k,l,c[i].link[j].e[k][l].real);
#endif
          }
        }
      }
    }
  }
}

