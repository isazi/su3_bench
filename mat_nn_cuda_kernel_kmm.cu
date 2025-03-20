#ifdef kernel_tuner
#include <lattice.hpp>
#endif

//*******************  m_mat_nn.c  (in su3.a) ****************************
//  void mult_su3_nn( su3_matrix *a,*b,*c )
//  matrix multiply, no adjoints
//  C  <-  A*B
__global__ void k_mat_nn(
    kmm::Range<int64_t> range,
    kmm::GPUSubview<site> a,
    kmm::GPUSubview<su3_matrix> b,
    kmm::GPUSubviewMut<site> c,
    int total_sites)
{
    int myThread = blockDim.x * blockIdx.x + threadIdx.x + range.begin;
    int mySite = myThread/36;

    if (mySite < range.end && mySite < total_sites) {
        int j = (myThread%36)/9;
        int k = (myThread%9)/3;
        int l = myThread%3;
        Complx cc = {0.0, 0.0};
        for (int m=0;m<3;m++)
#ifdef MILC_COMPLEX
            CMULSUM(a[mySite].link[j].e[k][m], b[j].e[m][l], cc);
#else
                cc += a[mySite].link[j].e[k][m] * b[j].e[m][l];
#endif
        c[mySite].link[j].e[k][l] = cc;
    }
}