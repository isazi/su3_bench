#ifdef kernel_tuner
#include <lattice.hpp>
#endif

//*******************  m_mat_nn.c  (in su3.a) ****************************
//  void mult_su3_nn( su3_matrix *a,*b,*c )
//  matrix multiply, no adjoints
//  C  <-  A*B
static const char kernel_src[] =
    "#include <lattice.hpp>\n"
    "__kernel void k_mat_nn(\n"
    "  __global const site*       restrict a,\n"
    "  __global const su3_matrix* restrict b,\n"
    "  __global       site*       restrict c,\n"
    "           const int         total_sites)\n"
    "{\n"
    "  int myThread = get_global_id(0);\n"
    "  int mySite = myThread/36;\n"
    "  if (mySite < total_sites) {\n"
    "    int j = (myThread%36)/9;\n"
    "    int k = (myThread%9)/3;\n"
    "    int l = myThread%3;\n"
    "    Complx cc = {0.0, 0.0};\n"
    "    for (int m=0;m<3;m++)\n"
    "      CMULSUM(a[mySite].link[j].e[k][m], b[j].e[m][l], cc);\n"
    "    c[mySite].link[j].e[k][l].real = cc.real;\n"
    "    c[mySite].link[j].e[k][l].imag = cc.imag;\n"
    "  }\n"
    "}\n";