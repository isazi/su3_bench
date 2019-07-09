// Derived from make_lattice() in MILC versin 7
#include <stdio.h>
#include <stdlib.h>
#include "su3.hpp"
#include "lattice.hpp"

void make_lattice(site **lattice, int n, int *total_sites) {

  /* allocate space for lattice, fill in parity, coordinates and index */
  *total_sites = n*n*n*n;
  site *s;
  int errval;
  if ((errval = posix_memalign((void **)&s, ALIGN_N, *total_sites * sizeof(site))) != 0) {
    if (errval == EINVAL)
      printf("ERROR: Invalid alignment for lattice allocation\n");
    else
      printf("ERROR: Insufficient memory for lattice allocation\n");
    exit(errval);
  }

  int nx=n;
  int ny=n;
  int nz=n;
  int nt=n;
  #pragma omp parallel for
  for(int t=0;t<nt;t++) {
    int i=t*nz*ny*nx;
    for(int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++,i++){
      s[i].x=x; s[i].y=y; s[i].z=z; s[i].t=t;
      s[i].index = x+nx*(y+ny*(z+nz*t));
      if( (x+y+z+t)%2 == 0)
        s[i].parity=EVEN;
      else
        s[i].parity=ODD;
      for(int j=0; j<4; ++j) for(int k=0; k<3; ++k) for(int l=0; l<3; ++l) {
        s[i].link[j].e[k][l]=Complx(1.0,0.0);
      }
    }
  }
  *lattice = s;
}

void free_lattice(site *lattice)
{
  free(lattice);
}
