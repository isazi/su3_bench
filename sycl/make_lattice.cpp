// Derived from make_lattice() in MILC versin 7
#include <stdio.h>
#include <stdlib.h>
#include "lattice.hpp"

void init_link(su3_matrix *s, Complx val) {
  for(int j=0; j<4; ++j) for(int k=0; k<3; ++k) for(int l=0; l<3; ++l) {
    s[j].e[k][l].real=val.real;
    s[j].e[k][l].imag=val.imag;
  }
}

void make_lattice(site *s, int n) {
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
      Complx val = {1.0, 0.0};
      init_link(&s[i].link[0], val);
    }
  }
}

