#include <stdio.h>
#include <stdlib.h>
#include "su3.h"
#include "lattice.h"

void make_lattice(){
  /* allocate space for lattice, fill in parity, coordinates and index */
#ifdef DEBUG
  printf("Mallocing %d Bytes per node for lattice\n",
	       sites_on_node * sizeof(site));
#endif
  lattice = (site *)malloc( sites_on_node * sizeof(site) );
  if(lattice==NULL){
    printf("ERROR: no room for lattice\n");
    exit(1);
  }

  int i=0;
  for(int t=0;t<nt;t++)for(int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++){
      lattice[i].x=x;	lattice[i].y=y;	lattice[i].z=z;	lattice[i].t=t;
      lattice[i].index = x+nx*(y+ny*(z+nz*t));
      if( (x+y+z+t)%2 == 0)lattice[i].parity=EVEN;
      else	         lattice[i].parity=ODD;

      for(int j=0; j<4; ++j) for(int k=0; k<3; ++k) for(int l=0; l<3; ++l) {
        lattice[i].link[j].e[k][l]=1.0+0.0*I;
      }

      ++i;
  }
}

void free_lattice()
{
  free(lattice);
}
