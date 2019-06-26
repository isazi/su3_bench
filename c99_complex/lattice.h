#ifndef _LATTICE_H
#define _LATTICE_H
/****************************** lattice.h ********************************/

/* include file for MIMD version 7
   This file defines global scalars and the fields in the lattice.

   Directory for dynamical improved KS action.  Allow:
	arbitrary paths in quark action 
	arbitrary paths in gauge action (eg Symanzik imp.)

   If "FN" is defined,
     Includes storage for Naik improvement (longlink[4], templongvec[4],
     gen_pt[16], etc.
*/

#include "su3.h"
#define EVEN 0x02
#define ODD 0x01
#define EVENANDODD 0x03

/* The lattice is an array of sites.  */
typedef struct {
	/* gauge field */
	su3_matrix link[4];	/* the fundamental field */
	/* coordinates of this site */
	short x,y,z,t;
	/* is it even or odd? */
	char parity;
	/* my index in the array */
	int index;
//} site;
} site __attribute__ ((aligned (64)));

// globals related to the lattice
extern void make_lattice();
extern void free_lattice();
extern site *lattice;		// lattice store
extern int nx,ny,nz,nt;	// lattice dimensions

/* Some of these global variables are node dependent */
/* They are set in "make_lattice()" */
extern  int sites_on_node;              /* number of sites on this node */
extern  int even_sites_on_node; /* number of even sites on this node */
extern  int odd_sites_on_node;  /* number of odd sites on this node */
extern  int number_of_nodes;    /* number of nodes in use */
extern  int this_node;          /* node number of this node */

/* Vectors for addressing */
/* Generic pointers, for gather routines */
#define N_POINTERS 16
extern char ** gen_pt[N_POINTERS];

#endif
