#ifndef _LATTICE_HPP
#define _LATTICE_HPP
// Adapted from lattice.h in MILC version 7

#ifdef __OPENCL_VERSION__
#  include "su3.h"
#else
#  include "su3.hpp"
#endif

#define EVEN 0x02
#define ODD 0x01

#define ALIGN_N 8

// The lattice is an array of sites
typedef struct {
	su3_matrix link[4];  // the fundamental gauge field
	int x,y,z,t;         // coordinates of this site
	int index;           // my index in the array
	char parity;         // is it even or odd?
#if (PRECISION==1)
        int pad[2];          // pad out to 64 byte alignment
#else
        int pad[10];         // pad out to 64 byte alignment
#endif
} site __attribute__ ((aligned));

// globals related to the lattice
extern void init_link(su3_matrix *, Complx);
extern void make_lattice(site *, int);

#endif // _LATTICE_HPP
