#ifndef _SU3_H
#define _SU3_H

#include <complex.h>

/******************************  su3.h **********************************
*									*
*  Defines and subroutine declarations for SU3 simulation		*
*  MIMD version 7 							*
*									*
*/
/* SU(3) */

typedef struct { float complex e[3][3]; } fsu3_matrix;
typedef struct { float complex c[3]; } fsu3_vector;

typedef struct { double complex e[3][3]; } dsu3_matrix;
typedef struct { double complex c[3]; } dsu3_vector;

#if (PRECISION==1)

#define su3_matrix      fsu3_matrix
#define su3_vector      fsu3_vector
#define Real		float
#define Complx          float complex

#else

#define su3_matrix      dsu3_matrix
#define su3_vector      dsu3_vector
#define Real		double
#define Complx          double complex

#endif

#endif

