#!/bin/bash

exe=mult_su3_nn_f64.exe
#exe=mult_su3_nn_f32.exe
#for l in 8 16 24 32 40 48; do
for l in 8 16 24 32 40; do
  for t in 1 2 4 8 16 32 64; do
    printf "%2s  %2s  " $l $t
    srun $exe -v 0 -l $l -t $t | awk {'print $4'}
  done
done

