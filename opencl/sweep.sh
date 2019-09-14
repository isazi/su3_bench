#!/bin/bash

#exe=mult_su3_nn_f64.exe
exe=mult_su3_nn_f32.exe
for n in 8 16 24 32 40 48; do
  for g in 0 1 2 4 8 16 32 64; do
    printf "%2s  %2s  " $n $g
    srun $exe -v 0 -n $n -g $g | awk {'print $4'}
  done
done

