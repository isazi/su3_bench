#!/bin/bash
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=8

for exe in *.exe; do 
  echo "---------> ${exe}"
#  for trial in 1 2 3; do
#    echo "---------> Trial $trial"
   ./${exe}
#  done
done

