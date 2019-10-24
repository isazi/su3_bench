#!/bin/bash

exe=mult_su3_nn_f64.exe
i=100
t=4
while [ "$#" -ge 1 ] ; do
  case "$1" in
    "-h" | "--help")    echo "usage:$0 [-d device] [-s <value only>] [-exe binary]"; exit;;
    "-e" | "--exe")     exe=$2; shift 2;;
    "-i" | "--iters")   i=$2; shift 2;;
    "-t" | "--threads") t=$2; shift 2;;
    *)                  break;;
  esac
done

command="srun nvprof --kernels k_mat_nn --metrics flop_count_dp --metrics dram_read_transactions --metrics dram_write_transactions ./$exe -i $i -t $t"
echo $command
$command
