#!/bin/bash
export MV2_ENABLE_AFFINITY=0
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=$((SLURM_CPUS_ON_NODE / 2))

set -x

kernel=k_mat_nn
threads=512

filename=profile-${kernel}-t${threads}
skip=1
count=10

srun nv-nsight-cu-cli -o $filename -k $kernel -s $skip -c $count ./bench_f32_cuda.exe -t $threads

