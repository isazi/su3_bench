from tuning_common import parse_cli, compute_sizes, add_metrics
import numpy as np
from kernel_tuner import tune_kernel


# read CUDA code
with open("../mat_nn_opencl_kernel.cpp", "r") as file:
    kernel_code = file.read()

arguments = parse_cli()
total_sites = np.int32(arguments.ldim**4)
compiler_options = ["-std=c++14", "-I..", "-default-device", "-DUSE_OPENCL"]
if arguments.milc:
    compiler_options += ["-DMILC_COMPLEX"]
if arguments.precision == 1:
    compiler_options += ["-DPRECISION=1"]

# allocate memory and load
site_size, matrix_size = compute_sizes(arguments.precision)
a = np.random.rand(total_sites * site_size).astype(np.byte)
b = np.random.rand(matrix_size).astype(np.byte)
c = np.zeros_like(a)

args = [a, b, c, total_sites]

# tunable parameters
tune_params = dict()
tune_params["block_size_x"] = [32 * i for i in range(1, 33)]

# metrics
metrics = dict()
add_metrics(metrics, total_sites, site_size, matrix_size)

results, _ = tune_kernel(
    "k_mat_nn",
    kernel_code,
    total_sites * arguments.threads,
    args,
    tune_params,
    lang="opencl",
    compiler_options=compiler_options,
    metrics=metrics,
    )
