import argparse
import numpy as np
from kernel_tuner import tune_kernel


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ldim", help="Lattice size in one dimension.", type=int, default=32
    )
    parser.add_argument(
        "--precision",
        help="Floating point precision.",
        type=int,
        choices=range(1, 3),
        default=2,
    )
    parser.add_argument("--milc", help="Enable MILC_COMPLEX.", action="store_true")
    return parser.parse_args()


# read CUDA code
with open("../mat_nn_cuda_kernel.cu", "r") as file:
    kernel_code = file.read()

arguments = parse_cli()
total_sites = np.int32(arguments.ldim**4)
compiler_options = ["-std=c++14", "-I..", "-default-device", "-DUSE_CUDA"]
if arguments.milc:
    compiler_options += ["-DMILC_COMPLEX"]
if arguments.precision == 1:
    compiler_options += ["-DPRECISION=1"]

# allocate memory
site_size = 0
matrix_size = 0
if arguments.precision == 1:
    site_size = 320
    matrix_size = 288
else:
    site_size = 640
    matrix_size = 576
a = np.random.rand(total_sites * site_size).astype(np.byte)
b = np.random.rand(matrix_size).astype(np.byte)
c = np.zeros_like(a)
args = [a, b, c, total_sites]

# tunable parameters
tune_params = dict()
tune_params["block_size_x"] = [32 * i for i in range(1, 33)]

# metrics
metrics = dict()
metrics["GFLOP/s"] = lambda p: (total_sites * 864.0) / (p["time"] / 1000) / 10**9
metrics["GB/s"] = lambda p: (len(a) + len(c) + len(b)) / (p["time"] / 1000) / 10**9

results, _ = tune_kernel(
    "k_mat_nn",
    kernel_code,
    total_sites,
    args,
    tune_params,
    lang="cupy",
    compiler_options=compiler_options,
    metrics=metrics,
)
