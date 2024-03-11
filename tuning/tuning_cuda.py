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
with open("../mat_nn_cuda.hpp", "r") as file:
    kernel_code = file.read()

arguments = parse_cli()
total_sites = np.int32(arguments.ldim**4)
compiler_options = ["-I.."]
if arguments.milc:
    compiler_options += ["-DMILC_COMPLEX"]
if arguments.precision == 1:
    compiler_options += ["-DPRECISION=1"]

# allocate memory
if arguments.precision == 1:
    a = np.random.rand(total_sites * 320).astype(np.byte)
    b = np.random.rand(288).astype(np.byte)
else:
    a = np.random.rand(total_sites * 640).astype(np.byte)
    b = np.random.rand(576).astype(np.byte)
c = np.zeros_like(a)
args = [a, b, c, total_sites]

# tunable parameters
tune_params = dict()
tune_params["block_size_x"] = [32 * i for i in range(1, 33)]

results, _ = tune_kernel(
    "k_mat_nn",
    kernel_code,
    total_sites,
    args,
    tune_params,
    lang="cupy",
    compiler_options=compiler_options,
)
