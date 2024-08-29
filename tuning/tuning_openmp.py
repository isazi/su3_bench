from tuning_common import parse_cli, compute_sizes, add_metrics
import numpy as np
from kernel_tuner import tune_kernel
from kernel_tuner.utils.directives import (
    Code,
    OpenMP,
    Cxx,
    extract_directive_code,
    extract_initialization_code,
    extract_deinitialization_code,
    extract_directive_signature,
    generate_directive_function,
)


# read OpenACC file
with open("../mat_nn_openmp.hpp", "r") as file:
    kernel_code = file.read()

arguments = parse_cli()
total_sites = np.int32(arguments.ldim**4)
compiler_options = [
    "-std=c++14",
    "-fast",
    "-mp=gpu",
    "-I..",
    "-DUSE_OPENMP",
]
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

# generate code
preprocessor = ["#include <lattice.hpp>"]
dimensions = dict()
dimensions["len_a"] = total_sites
dimensions["len_b"] = 4
dimensions["len_c"] = total_sites
app = Code(OpenMP(), Cxx())
init = extract_initialization_code(kernel_code, app)
deinit = extract_deinitialization_code(kernel_code, app)
signature = extract_directive_signature(kernel_code, app, "k_mat_nn")
body = extract_directive_code(kernel_code, app, "k_mat_nn")
kernel_string = generate_directive_function(
    preprocessor,
    signature["k_mat_nn"],
    body["k_mat_nn"],
    app,
    initialization=init,
    deinitialization=deinit,
    user_dimensions=dimensions,
)

# tunable parameters
tune_params = dict()
tune_params["USE_VERSION"] = [0, 1, 2, 3, 4, 5]

# metrics
metrics = dict()
add_metrics(metrics, total_sites, site_size, matrix_size)

results, _ = tune_kernel(
    "k_mat_nn",
    kernel_string,
    0,
    args,
    tune_params,
    compiler="nvc++",
    compiler_options=compiler_options,
    metrics=metrics,
)
