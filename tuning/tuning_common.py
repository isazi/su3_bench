import argparse


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
    parser.add_argument(
        "--threads", help="The number of threads per site.", type=int, default=36
    )
    return parser.parse_args()


def compute_sizes(precision: int) -> tuple[int, int]:
    if precision == 1:
        site_size = 320
        matrix_size = 288
    else:
        site_size = 640
        matrix_size = 576
    return site_size, matrix_size


def add_metrics(metrics: dict, sites: int, site_size: int, matrix_size: int):
    metrics["GFLOP/s"] = lambda p: (sites * 864.0) / (p["time"] / 1000) / 10**9
    metrics["GB/s"] = (
        lambda p: ((sites * site_size * 2) + matrix_size) / (p["time"] / 1000) / 10**9
    )
