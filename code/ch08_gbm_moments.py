#!/usr/bin/env python3
"""Reproduce the GBM moment check used in Chapter 8."""

import argparse
import math
import random
import statistics


SEED = 2026
S0 = 100.0
MU = 0.08
SIGMA = 0.20
T = 1.0
N_PATHS = 200_000


def path_count(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("--paths must be at least 2")
    return parsed


def main(paths: int) -> None:
    rng = random.Random(SEED)
    terminal = [
        S0
        * math.exp(
            (MU - 0.5 * SIGMA**2) * T
            + SIGMA * math.sqrt(T) * rng.gauss(0.0, 1.0)
        )
        for _ in range(paths)
    ]

    sample_mean = statistics.fmean(terminal)
    sample_variance = statistics.variance(terminal)
    theoretical_mean = S0 * math.exp(MU * T)
    theoretical_variance = (
        S0**2 * math.exp(2 * MU * T) * (math.exp(SIGMA**2 * T) - 1)
    )

    print(f"sample mean:        {sample_mean:.6f}")
    print(f"theoretical mean:   {theoretical_mean:.6f}")
    print(f"sample variance:    {sample_variance:.6f}")
    print(f"theoretical variance: {theoretical_variance:.6f}")

    if paths == N_PATHS:
        assert abs(sample_mean / theoretical_mean - 1.0) < 0.01
        assert abs(sample_variance / theoretical_variance - 1.0) < 0.03


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths",
        type=path_count,
        default=N_PATHS,
        help=f"number of simulated paths (default: {N_PATHS})",
    )
    args = parser.parse_args()
    main(args.paths)
