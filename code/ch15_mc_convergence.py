#!/usr/bin/env python3
"""Check the Monte Carlo root-M convergence rate used in Chapter 15."""

import math
import random
import statistics


SEED = 2026
S0 = K = 100.0
R = 0.05
SIGMA = 0.20
T = 1.0
SAMPLE_SIZES = [2_000, 8_000, 32_000, 128_000]


def main() -> None:
    rng = random.Random(SEED)
    standard_errors = []
    for size in SAMPLE_SIZES:
        discounted_payoff = []
        for _ in range(size):
            z = rng.gauss(0.0, 1.0)
            terminal = S0 * math.exp(
                (R - 0.5 * SIGMA**2) * T + SIGMA * math.sqrt(T) * z
            )
            discounted_payoff.append(
                math.exp(-R * T) * max(terminal - K, 0.0)
            )
        standard_errors.append(
            statistics.stdev(discounted_payoff) / math.sqrt(size)
        )

    xs = [math.log(size) for size in SAMPLE_SIZES]
    ys = [math.log(se) for se in standard_errors]
    xbar = statistics.fmean(xs)
    ybar = statistics.fmean(ys)
    slope = sum(
        (x - xbar) * (y - ybar) for x, y in zip(xs, ys)
    ) / sum((x - xbar) ** 2 for x in xs)

    for size, se in zip(SAMPLE_SIZES, standard_errors):
        print(f"M={size:6d}  SE={se:.6f}")
    print(f"log-log slope: {slope:.6f}")

    assert -0.65 < slope < -0.35


if __name__ == "__main__":
    main()
