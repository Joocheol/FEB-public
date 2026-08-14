from math import erf, exp, log, sqrt


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_call(S: float, K: float, r: float, q: float, sigma: float, tau: float) -> float:
    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt(tau))
    d2 = d1 - sigma * sqrt(tau)
    return S * exp(-q * tau) * normal_cdf(d1) - K * exp(-r * tau) * normal_cdf(d2)


def main() -> None:
    S = 100.0
    K = 100.0
    H = 80.0
    r = 0.04
    q = 0.01
    sigma = 0.20
    tau = 0.5

    lam = (r - q + 0.5 * sigma**2) / sigma**2
    reflected_S = H**2 / S
    reflection_weight = (H / S) ** (2.0 * lam - 2.0)

    vanilla = bs_call(S, K, r, q, sigma, tau)
    reflected = bs_call(reflected_S, K, r, q, sigma, tau)
    down_in = reflection_weight * reflected
    down_out = vanilla - down_in

    print(f"lambda              = {lam:.8f}")
    print(f"reflection weight   = {reflection_weight:.8f}")
    print(f"reflected stock     = {reflected_S:.8f}")
    print(f"vanilla call        = {vanilla:.8f}")
    print(f"reflected call      = {reflected:.8f}")
    print(f"down-and-out call   = {down_out:.8f}")
    print(f"down-and-in call    = {down_in:.8f}")

    assert abs(lam - 1.25) < 1e-12
    assert abs(reflected_S - 64.0) < 1e-12
    assert abs(reflected - 0.00357) < 5e-6
    assert abs(down_out - 6.33606) < 5e-6
    assert abs(down_in - 0.00319) < 5e-6
    assert abs((down_out + down_in) - vanilla) < 1e-12


if __name__ == "__main__":
    main()
