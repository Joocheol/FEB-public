#!/usr/bin/env python3
"""Reproduce the Chapter 17 deep-hedging benchmark and result tables."""
from __future__ import annotations

import argparse
import csv
import math
import platform
from pathlib import Path

import numpy as np
import torch
from torch import nn

S0 = 100.0
K = 100.0
R = 0.05
SIGMA = 0.20
T = 1.0
STEPS = 20


def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def bs_price() -> float:
    d1 = (math.log(S0 / K) + (R + 0.5 * SIGMA**2) * T) / (SIGMA * math.sqrt(T))
    d2 = d1 - SIGMA * math.sqrt(T)
    normal = torch.distributions.Normal(0.0, 1.0)
    return float(S0 * normal.cdf(torch.tensor(d1)) - K * math.exp(-R*T) * normal.cdf(torch.tensor(d2)))


def bs_delta(stock: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    root = torch.sqrt(torch.clamp(tau, min=1e-8))
    d1 = (torch.log(stock / K) + (R + 0.5 * SIGMA**2) * tau) / (SIGMA * root)
    return normal_cdf(d1)


class HedgePolicy(nn.Module):
    def __init__(self, hidden: int = 32, delta_max: float = 1.2) -> None:
        super().__init__()
        self.delta_max = delta_max
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.delta_max * torch.sigmoid(self.net(features).squeeze(-1))


def simulate(batch: int, generator: torch.Generator) -> torch.Tensor:
    dt = T / STEPS
    z = torch.randn(batch, STEPS, generator=generator)
    step = (R - 0.5 * SIGMA**2) * dt + SIGMA * math.sqrt(dt) * z
    future = S0 * torch.exp(torch.cumsum(step, dim=1))
    return torch.cat([torch.full((batch, 1), S0), future], dim=1)


def execute(policy: HedgePolicy, stock: torch.Tensor, premium: float, kappa: float):
    batch = stock.shape[0]
    dt = T / STEPS
    cash = torch.full((batch,), premium)
    previous = torch.zeros_like(cash)
    total_cost = torch.zeros_like(cash)
    turnover = torch.zeros_like(cash)
    for i in range(STEPS):
        tau_ratio = (T - i * dt) / T
        features = torch.stack([
            torch.log(stock[:, i] / K),
            torch.full_like(cash, tau_ratio),
            previous,
        ], dim=1)
        delta = policy(features)
        trade = delta - previous
        cost = kappa * stock[:, i] * torch.abs(trade)
        cash = (cash - trade * stock[:, i] - cost) * math.exp(R * dt)
        total_cost += cost
        turnover += torch.abs(trade)
        previous = delta
    liquidation = kappa * stock[:, -1] * torch.abs(previous)
    total_cost += liquidation
    turnover += torch.abs(previous)
    wealth = cash + previous * stock[:, -1] - liquidation
    payoff = torch.relu(stock[:, -1] - K)
    return payoff - wealth, total_cost, turnover


def train(seed: int, kappa: float, phases: list[tuple[int, float]], batch: int, validation: int):
    torch.manual_seed(seed)
    policy = HedgePolicy()
    generator = torch.Generator().manual_seed(seed + 1)
    val_generator = torch.Generator().manual_seed(seed + 2)
    val_stock = simulate(validation, val_generator)
    premium = bs_price()
    history = []
    elapsed = 0
    for epochs, learning_rate in phases:
        optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        for _ in range(epochs):
            elapsed += 1
            stock = simulate(batch, generator)
            error, _, _ = execute(policy, stock, premium, kappa)
            objective = torch.mean(error**2)
            optimizer.zero_grad()
            objective.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            optimizer.step()
            if elapsed == 1 or elapsed % 25 == 0:
                with torch.no_grad():
                    val_error, _, _ = execute(policy, val_stock, premium, kappa)
                    history.append((elapsed, float(objective), float(torch.mean(val_error**2))))
    return policy, history


def execute_bs(stock: torch.Tensor, premium: float, kappa: float):
    batch = stock.shape[0]
    dt = T / STEPS
    cash = torch.full((batch,), premium)
    previous = torch.zeros_like(cash)
    total_cost = torch.zeros_like(cash)
    turnover = torch.zeros_like(cash)
    for i in range(STEPS):
        tau = torch.full_like(cash, T - i * dt)
        delta = bs_delta(stock[:, i], tau)
        trade = delta - previous
        cost = kappa * stock[:, i] * torch.abs(trade)
        cash = (cash - trade * stock[:, i] - cost) * math.exp(R * dt)
        total_cost += cost
        turnover += torch.abs(trade)
        previous = delta
    liquidation = kappa * stock[:, -1] * torch.abs(previous)
    total_cost += liquidation
    turnover += torch.abs(previous)
    wealth = cash + previous * stock[:, -1] - liquidation
    payoff = torch.relu(stock[:, -1] - K)
    return payoff - wealth, total_cost, turnover


def summarize(error: torch.Tensor, cost: torch.Tensor, turnover: torch.Tensor):
    values = error.detach().numpy()
    threshold = np.quantile(values, 0.95, method="higher")
    return {
        "mean_error": float(values.mean()),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "es95": float(values[values >= threshold].mean()),
        "average_cost": float(cost.mean()),
        "turnover": float(turnover.mean()),
    }


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="run a short smoke test")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    torch.set_num_threads(4)
    print(f"Python {platform.python_version()}, torch {torch.__version__}, numpy {np.__version__}")
    if args.quick:
        no_cost_phases = [(20, 2e-3)]
        cost_phases = [(20, 2e-3)]
        batch, validation, test_paths = 512, 2048, 5000
    else:
        no_cost_phases = [(800, 2e-3), (700, 5e-4)]
        cost_phases = [(800, 2e-3), (1000, 5e-4)]
        batch, validation, test_paths = 2048, 16384, 100000

    premium = bs_price()
    policy_zero, history_zero = train(2026, 0.0, no_cost_phases, batch, validation)
    policy_cost, history_cost = train(3026, 0.005, cost_phases, batch, validation)

    test = simulate(test_paths, torch.Generator().manual_seed(9999))
    with torch.no_grad():
        e0, c0, t0 = execute(policy_zero, test, premium, 0.0)
        eb0, cb0, tb0 = execute_bs(test, premium, 0.0)
        ec, cc, tc = execute(policy_cost, test, premium, 0.005)
        ebc, cbc, tbc = execute_bs(test, premium, 0.005)
        payoff = torch.relu(test[:, -1] - K)
        no_trade_error = payoff - premium * math.exp(R*T)
        zeros = torch.zeros_like(no_trade_error)

    rows = []
    for strategy, kappa, triplet in [
        ("learned policy", 0.0, (e0,c0,t0)),
        ("Black--Scholes delta", 0.0, (eb0,cb0,tb0)),
        ("learned policy", 0.005, (ec,cc,tc)),
        ("Black--Scholes delta", 0.005, (ebc,cbc,tbc)),
        ("no trade", 0.0, (no_trade_error,zeros,zeros)),
    ]:
        row = {"strategy": strategy, "kappa": kappa}
        row.update(summarize(*triplet))
        rows.append(row)
    save_csv(args.output / "ch17_metrics_generated.csv", rows)

    history_rows = [
        {"model": "no cost", "epoch": e, "training_mse": tr, "validation_mse": va}
        for e,tr,va in history_zero
    ] + [
        {"model": "cost aware", "epoch": e, "training_mse": tr, "validation_mse": va}
        for e,tr,va in history_cost
    ]
    save_csv(args.output / "ch17_learning_curve_generated.csv", history_rows)

    with torch.no_grad():
        stock_grid = torch.linspace(70.0, 130.0, 13)
        tau = torch.full_like(stock_grid, 0.5)
        zero_previous = torch.zeros_like(stock_grid)
        learned = policy_zero(torch.stack([torch.log(stock_grid/K), tau, zero_previous], 1))
        benchmark = bs_delta(stock_grid, tau)
        policy_rows = [
            {"stock": float(s), "learned_position": float(a), "black_scholes_delta": float(b)}
            for s,a,b in zip(stock_grid, learned, benchmark)
        ]
        save_csv(args.output / "ch17_policy_comparison_generated.csv", policy_rows)

        previous = torch.linspace(0.0, 1.0, 21)
        features = torch.stack([torch.zeros_like(previous), torch.full_like(previous,0.5), previous], 1)
        trade_zero = policy_zero(features) - previous
        trade_cost = policy_cost(features) - previous
        trade_rows = [
            {"previous_position": float(p), "trade_without_cost": float(a), "trade_with_cost": float(b)}
            for p,a,b in zip(previous, trade_zero, trade_cost)
        ]
        save_csv(args.output / "ch17_trade_adjustment_generated.csv", trade_rows)

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
