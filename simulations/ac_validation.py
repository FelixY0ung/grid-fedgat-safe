"""Nonlinear radial AC validation for Grid-FedGAT-Safe schedules.

The main controller uses a linear voltage-sensitivity projection. This script
checks the resulting schedules with a backward-forward-sweep AC power-flow
model on radial chain feeders. It is a local validation substitute because
pandapower/OpenDSS are not installed in the execution environment.

Outputs:
  results/ac_validation_metrics.csv
  results/ac_validation_aggregate.csv
  results/ac_validation_summary.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import pilot_grid_fedgat as sim
import robust_grid_fedgat as robust


PROJECT_ROOT = Path(__file__).resolve().parents[1]


S_BASE_KVA = 1000.0
BASE_POWER_FACTOR = 0.95
AC_TOL = 1e-8
AC_MAX_ITER = 100
AC_IMPEDANCE_SCALE = 0.85
POLICIES = [
    "uncontrolled",
    "edf_safe",
    "fedavg_local_safe",
    "fedavg_graph_safe",
    "grid_fedgat_pilot",
]


def line_impedances_from_sensitivity() -> np.ndarray:
    """Build chain-feeder line impedances consistent with sensitivity stress.

    For a radial feeder, the LinDistFlow coefficient for active-power voltage
    drop is approximately 2 * cumulative_r / S_base. We recover cumulative
    resistance from the diagonal sensitivity terms and use a fixed X/R ratio.
    """
    diag = np.maximum(np.diag(sim.SENS), 1e-7)
    cumulative_r = np.maximum.accumulate(0.5 * S_BASE_KVA * diag)
    line_r = np.diff(np.concatenate([[0.0], cumulative_r]))
    line_r = np.maximum(line_r, 1e-5)
    line_x = 0.65 * line_r
    return AC_IMPEDANCE_SCALE * (line_r + 1j * line_x)


def base_load_by_station(base_scalar: float) -> Tuple[np.ndarray, np.ndarray]:
    total_kw = 90.0 * base_scalar
    weights = 0.7 + 0.6 * sim.DEPTH / max(float(np.max(sim.DEPTH)), 1e-9)
    weights = weights / weights.sum()
    p_kw = total_kw * weights
    q_kw = p_kw * np.tan(np.arccos(BASE_POWER_FACTOR))
    return p_kw, q_kw


def bfs_power_flow(p_kw: np.ndarray, q_kvar: np.ndarray, z_line: np.ndarray) -> Tuple[np.ndarray, bool, int]:
    n = len(p_kw)
    s_pu = (p_kw + 1j * q_kvar) / S_BASE_KVA
    v = np.ones(n + 1, dtype=complex)
    converged = False
    for iteration in range(1, AC_MAX_ITER + 1):
        prev = v.copy()
        i_load = np.conj(s_pu / np.maximum(v[1:], 1e-6))
        i_branch = np.zeros(n, dtype=complex)
        downstream = 0.0j
        for idx in range(n - 1, -1, -1):
            downstream += i_load[idx]
            i_branch[idx] = downstream
        v[0] = 1.0 + 0.0j
        for idx in range(n):
            v[idx + 1] = v[idx] - z_line[idx] * i_branch[idx]
        if np.max(np.abs(v - prev)) < AC_TOL:
            converged = True
            break
    return np.abs(v[1:]), converged, iteration


def train_models(train_days: Iterable[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs_local, ys_local, _ = sim.collect_training_data(train_days, local_only=True)
    xs_graph, ys_graph, _ = sim.collect_training_data(train_days, local_only=False)
    w_local = sim.federated_ridge(xs_local, ys_local, network_aware=False)
    w_graph = sim.federated_ridge(xs_graph, ys_graph, network_aware=False)
    w_prop = sim.federated_sufficient_stat_ridge(xs_graph, ys_graph)
    return w_local, w_graph, w_prop


def schedule_policy_day(
    policy: str,
    day: int,
    w_local: np.ndarray,
    w_graph: np.ndarray,
    w_prop: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    base = sim.base_load_profile(day + 1000)
    sessions = sim.generate_sessions(day + 1000)
    remaining = np.array([s.energy_kwh for s in sessions])
    caps = sim.transformer_cap(base)
    schedule = np.zeros((sim.N_SLOTS, sim.N_STATIONS))
    for t in range(sim.N_SLOTS):
        ub = sim.upper_bound_power(sessions, remaining, t)
        raw = sim.raw_policy(policy, sessions, remaining, t, base, w_local, w_graph, w_prop)
        if policy == "uncontrolled":
            station_power = np.minimum(raw, ub)
        else:
            station_power = sim.project_power(raw, ub, base[t], caps[t])
        schedule[t, :] = station_power
        sim.allocate_edf(sessions, remaining, station_power, t)
    return base, schedule


def validate_policy(
    policy: str,
    test_days: Iterable[int],
    w_local: np.ndarray,
    w_graph: np.ndarray,
    w_prop: np.ndarray,
) -> Dict[str, float]:
    z_line = line_impedances_from_sensitivity()
    min_voltages: List[float] = []
    violations = 0
    nonconverged = 0
    slots = 0
    for day in test_days:
        base, schedule = schedule_policy_day(policy, day, w_local, w_graph, w_prop)
        for t in range(sim.N_SLOTS):
            p_base, q_base = base_load_by_station(base[t])
            p_kw = p_base + schedule[t, :]
            q_kvar = q_base
            vmag, converged, _iters = bfs_power_flow(p_kw, q_kvar, z_line)
            min_v = float(np.min(vmag))
            min_voltages.append(min_v)
            violations += int(np.sum(vmag < sim.V_MIN - 1e-6))
            nonconverged += int(not converged)
            slots += 1
    return {
        "policy": policy,
        "ac_min_voltage": float(np.min(min_voltages)),
        "ac_mean_min_voltage": float(np.mean(min_voltages)),
        "ac_voltage_violation_rate_percent": 100.0 * violations / max(slots * sim.N_STATIONS, 1),
        "ac_nonconverged_slots": nonconverged,
    }


def run_one(feeder: str, cfg: Dict[str, float], seed: int) -> pd.DataFrame:
    robust.configure(seed, cfg)
    w_local, w_graph, w_prop = train_models(range(robust.TRAIN_DAYS))
    rows = [
        validate_policy(policy, range(robust.TEST_DAYS), w_local, w_graph, w_prop)
        for policy in POLICIES
    ]
    df = pd.DataFrame(rows)
    df.insert(0, "seed", seed)
    df.insert(0, "feeder", feeder)
    return df


def aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "ac_min_voltage",
        "ac_mean_min_voltage",
        "ac_voltage_violation_rate_percent",
        "ac_nonconverged_slots",
    ]
    out = metrics.groupby("policy")[cols].agg(["mean", "std", "min", "max"]).reset_index()
    out.columns = ["_".join(c).strip("_") for c in out.columns.to_flat_index()]
    return out


def fmt(agg: pd.DataFrame, policy: str, metric: str, digits: int = 4) -> str:
    row = agg.loc[agg["policy"] == policy].iloc[0]
    return f"{row[f'{metric}_mean']:.{digits}f} +/- {row[f'{metric}_std']:.{digits}f}"


def write_summary(metrics: pd.DataFrame, agg: pd.DataFrame) -> None:
    out = PROJECT_ROOT / "results"
    prop = metrics[metrics["policy"] == "grid_fedgat_pilot"]
    zero_ac = float((prop["ac_voltage_violation_rate_percent"] <= 1e-9).mean())
    nonconv = float(prop["ac_nonconverged_slots"].sum())
    summary = f"""# Nonlinear AC Validation Summary

Generated by `simulations/ac_validation.py`.

The controller enforces a linear voltage-sensitivity safety projection. This
validation replays the resulting schedules through a radial backward-forward
sweep AC power-flow model across `{len(robust.FEEDERS)}` feeder-stress profiles
and `{len(robust.SEEDS)}` seeds.

## Aggregate AC Metrics

| Policy | Minimum voltage | Mean daily slot-min voltage | AC voltage violations (%) | Nonconverged slots |
|---|---:|---:|---:|---:|
"""
    for policy in POLICIES:
        summary += (
            f"| {policy} | {fmt(agg, policy, 'ac_min_voltage')} | "
            f"{fmt(agg, policy, 'ac_mean_min_voltage')} | "
            f"{fmt(agg, policy, 'ac_voltage_violation_rate_percent', 3)} | "
            f"{fmt(agg, policy, 'ac_nonconverged_slots', 2)} |\n"
        )

    summary += f"""
## Proposed-Method Gate

- Proposed policy zero-AC-violation scenario rate: {100.0 * zero_ac:.1f}%.
- Proposed policy nonconverged AC slots: {nonconv:.0f}.

## Interpretation

The AC validator is still synthetic, but it directly tests the main theoretical
risk: whether a linear safety projection remains safe under a nonlinear radial
power-flow approximation. Any nonzero proposed-method AC violation would require
adding a tighter voltage reserve to the projection before making a final paper
claim.
"""
    (out / "ac_validation_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    out = PROJECT_ROOT / "results"
    out.mkdir(exist_ok=True)
    frames = []
    for feeder, cfg in robust.FEEDERS.items():
        for seed in robust.SEEDS:
            print(f"AC validating feeder={feeder} seed={seed}", flush=True)
            frames.append(run_one(feeder, cfg, seed))
    metrics = pd.concat(frames, ignore_index=True)
    agg = aggregate(metrics)
    metrics.to_csv(out / "ac_validation_metrics.csv", index=False)
    agg.to_csv(out / "ac_validation_aggregate.csv", index=False)
    write_summary(metrics, agg)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
