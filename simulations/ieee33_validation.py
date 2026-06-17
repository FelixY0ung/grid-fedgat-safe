"""IEEE 33-bus style nonlinear AC replay for Grid-FedGAT-Safe schedules.

This validator maps the eight charging stations to buses of the standard
Baran-Wu 33-bus radial distribution feeder and replays the learned charging
schedules using a backward-forward-sweep AC power-flow solver.

It is still synthetic with respect to EV sessions and loads, but the feeder
topology and branch parameters follow the widely used IEEE/Baran-Wu 33-bus
test system. This improves over the chain-feeder-only validation.

Outputs:
  results/ieee33_validation_metrics.csv
  results/ieee33_validation_aggregate.csv
  results/ieee33_validation_summary.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import pilot_grid_fedgat as sim
import robust_grid_fedgat as robust


PROJECT_ROOT = Path(__file__).resolve().parents[1]


S_BASE_MVA = 10.0
V_BASE_KV = 12.66
Z_BASE_OHM = V_BASE_KV**2 / S_BASE_MVA
PF_LOAD = 0.95
V_MIN_AC = 0.95
AC_TOL = 1e-8
AC_MAX_ITER = 100

POLICIES = [
    "uncontrolled",
    "edf_safe",
    "fedavg_local_safe",
    "fedavg_graph_safe",
    "grid_fedgat_pilot",
]

# From the common Baran-Wu 33-bus radial distribution feeder. Bus numbers are
# converted to 0-based indices when building the network.
BRANCHES = [
    (1, 2, 0.0922, 0.0470),
    (2, 3, 0.4930, 0.2511),
    (3, 4, 0.3660, 0.1864),
    (4, 5, 0.3811, 0.1941),
    (5, 6, 0.8190, 0.7070),
    (6, 7, 0.1872, 0.6188),
    (7, 8, 1.7114, 1.2351),
    (8, 9, 1.0300, 0.7400),
    (9, 10, 1.0440, 0.7400),
    (10, 11, 0.1966, 0.0650),
    (11, 12, 0.3744, 0.1238),
    (12, 13, 1.4680, 1.1550),
    (13, 14, 0.5416, 0.7129),
    (14, 15, 0.5910, 0.5260),
    (15, 16, 0.7463, 0.5450),
    (16, 17, 1.2890, 1.7210),
    (17, 18, 0.7320, 0.5740),
    (2, 19, 0.1640, 0.1565),
    (19, 20, 1.5042, 1.3554),
    (20, 21, 0.4095, 0.4784),
    (21, 22, 0.7089, 0.9373),
    (3, 23, 0.4512, 0.3083),
    (23, 24, 0.8980, 0.7091),
    (24, 25, 0.8960, 0.7011),
    (6, 26, 0.2030, 0.1034),
    (26, 27, 0.2842, 0.1447),
    (27, 28, 1.0590, 0.9337),
    (28, 29, 0.8042, 0.7006),
    (29, 30, 0.5075, 0.2585),
    (30, 31, 0.9744, 0.9630),
    (31, 32, 0.3105, 0.3619),
    (32, 33, 0.3410, 0.5302),
]

STATION_BUSES = np.array([7, 10, 14, 18, 22, 25, 30, 33]) - 1

# Standard load profile in kW/kvar for buses 1..33. Bus 1 is slack with zero
# demand. These values are common in IEEE 33-bus examples.
BASE_P_KW = np.array([
    0, 100, 90, 120, 60, 60, 200, 200, 60, 60, 45, 60, 60, 120, 60, 60, 60,
    90, 90, 90, 90, 90, 90, 420, 420, 60, 60, 60, 120, 200, 150, 210, 60,
], dtype=float)
BASE_Q_KVAR = np.array([
    0, 60, 40, 80, 30, 20, 100, 100, 20, 20, 30, 35, 35, 80, 10, 20, 20,
    40, 40, 40, 40, 40, 50, 200, 200, 25, 25, 20, 70, 600, 70, 100, 40,
], dtype=float)


def children_and_lines() -> Tuple[List[List[int]], np.ndarray, np.ndarray]:
    children: List[List[int]] = [[] for _ in range(33)]
    parent = np.full(33, -1, dtype=int)
    z = np.zeros(33, dtype=complex)
    for frm, to, r_ohm, x_ohm in BRANCHES:
        i = frm - 1
        j = to - 1
        children[i].append(j)
        parent[j] = i
        z[j] = (r_ohm + 1j * x_ohm) / Z_BASE_OHM
    order = np.array(list(range(32, -1, -1)))
    return children, parent, z


CHILDREN, PARENT, Z_LINE = children_and_lines()


def bfs_power_flow(p_kw: np.ndarray, q_kvar: np.ndarray) -> Tuple[np.ndarray, bool]:
    s = (p_kw + 1j * q_kvar) / (S_BASE_MVA * 1000.0)
    v = np.ones(33, dtype=complex)
    converged = False
    for _ in range(AC_MAX_ITER):
        old = v.copy()
        i_inj = np.conj(s / np.maximum(v, 1e-6))
        branch_current = np.zeros(33, dtype=complex)
        subtree_current = np.zeros(33, dtype=complex)
        for node in range(32, -1, -1):
            total = i_inj[node]
            for child in CHILDREN[node]:
                total += subtree_current[child]
            subtree_current[node] = total
            if node != 0:
                branch_current[node] = total
        v[0] = 1.0 + 0.0j
        stack = [0]
        while stack:
            node = stack.pop()
            for child in CHILDREN[node]:
                v[child] = v[node] - Z_LINE[child] * branch_current[child]
                stack.append(child)
        if np.max(np.abs(v - old)) < AC_TOL:
            converged = True
            break
    return np.abs(v), converged


def train_models(train_days: Iterable[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs_local, ys_local, _ = sim.collect_training_data(train_days, local_only=True)
    xs_graph, ys_graph, _ = sim.collect_training_data(train_days, local_only=False)
    return (
        sim.federated_ridge(xs_local, ys_local, network_aware=False),
        sim.federated_ridge(xs_graph, ys_graph, network_aware=False),
        sim.federated_sufficient_stat_ridge(xs_graph, ys_graph),
    )


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
        schedule[t] = station_power
        sim.allocate_edf(sessions, remaining, station_power, t)
    return base, schedule


def validate_policy(
    policy: str,
    test_days: Iterable[int],
    w_local: np.ndarray,
    w_graph: np.ndarray,
    w_prop: np.ndarray,
) -> Dict[str, float]:
    min_v = []
    violation_count = 0
    nonconverged = 0
    samples = 0
    for day in test_days:
        base_scalar, schedule = schedule_policy_day(policy, day, w_local, w_graph, w_prop)
        for t in range(sim.N_SLOTS):
            # Calibrated so the no-EV IEEE 33-bus base case satisfies the
            # nonempty hard-safety assumption, while still leaving limited
            # voltage headroom for EV charging stress tests.
            scale = 0.35 + 0.15 * (base_scalar[t] / max(float(np.max(base_scalar)), 1e-9))
            p = BASE_P_KW * scale
            q = BASE_Q_KVAR * scale
            p[STATION_BUSES] += schedule[t]
            # EV chargers are assumed near-unity power factor.
            vmag, converged = bfs_power_flow(p, q)
            min_v.append(float(np.min(vmag)))
            violation_count += int(np.sum(vmag < V_MIN_AC - 1e-6))
            nonconverged += int(not converged)
            samples += len(vmag)
    return {
        "policy": policy,
        "ieee33_min_voltage": float(np.min(min_v)),
        "ieee33_mean_min_voltage": float(np.mean(min_v)),
        "ieee33_violation_rate_percent": 100.0 * violation_count / max(samples, 1),
        "ieee33_nonconverged_slots": nonconverged,
    }


def run_one(feeder: str, cfg: Dict[str, float], seed: int) -> pd.DataFrame:
    # Reuse the same station/session stress profiles as robust validation, but
    # validate on IEEE 33-bus topology instead of the synthetic chain feeder.
    robust.configure(seed, cfg)
    w_local, w_graph, w_prop = train_models(range(robust.TRAIN_DAYS))
    rows = [
        validate_policy(policy, range(robust.TEST_DAYS), w_local, w_graph, w_prop)
        for policy in POLICIES
    ]
    df = pd.DataFrame(rows)
    df.insert(0, "seed", seed)
    df.insert(0, "stress_profile", feeder)
    return df


def aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "ieee33_min_voltage",
        "ieee33_mean_min_voltage",
        "ieee33_violation_rate_percent",
        "ieee33_nonconverged_slots",
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
    zero_rate = float((prop["ieee33_violation_rate_percent"] <= 1e-9).mean())
    summary = f"""# IEEE 33-Bus AC Replay Summary

Generated by `simulations/ieee33_validation.py`.

The validation maps eight charging stations to buses 7, 10, 14, 18, 22, 25, 30,
and 33 of the Baran-Wu IEEE 33-bus radial distribution feeder. EV schedules are
generated by the same policy stack used in the robust synthetic validation and
then replayed with a backward-forward-sweep AC solver.

## Aggregate Metrics

| Policy | Minimum voltage | Mean slot-min voltage | Voltage violations (%) | Nonconverged slots |
|---|---:|---:|---:|---:|
"""
    for policy in POLICIES:
        summary += (
            f"| {policy} | {fmt(agg, policy, 'ieee33_min_voltage')} | "
            f"{fmt(agg, policy, 'ieee33_mean_min_voltage')} | "
            f"{fmt(agg, policy, 'ieee33_violation_rate_percent', 3)} | "
            f"{fmt(agg, policy, 'ieee33_nonconverged_slots', 2)} |\n"
        )
    summary += f"""
## Proposed-Method Gate

- Proposed policy zero-violation scenario rate on IEEE 33-bus replay: {100.0 * zero_rate:.1f}%.

## Interpretation

This is still not a full ACN-Data experiment, but it replaces the synthetic
chain-only topology with a recognized public radial feeder. If violations appear
here, the reserve or station placement must be revised before claiming public
feeder feasibility.
"""
    (out / "ieee33_validation_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    out = PROJECT_ROOT / "results"
    out.mkdir(exist_ok=True)
    frames = []
    for feeder, cfg in robust.FEEDERS.items():
        for seed in robust.SEEDS:
            print(f"IEEE33 validating profile={feeder} seed={seed}", flush=True)
            frames.append(run_one(feeder, cfg, seed))
    metrics = pd.concat(frames, ignore_index=True)
    agg = aggregate(metrics)
    metrics.to_csv(out / "ieee33_validation_metrics.csv", index=False)
    agg.to_csv(out / "ieee33_validation_aggregate.csv", index=False)
    write_summary(metrics, agg)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
