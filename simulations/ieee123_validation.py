"""IEEE 123-bus OpenDSS replay for Grid-FedGAT-Safe schedules.

This validator uses the public IEEE 123-bus OpenDSS model cached under
``data/ieee123``. It replays the same EV charging schedules used in the robust
synthetic validation by injecting eight three-phase EV loads into downstream
IEEE 123-bus locations, then checks unbalanced AC voltages with OpenDSSDirect.

Outputs:
  results/ieee123_validation_metrics.csv
  results/ieee123_validation_aggregate.csv
  results/ieee123_validation_summary.md
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

import opendssdirect as dss

import pilot_grid_fedgat as sim
import robust_grid_fedgat as robust


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = PROJECT_ROOT / "data/ieee123"
MASTER_FILE = CASE_DIR / "IEEE123Master.dss"
STATION_BUSES = ["18", "35", "47", "60", "65", "76", "95", "101"]
POLICIES = [
    "uncontrolled",
    "edf_safe",
    "fedavg_local_safe",
    "fedavg_graph_safe",
    "grid_fedgat_pilot",
]
V_MIN_AC = 0.95
BASE_TARGET_MIN_VOLTAGE = 0.952
DEFAULT_LOAD_MULT = 0.95


def compile_case(load_mult: float) -> None:
    dss.Basic.ClearAll()
    dss.Text.Command(f"Compile [{MASTER_FILE.resolve()}]")
    dss.Solution.LoadMult(load_mult)
    dss.Solution.MaxIterations(80)
    dss.Solution.ControlMode(0)
    dss.Text.Command("Set mode=snapshot")


def solve_min_voltage() -> Tuple[float, bool]:
    dss.Solution.Solve()
    vals = [v for v in dss.Circuit.AllBusMagPu() if v > 1e-9]
    return float(min(vals)), bool(dss.Solution.Converged())


def make_ev_loads() -> None:
    for idx, bus in enumerate(STATION_BUSES):
        dss.Text.Command(
            f"New Load.EV{idx} Bus1={bus}.1.2.3 Phases=3 Conn=Wye Model=1 "
            "kV=4.16 kW=0 kvar=0"
        )


def set_ev_loads(ev_kw: np.ndarray) -> None:
    for idx, kw in enumerate(ev_kw):
        dss.Loads.Name(f"EV{idx}")
        dss.Loads.kW(float(kw))
        dss.Loads.kvar(0.0)


def base_load_multiplier() -> float:
    def feasible(mult: float) -> bool:
        compile_case(mult)
        min_v, conv = solve_min_voltage()
        return conv and min_v >= BASE_TARGET_MIN_VOLTAGE

    lo, hi = 0.25, DEFAULT_LOAD_MULT
    if feasible(hi):
        return hi
    for _ in range(24):
        mid = (lo + hi) / 2.0
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return lo


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


def dss_ac_filter(ev_kw: np.ndarray) -> Tuple[np.ndarray, float, float, bool, float]:
    start = perf_counter()
    set_ev_loads(ev_kw)
    min_v, conv = solve_min_voltage()
    if conv and min_v >= V_MIN_AC - 1e-6:
        return ev_kw, 1.0, min_v, True, 1000.0 * (perf_counter() - start)

    lo, hi = 0.0, 1.0
    best_min_v = -np.inf
    best_conv = False
    for _ in range(24):
        mid = (lo + hi) / 2.0
        set_ev_loads(mid * ev_kw)
        min_v, conv = solve_min_voltage()
        if conv and min_v >= V_MIN_AC - 1e-6:
            lo = mid
            best_min_v = min_v
            best_conv = conv
        else:
            hi = mid
    return lo * ev_kw, lo, best_min_v, best_conv, 1000.0 * (perf_counter() - start)


def validate_policy(
    policy: str,
    test_days: Iterable[int],
    w_local: np.ndarray,
    w_graph: np.ndarray,
    w_prop: np.ndarray,
    load_mult: float,
) -> Dict[str, float | str]:
    compile_case(load_mult)
    make_ev_loads()
    min_v = []
    violations = 0
    samples = 0
    nonconverged = 0
    delivered_before_filter = 0.0
    delivered_after_filter = 0.0
    filter_alpha_sum = 0.0
    filter_runtime_ms = 0.0
    filter_count = 0
    for day in test_days:
        _base, schedule = schedule_policy_day(policy, day, w_local, w_graph, w_prop)
        for t in range(sim.N_SLOTS):
            ev_kw = schedule[t].copy()
            delivered_before_filter += float(ev_kw.sum()) * sim.DT_HOURS
            if policy != "uncontrolled":
                ev_kw, alpha, _filter_min_v, _filter_conv, elapsed_ms = dss_ac_filter(ev_kw)
                filter_alpha_sum += alpha
                filter_runtime_ms += elapsed_ms
                filter_count += 1
            delivered_after_filter += float(ev_kw.sum()) * sim.DT_HOURS
            set_ev_loads(ev_kw)
            slot_min_v, conv = solve_min_voltage()
            min_v.append(slot_min_v)
            all_v = [v for v in dss.Circuit.AllBusMagPu() if v > 1e-9]
            violations += int(sum(v < V_MIN_AC - 1e-6 for v in all_v))
            samples += len(all_v)
            nonconverged += int(not conv)
    return {
        "policy": policy,
        "ieee123_min_voltage": float(np.min(min_v)),
        "ieee123_mean_min_voltage": float(np.mean(min_v)),
        "ieee123_violation_rate_percent": 100.0 * violations / max(samples, 1),
        "ieee123_nonconverged_slots": nonconverged,
        "ieee123_load_multiplier": load_mult,
        "ac_filter_energy_retained_percent": 100.0 * delivered_after_filter / max(delivered_before_filter, 1e-9),
        "ac_filter_mean_alpha": filter_alpha_sum / max(filter_count, 1),
        "ac_filter_runtime_ms_mean": filter_runtime_ms / max(filter_count, 1),
    }


def run_one(feeder: str, cfg: Dict[str, float], seed: int, load_mult: float) -> pd.DataFrame:
    robust.configure(seed, cfg)
    w_local, w_graph, w_prop = train_models(range(robust.TRAIN_DAYS))
    rows = [
        validate_policy(policy, range(robust.TEST_DAYS), w_local, w_graph, w_prop, load_mult)
        for policy in POLICIES
    ]
    df = pd.DataFrame(rows)
    df.insert(0, "seed", seed)
    df.insert(0, "stress_profile", feeder)
    return df


def aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "ieee123_min_voltage",
        "ieee123_mean_min_voltage",
        "ieee123_violation_rate_percent",
        "ieee123_nonconverged_slots",
        "ac_filter_energy_retained_percent",
        "ac_filter_mean_alpha",
        "ac_filter_runtime_ms_mean",
    ]
    out = metrics.groupby("policy")[cols].agg(["mean", "std", "min", "max"]).reset_index()
    out.columns = ["_".join(c).strip("_") for c in out.columns.to_flat_index()]
    return out


def fmt(row: pd.Series, metric: str, digits: int = 4) -> str:
    return f"{row[f'{metric}_mean']:.{digits}f} +/- {row[f'{metric}_std']:.{digits}f}"


def write_summary(metrics: pd.DataFrame, agg: pd.DataFrame, load_mult: float) -> None:
    out = PROJECT_ROOT / "results"
    prop = metrics[metrics["policy"] == "grid_fedgat_pilot"]
    zero_rate = float((prop["ieee123_violation_rate_percent"] <= 1e-9).mean())
    summary = f"""# IEEE 123-Bus OpenDSS AC Replay Summary

Generated by `simulations/ieee123_validation.py`.

The validation uses the public IEEE 123-bus OpenDSS model cached under
`data/ieee123`, maps eight charging stations to downstream three-phase buses
{', '.join(STATION_BUSES)}, calibrates the base load multiplier to `{load_mult:.4f}`,
and replays EV schedules with OpenDSSDirect.

## Aggregate Metrics

| Policy | Minimum voltage | Mean slot-min voltage | Voltage violations (%) | Energy retained after AC filter (%) | Mean AC filter alpha | AC filter ms/call |
|---|---:|---:|---:|---:|---:|---:|
"""
    for _, row in agg.iterrows():
        runtime = "-"
        if row["policy"] != "uncontrolled":
            runtime = fmt(row, "ac_filter_runtime_ms_mean", 3)
        summary += (
            f"| {row['policy']} | {fmt(row, 'ieee123_min_voltage')} | "
            f"{fmt(row, 'ieee123_mean_min_voltage')} | "
            f"{fmt(row, 'ieee123_violation_rate_percent', 3)} | "
            f"{fmt(row, 'ac_filter_energy_retained_percent', 2)} | "
            f"{fmt(row, 'ac_filter_mean_alpha', 4)} | {runtime} |\n"
        )
    summary += f"""
## Proposed-Method Gate

- Proposed policy zero-violation scenario rate on IEEE 123-bus replay: {100.0 * zero_rate:.1f}%.

## Interpretation

This validation uses synthetic EV sessions, but the feeder topology and
unbalanced AC solver are public and substantially larger than the MATPOWER
33/69-bus radial replay. The AC filter is applied directly through OpenDSS
solves, so the result is a stronger public-feeder safety check than the
single-phase MATPOWER replay.
"""
    (out / "ieee123_validation_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    out = PROJECT_ROOT / "results"
    out.mkdir(exist_ok=True)
    load_mult = base_load_multiplier()
    frames = []
    for feeder, cfg in robust.FEEDERS.items():
        for seed in robust.SEEDS:
            print(f"IEEE123 validating profile={feeder} seed={seed}", flush=True)
            frames.append(run_one(feeder, cfg, seed, load_mult))
    metrics = pd.concat(frames, ignore_index=True)
    agg = aggregate(metrics)
    metrics.to_csv(out / "ieee123_validation_metrics.csv", index=False)
    agg.to_csv(out / "ieee123_validation_aggregate.csv", index=False)
    write_summary(metrics, agg, load_mult)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
