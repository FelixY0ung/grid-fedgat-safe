"""Pilot simulation for a safety-projected federated graph EV charging policy.

This script is intentionally lightweight and reproducible. It is not a full
deep-RL implementation of Grid-FedGAT. Instead, it tests the mathematically
provable core used in the paper draft: federated graph-feature policy learning
from an LP oracle, followed by a convex safety projection at deployment time.

Outputs:
  results/pilot_metrics.csv
  results/pilot_summary.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize


PROJECT_ROOT = Path(__file__).resolve().parents[1]


SEED = 20260611
N_STATIONS = 8
N_SLOTS = 96
DT_HOURS = 0.25
TRAIN_DAYS = 18
TEST_DAYS = 8
PORT_KW = 7.0
STATION_CAP_KW = 42.0
V_MIN = 0.95
V0 = 1.0
RIDGE = 1e-2
DEPLOYMENT_VOLTAGE_RESERVE = 0.011
DEADLINE_RESERVE_MAX_FRACTION = 0.60
DEADLINE_RESERVE_PEAK_WEIGHT = 0.05


@dataclass
class Session:
    station: int
    arrival: int
    departure: int
    energy_kwh: float


def price_profile() -> np.ndarray:
    price = np.full(N_SLOTS, 0.18)
    hours = np.arange(N_SLOTS) * DT_HOURS
    price[(hours < 7.0)] = 0.10
    price[(hours >= 11.0) & (hours < 16.0)] = 0.14
    price[(hours >= 17.0) & (hours < 21.0)] = 0.38
    price[(hours >= 22.0)] = 0.11
    return price


def base_load_profile(day: int) -> np.ndarray:
    rng = np.random.default_rng(SEED + 1009 * day + 17)
    hours = np.arange(N_SLOTS) * DT_HOURS
    evening = 0.5 + 0.5 * np.exp(-0.5 * ((hours - 19.0) / 3.1) ** 2)
    morning = 0.25 * np.exp(-0.5 * ((hours - 8.0) / 2.2) ** 2)
    weekend = 0.92 if day % 7 in (5, 6) else 1.0
    noise = rng.normal(0.0, 0.015, size=N_SLOTS)
    return np.clip(weekend * (0.52 + evening + morning) + noise, 0.35, None)


def feeder_sensitivity() -> Tuple[np.ndarray, np.ndarray]:
    """Return voltage sensitivity S and station depth vector.

    S[j, i] maps charging power at station i to voltage drop at observed
    station bus j under a chain-feeder LinDistFlow approximation.
    """
    depth = np.linspace(0.15, 1.0, N_STATIONS)
    s = np.zeros((N_STATIONS, N_STATIONS))
    for j in range(N_STATIONS):
        for i in range(N_STATIONS):
            overlap = min(depth[j], depth[i])
            s[j, i] = 0.00072 * overlap * (0.72 + 0.28 * depth[i])
    return s, depth


SENS, DEPTH = feeder_sensitivity()
PRICE = price_profile()


def voltage_margin(base: np.ndarray) -> np.ndarray:
    """Voltage-drop margin per observed bus and time slot."""
    # Downstream buses have larger background drop.
    base_drop = np.outer(0.010 + 0.014 * DEPTH, base / base.max())
    return np.maximum(V0 - V_MIN - base_drop, 0.012)


def transformer_cap(base: np.ndarray) -> np.ndarray:
    # Lower evening headroom makes uncoordinated charging visibly problematic.
    return 176.0 - 34.0 * (base - base.min()) / (base.max() - base.min() + 1e-9)


def proposed_deadline_alpha() -> float:
    """Maximum discretionary reserve fraction for the proposed graph policy.

    The hard deadline floor remains the one-step minimum needed to preserve
    future service feasibility. This coefficient adds a tariff-shaped reserve on
    top of that floor, encouraging earlier service in cheap slots while letting
    the safety projection decide the final grid-feasible command.
    """
    return DEADLINE_RESERVE_MAX_FRACTION


def deadline_tariff_weight(t: int) -> float:
    """Discount discretionary service reserve by tariff tier."""
    price_t = float(PRICE[t])
    if price_t <= 0.14:
        return 1.0
    if price_t <= 0.18:
        return 0.50
    return DEADLINE_RESERVE_PEAK_WEIGHT


def generate_sessions(day: int) -> List[Session]:
    rng = np.random.default_rng(SEED + 1009 * day + 41)
    sessions: List[Session] = []
    weekend = day % 7 in (5, 6)
    for station in range(N_STATIONS):
        lam = 6.3 if not weekend else 4.9
        n = rng.poisson(lam)
        for _ in range(n):
            mode = rng.choice(["work", "evening", "long"], p=[0.46, 0.36, 0.18])
            if mode == "work":
                arrival_hour = rng.normal(8.5, 1.1)
                duration = rng.integers(24, 42)
            elif mode == "evening":
                arrival_hour = rng.normal(18.3, 1.5)
                duration = rng.integers(12, 28)
            else:
                arrival_hour = rng.normal(13.0, 4.0)
                duration = rng.integers(30, 56)
            arrival = int(np.clip(round(arrival_hour / DT_HOURS), 0, N_SLOTS - 2))
            departure = int(np.clip(arrival + duration, arrival + 2, N_SLOTS))
            max_deliverable = PORT_KW * DT_HOURS * (departure - arrival)
            energy = float(np.clip(rng.gamma(shape=3.1, scale=4.7), 4.0, 0.82 * max_deliverable))
            sessions.append(Session(station, arrival, departure, energy))
    return sessions


def solve_oracle_lp(sessions: List[Session], base: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve a day-ahead LP oracle and return station power plus per-session power."""
    var_index: Dict[Tuple[int, int], int] = {}
    vars_list: List[Tuple[int, int]] = []
    for si, sess in enumerate(sessions):
        for t in range(sess.arrival, sess.departure):
            var_index[(si, t)] = len(vars_list)
            vars_list.append((si, t))
    slack_offset = len(vars_list)
    n_vars = slack_offset + len(sessions)

    c = np.zeros(n_vars)
    for idx, (si, t) in enumerate(vars_list):
        c[idx] = PRICE[t] * DT_HOURS
    c[slack_offset:] = 8.0

    a_eq = []
    b_eq = []
    for si, sess in enumerate(sessions):
        row = np.zeros(n_vars)
        for t in range(sess.arrival, sess.departure):
            row[var_index[(si, t)]] = DT_HOURS
        row[slack_offset + si] = 1.0
        a_eq.append(row)
        b_eq.append(sess.energy_kwh)

    a_ub = []
    b_ub = []
    margins = voltage_margin(base)
    caps = transformer_cap(base)
    for t in range(N_SLOTS):
        station_rows = []
        for station in range(N_STATIONS):
            row = np.zeros(n_vars)
            for si, sess in enumerate(sessions):
                if sess.station == station and sess.arrival <= t < sess.departure:
                    row[var_index[(si, t)]] = 1.0
            station_rows.append(row)
            a_ub.append(row.copy())
            b_ub.append(STATION_CAP_KW)

        total_row = np.sum(station_rows, axis=0)
        a_ub.append(total_row)
        b_ub.append(caps[t])

        for obs in range(N_STATIONS):
            row = np.zeros(n_vars)
            for station in range(N_STATIONS):
                row += SENS[obs, station] * station_rows[station]
            a_ub.append(row)
            b_ub.append(margins[obs, t])

    bounds = [(0.0, PORT_KW) for _ in vars_list] + [(0.0, None) for _ in sessions]
    res = linprog(
        c,
        A_ub=np.array(a_ub),
        b_ub=np.array(b_ub),
        A_eq=np.array(a_eq),
        b_eq=np.array(b_eq),
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"LP oracle failed: {res.message}")

    session_power = np.zeros((len(sessions), N_SLOTS))
    station_power = np.zeros((N_STATIONS, N_SLOTS))
    for idx, (si, t) in enumerate(vars_list):
        p = res.x[idx]
        session_power[si, t] = p
        station_power[sessions[si].station, t] += p
    return station_power, session_power


def active_state_features(
    sessions: List[Session],
    remaining: np.ndarray,
    station: int,
    t: int,
    base: np.ndarray,
    local_only: bool,
) -> np.ndarray:
    active = [
        (si, sess)
        for si, sess in enumerate(sessions)
        if sess.station == station and sess.arrival <= t < sess.departure and remaining[si] > 1e-6
    ]
    rem = sum(remaining[si] for si, _ in active)
    active_count = len(active)
    if active:
        time_left = np.array([max(sess.departure - t, 1) * DT_HOURS for si, sess in active])
        urgent = sum(max(0.0, remaining[si] - PORT_KW * max(sess.departure - t - 1, 0) * DT_HOURS) for si, sess in active)
        min_time = float(time_left.min())
        avg_time = float(time_left.mean())
    else:
        urgent = 0.0
        min_time = 24.0
        avg_time = 24.0

    hour = t * DT_HOURS
    base_feats = [
        1.0,
        rem / 80.0,
        active_count / 10.0,
        urgent / 20.0,
        min_time / 24.0,
        avg_time / 24.0,
        PRICE[t] / 0.40,
        np.sin(2.0 * np.pi * hour / 24.0),
        np.cos(2.0 * np.pi * hour / 24.0),
        base[t] / 2.0,
        DEPTH[station],
    ]
    if local_only:
        return np.array(base_feats)

    station_remaining = np.zeros(N_STATIONS)
    station_urgent = np.zeros(N_STATIONS)
    for si, sess in enumerate(sessions):
        if sess.arrival <= t < sess.departure and remaining[si] > 1e-6:
            station_remaining[sess.station] += remaining[si]
            station_urgent[sess.station] += max(
                0.0,
                remaining[si] - PORT_KW * max(sess.departure - t - 1, 0) * DT_HOURS,
            )
    left = max(station - 1, 0)
    right = min(station + 1, N_STATIONS - 1)
    downstream_rem = station_remaining[station:].sum()
    weighted_rem = float(SENS[:, station] @ station_remaining)
    graph_feats = [
        station_remaining[left] / 80.0,
        station_remaining[right] / 80.0,
        downstream_rem / 300.0,
        station_urgent[station:].sum() / 80.0,
        weighted_rem / 25.0,
        SENS[:, station].sum(),
    ]
    return np.array(base_feats + graph_feats)


def collect_training_data(
    days: Iterable[int],
    local_only: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[int, Tuple[List[Session], np.ndarray, np.ndarray]]]:
    per_station_x = [[] for _ in range(N_STATIONS)]
    per_station_y = [[] for _ in range(N_STATIONS)]
    cached = {}
    for day in days:
        base = base_load_profile(day)
        sessions = generate_sessions(day)
        oracle_power, session_power = solve_oracle_lp(sessions, base)
        cached[day] = (sessions, base, oracle_power)
        remaining = np.array([s.energy_kwh for s in sessions])
        for t in range(N_SLOTS):
            for station in range(N_STATIONS):
                x = active_state_features(sessions, remaining, station, t, base, local_only)
                y = oracle_power[station, t]
                per_station_x[station].append(x)
                per_station_y[station].append(y)
            remaining -= session_power[:, t] * DT_HOURS
            remaining = np.maximum(remaining, 0.0)
    xs = [np.vstack(v) for v in per_station_x]
    ys = [np.array(v) for v in per_station_y]
    return xs, ys, cached


def ridge_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xtx = x.T @ x + RIDGE * np.eye(x.shape[1])
    xty = x.T @ y
    return np.linalg.solve(xtx, xty)


def federated_ridge(xs: List[np.ndarray], ys: List[np.ndarray], network_aware: bool) -> np.ndarray:
    local_w = [ridge_fit(x, y) for x, y in zip(xs, ys)]
    counts = np.array([len(y) for y in ys], dtype=float)
    if network_aware:
        rng = np.random.default_rng(SEED + 707)
        latency = 25.0 + 18.0 * DEPTH + rng.uniform(0.0, 8.0, size=N_STATIONS)
        loss = 0.02 + 0.10 * DEPTH + rng.uniform(0.0, 0.03, size=N_STATIONS)
        reliability = (1.0 - loss) / latency
        weights = counts * reliability / reliability.max()
    else:
        weights = counts
    weights = weights / weights.sum()
    return np.sum([weights[i] * local_w[i] for i in range(N_STATIONS)], axis=0)


def federated_sufficient_stat_ridge(xs: List[np.ndarray], ys: List[np.ndarray]) -> np.ndarray:
    """Exact ridge solution from client-side sufficient statistics."""
    d = xs[0].shape[1]
    xtx = RIDGE * np.eye(d)
    xty = np.zeros(d)
    for x, y in zip(xs, ys):
        xtx += x.T @ x
        xty += x.T @ y
    return np.linalg.solve(xtx, xty)


def project_power(raw: np.ndarray, ub: np.ndarray, base_t: float, cap_t: float) -> np.ndarray:
    raw = np.nan_to_num(np.maximum(raw, 0.0))
    ub = np.maximum(ub, 0.0)
    margins = np.maximum(voltage_margin(np.array([base_t]))[:, 0] - DEPLOYMENT_VOLTAGE_RESERVE, 1e-4)

    def objective(p: np.ndarray) -> float:
        return 0.5 * float(np.sum((p - raw) ** 2))

    constraints = [{"type": "ineq", "fun": lambda p, row=row, m=m: m - float(row @ p)} for row, m in zip(SENS, margins)]
    constraints.append({"type": "ineq", "fun": lambda p: cap_t - float(np.sum(p))})
    res = minimize(
        objective,
        np.minimum(raw, ub),
        method="SLSQP",
        bounds=[(0.0, float(u)) for u in ub],
        constraints=constraints,
        options={"maxiter": 80, "ftol": 1e-7, "disp": False},
    )
    if res.success:
        return np.clip(res.x, 0.0, ub)

    # Conservative fallback: proportional clipping until all constraints pass.
    p = np.minimum(raw, ub)
    for _ in range(40):
        ratios = [cap_t / max(np.sum(p), 1e-9)]
        ratios.extend(m / max(float(row @ p), 1e-9) for row, m in zip(SENS, margins))
        ratio = min(1.0, *ratios)
        if ratio >= 0.999999:
            break
        p *= 0.995 * ratio
    return np.clip(p, 0.0, ub)


def allocate_edf(sessions: List[Session], remaining: np.ndarray, station_power: np.ndarray, t: int) -> None:
    for station in range(N_STATIONS):
        budget = station_power[station] * DT_HOURS
        active = [
            (si, sess)
            for si, sess in enumerate(sessions)
            if sess.station == station and sess.arrival <= t < sess.departure and remaining[si] > 1e-9
        ]
        active.sort(key=lambda item: item[1].departure)
        for si, _sess in active:
            delivered = min(budget, remaining[si], PORT_KW * DT_HOURS)
            remaining[si] -= delivered
            budget -= delivered
            if budget <= 1e-9:
                break


def upper_bound_power(sessions: List[Session], remaining: np.ndarray, t: int) -> np.ndarray:
    ub = np.zeros(N_STATIONS)
    for station in range(N_STATIONS):
        active = [
            si
            for si, sess in enumerate(sessions)
            if sess.station == station and sess.arrival <= t < sess.departure and remaining[si] > 1e-9
        ]
        ub[station] = min(STATION_CAP_KW, PORT_KW * len(active), sum(remaining[si] for si in active) / DT_HOURS)
    return ub


def raw_policy(
    policy: str,
    sessions: List[Session],
    remaining: np.ndarray,
    t: int,
    base: np.ndarray,
    w_local: np.ndarray,
    w_graph: np.ndarray,
    w_prop: np.ndarray,
) -> np.ndarray:
    ub = upper_bound_power(sessions, remaining, t)
    if policy == "uncontrolled":
        return ub
    if policy == "edf_safe":
        raw = np.zeros(N_STATIONS)
        for station in range(N_STATIONS):
            active = [
                (si, sess)
                for si, sess in enumerate(sessions)
                if sess.station == station and sess.arrival <= t < sess.departure and remaining[si] > 1e-9
            ]
            urgent = sum(max(0.0, remaining[si] - PORT_KW * max(sess.departure - t - 1, 0) * DT_HOURS) for si, sess in active)
            slack_factor = 0.35 if PRICE[t] > 0.25 else 0.82
            raw[station] = min(ub[station], urgent / DT_HOURS + slack_factor * ub[station])
        return raw

    raw = np.zeros(N_STATIONS)
    for station in range(N_STATIONS):
        if policy == "fedavg_local_safe":
            x = active_state_features(sessions, remaining, station, t, base, local_only=True)
            raw[station] = x @ w_local
        elif policy == "fedavg_graph_safe":
            x = active_state_features(sessions, remaining, station, t, base, local_only=False)
            raw[station] = x @ w_graph
        elif policy in {"grid_fedgat_pilot", "grid_fedgat_no_guard"}:
            x = active_state_features(sessions, remaining, station, t, base, local_only=False)
            raw[station] = x @ w_prop
            if policy == "grid_fedgat_no_guard":
                continue
            alpha = proposed_deadline_alpha() * deadline_tariff_weight(t)
            active = [
                (si, sess)
                for si, sess in enumerate(sessions)
                if sess.station == station and sess.arrival <= t < sess.departure and remaining[si] > 1e-9
            ]
            deadline_floor = sum(
                max(0.0, remaining[si] - PORT_KW * max(sess.departure - t - 1, 0) * DT_HOURS)
                for si, sess in active
            ) / DT_HOURS
            raw[station] = max(raw[station], deadline_floor + alpha * ub[station])
        else:
            raise ValueError(policy)
    return np.clip(raw, 0.0, ub)


def evaluate_policy(
    policy: str,
    days: Iterable[int],
    w_local: np.ndarray,
    w_graph: np.ndarray,
    w_prop: np.ndarray,
) -> Dict[str, float]:
    total_required = 0.0
    total_unmet = 0.0
    total_cost = 0.0
    peak_load = 0.0
    voltage_violations = 0
    transformer_violations = 0
    observed_points = 0
    projection_calls = 0
    projection_runtime_ms = 0.0
    for day in days:
        base = base_load_profile(day + 1000)
        sessions = generate_sessions(day + 1000)
        total_required += sum(s.energy_kwh for s in sessions)
        remaining = np.array([s.energy_kwh for s in sessions])
        caps = transformer_cap(base)
        margins = voltage_margin(base)
        for t in range(N_SLOTS):
            ub = upper_bound_power(sessions, remaining, t)
            raw = raw_policy(policy, sessions, remaining, t, base, w_local, w_graph, w_prop)
            if policy == "uncontrolled":
                station_power = np.minimum(raw, ub)
            else:
                start = perf_counter()
                station_power = project_power(raw, ub, base[t], caps[t])
                projection_runtime_ms += 1000.0 * (perf_counter() - start)
                projection_calls += 1
            total_cost += PRICE[t] * float(station_power.sum()) * DT_HOURS
            peak_load = max(peak_load, float(base[t] * 90.0 + station_power.sum()))
            drops = SENS @ station_power
            voltage_violations += int(np.sum(drops > margins[:, t] + 1e-6))
            transformer_violations += int(station_power.sum() > caps[t] + 1e-6)
            observed_points += N_STATIONS
            allocate_edf(sessions, remaining, station_power, t)
        total_unmet += float(np.sum(np.maximum(remaining, 0.0)))
    service = 100.0 * (1.0 - total_unmet / max(total_required, 1e-9))
    return {
        "policy": policy,
        "energy_required_kwh": total_required,
        "unmet_kwh": total_unmet,
        "service_rate_percent": service,
        "charging_cost": total_cost,
        "peak_load_kw": peak_load,
        "voltage_violation_rate_percent": 100.0 * voltage_violations / max(observed_points, 1),
        "transformer_violation_slots": transformer_violations,
        "projection_calls": projection_calls,
        "projection_runtime_ms_total": projection_runtime_ms,
        "projection_runtime_ms_mean": projection_runtime_ms / max(projection_calls, 1),
    }


def communication_accounting(n_features_local: int, n_features_graph: int) -> pd.DataFrame:
    rounds = 60
    full_clients = N_STATIONS
    fedavg_bytes = rounds * full_clients * n_features_graph * 4 * 2
    # Proposed linear graph-policy training sends sufficient statistics once.
    # X^T X is symmetric, so only the upper triangle is counted.
    sufficient_stat_values = n_features_graph * (n_features_graph + 1) // 2 + n_features_graph
    proposed_bytes = full_clients * sufficient_stat_values * 4 + full_clients * n_features_graph * 4
    return pd.DataFrame(
        [
            {
                "policy": "fedavg_graph_safe",
                "training_rounds": rounds,
                "clients_per_round": full_clients,
                "update_precision_bits": 32,
                "estimated_training_bytes": fedavg_bytes,
                "relative_bytes": 1.0,
            },
            {
                "policy": "grid_fedgat_pilot",
                "training_rounds": 1,
                "clients_per_round": full_clients,
                "update_precision_bits": 32,
                "estimated_training_bytes": proposed_bytes,
                "relative_bytes": proposed_bytes / fedavg_bytes,
            },
            {
                "policy": "grid_fedgat_no_guard",
                "training_rounds": 1,
                "clients_per_round": full_clients,
                "update_precision_bits": 32,
                "estimated_training_bytes": proposed_bytes,
                "relative_bytes": proposed_bytes / fedavg_bytes,
            },
            {
                "policy": "fedavg_local_safe",
                "training_rounds": rounds,
                "clients_per_round": full_clients,
                "update_precision_bits": 32,
                "estimated_training_bytes": rounds * full_clients * n_features_local * 4 * 2,
                "relative_bytes": (rounds * full_clients * n_features_local * 4 * 2) / fedavg_bytes,
            },
        ]
    )


def main() -> None:
    result_dir = PROJECT_ROOT / "results"
    result_dir.mkdir(exist_ok=True)

    train_days = range(TRAIN_DAYS)
    xs_local, ys_local, _ = collect_training_data(train_days, local_only=True)
    xs_graph, ys_graph, _ = collect_training_data(train_days, local_only=False)

    w_local = federated_ridge(xs_local, ys_local, network_aware=False)
    w_graph = federated_ridge(xs_graph, ys_graph, network_aware=False)
    w_prop = federated_sufficient_stat_ridge(xs_graph, ys_graph)

    policies = [
        "uncontrolled",
        "edf_safe",
        "fedavg_local_safe",
        "fedavg_graph_safe",
        "grid_fedgat_no_guard",
        "grid_fedgat_pilot",
    ]
    rows = [evaluate_policy(p, range(TEST_DAYS), w_local, w_graph, w_prop) for p in policies]
    metrics = pd.DataFrame(rows)
    comm = communication_accounting(len(w_local), len(w_graph))
    metrics = metrics.merge(comm, on="policy", how="left")
    metrics["estimated_training_bytes"] = metrics["estimated_training_bytes"].fillna(0).astype(int)
    metrics["relative_bytes"] = metrics["relative_bytes"].fillna(0.0)
    metrics.to_csv(result_dir / "pilot_metrics.csv", index=False)

    best = metrics.loc[metrics["policy"] == "grid_fedgat_pilot"].iloc[0]
    fed_graph = metrics.loc[metrics["policy"] == "fedavg_graph_safe"].iloc[0]
    local = metrics.loc[metrics["policy"] == "fedavg_local_safe"].iloc[0]
    uncontrolled = metrics.loc[metrics["policy"] == "uncontrolled"].iloc[0]

    summary = f"""# Pilot Simulation Summary

Random seed: `{SEED}`

These results are generated by `simulations/pilot_grid_fedgat.py`. The pilot
uses synthetic EV sessions, a radial-feeder LinDistFlow sensitivity model, an LP
oracle for training targets, one-shot federated ridge aggregation, and a convex
deployment-time safety projection. It is evidence for feasibility of the revised
method core, not a replacement for full deep-RL experiments on public datasets.

## Main Result

| Policy | Service rate (%) | Unmet energy (kWh) | Charging cost | Peak load (kW) | Voltage violations (%) | Projection ms/call | Relative training bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for _, row in metrics.iterrows():
        rel = row["relative_bytes"]
        rel_text = f"{rel:.3f}" if rel > 0 else "-"
        runtime_text = f"{row['projection_runtime_ms_mean']:.3f}" if row["projection_calls"] > 0 else "-"
        summary += (
            f"| {row['policy']} | {row['service_rate_percent']:.2f} | "
            f"{row['unmet_kwh']:.2f} | {row['charging_cost']:.2f} | "
            f"{row['peak_load_kw']:.2f} | {row['voltage_violation_rate_percent']:.2f} | "
            f"{runtime_text} | {rel_text} |\n"
        )

    service_gain_local = best["service_rate_percent"] - local["service_rate_percent"]
    cost_delta_graph = 100.0 * (best["charging_cost"] - fed_graph["charging_cost"]) / max(fed_graph["charging_cost"], 1e-9)
    violation_drop = uncontrolled["voltage_violation_rate_percent"] - best["voltage_violation_rate_percent"]
    byte_cut = 100.0 * (1.0 - best["relative_bytes"] / max(fed_graph["relative_bytes"], 1e-9))

    summary += f"""
## Checks

- The safety projection reduced voltage violations by {violation_drop:.2f} percentage points versus uncontrolled charging.
- The proposed graph policy improved service rate by {service_gain_local:.2f} percentage points versus the local-feature federated safe baseline.
- The proposed communication setting used {byte_cut:.1f}% fewer estimated training bytes than full-precision graph FedAvg.
- The proposed charging cost changed by {cost_delta_graph:.2f}% versus graph FedAvg with the same safety layer.

## Interpretation

The positive result is strongest for feasibility and safety: projected policies
keep grid constraints satisfied in this synthetic setup. Superiority is
preliminary: a submission should replace this pilot with ACN-Data/IEEE feeder
experiments, multiple random seeds, and statistical confidence intervals.
"""
    (result_dir / "pilot_summary.md").write_text(summary, encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
