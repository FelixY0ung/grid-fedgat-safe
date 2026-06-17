"""Public EV-session replay using the Palo Alto ChargePoint dataset.

This script replaces synthetic EV arrivals with public session records from
the City of Palo Alto "Electric Vehicle Charging Station Usage (July 2011 -
Dec 2020)" dataset. It maps the most active public charging stations to the
eight-station simulator, trains the same federated graph-feature policies on
public-session days, and evaluates held-out public-session days under the same
grid-safety projection.

Outputs:
  results/public_ev_metrics.csv
  results/public_ev_summary.md
  results/public_ev_window_metrics.csv
  results/public_ev_window_aggregate.csv
  results/public_ev_window_comparisons.csv
  results/public_ev_window_summary.md
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

import pilot_grid_fedgat as sim


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(
    os.environ.get(
        "PUBLIC_EV_DATA_PATH",
        str(PROJECT_ROOT / "data/public_ev/palo_alto_chargepoint.csv"),
    )
)
RESULTS_DIR = PROJECT_ROOT / "results"
TRAIN_DAYS = 18
TEST_DAYS = 8
PUBLIC_WINDOW_COUNT = 5
MIN_SESSIONS_PER_DAY = 18
MAX_ENERGY_KWH = 0.95 * sim.PORT_KW * sim.DT_HOURS * sim.N_SLOTS
POLICIES = [
    "uncontrolled",
    "edf_safe",
    "fedavg_local_safe",
    "fedavg_graph_safe",
    "grid_fedgat_no_guard",
    "grid_fedgat_pilot",
]
POWER_MULTIPLIERS = {
    "nominal_public": 1.0,
    "stressed_public_4x": 4.0,
}


@dataclass
class PublicDataset:
    sessions_by_day: Dict[pd.Timestamp, List[sim.Session]]
    days: List[pd.Timestamp]
    station_map: Dict[str, int]
    raw_rows: int
    usable_rows: int


def load_public_sessions() -> PublicDataset:
    usecols = ["Station Name", "Start Date", "End Date", "Energy (kWh)"]
    df = pd.read_csv(DATA_PATH, usecols=usecols, low_memory=False)
    raw_rows = len(df)
    df["start"] = pd.to_datetime(df["Start Date"], errors="coerce", format="mixed")
    df["end"] = pd.to_datetime(df["End Date"], errors="coerce", format="mixed")
    df["energy"] = pd.to_numeric(df["Energy (kWh)"], errors="coerce")
    df = df.dropna(subset=["Station Name", "start", "end", "energy"]).copy()
    df = df[(df["energy"] > 0.05) & (df["end"] > df["start"])]
    df["duration_slots"] = np.ceil((df["end"] - df["start"]).dt.total_seconds() / (sim.DT_HOURS * 3600)).astype(int)
    df = df[df["duration_slots"] >= 1]
    df["energy"] = np.minimum(df["energy"], sim.PORT_KW * sim.DT_HOURS * df["duration_slots"])
    df = df[(df["energy"] > 0.05) & (df["energy"] <= MAX_ENERGY_KWH)]

    top_stations = df["Station Name"].value_counts().head(sim.N_STATIONS).index.tolist()
    station_map = {name: idx for idx, name in enumerate(top_stations)}
    df = df[df["Station Name"].isin(station_map)].copy()
    df["day"] = df["start"].dt.floor("D")
    day_counts = df.groupby("day").size()
    selected_days = day_counts[day_counts >= MIN_SESSIONS_PER_DAY].index.sort_values().tolist()
    if len(selected_days) < TRAIN_DAYS + TEST_DAYS:
        raise RuntimeError("Not enough public EV-session days after filtering")
    df = df[df["day"].isin(selected_days)].copy()

    sessions_by_day: Dict[pd.Timestamp, List[sim.Session]] = {}
    for day, group in df.groupby("day"):
        sessions: List[sim.Session] = []
        for _, row in group.iterrows():
            arrival = int(np.floor((row["start"] - day).total_seconds() / (sim.DT_HOURS * 3600)))
            departure = int(np.ceil((row["end"] - day).total_seconds() / (sim.DT_HOURS * 3600)))
            arrival = int(np.clip(arrival, 0, sim.N_SLOTS - 1))
            departure = int(np.clip(departure, arrival + 1, sim.N_SLOTS))
            max_energy = sim.PORT_KW * sim.DT_HOURS * (departure - arrival)
            energy = float(np.clip(row["energy"], 0.05, max_energy))
            sessions.append(sim.Session(station=station_map[row["Station Name"]], arrival=arrival, departure=departure, energy_kwh=energy))
        sessions_by_day[day] = sessions

    return PublicDataset(
        sessions_by_day=sessions_by_day,
        days=selected_days,
        station_map=station_map,
        raw_rows=raw_rows,
        usable_rows=len(df),
    )


def solve_public_oracle(sessions: List[sim.Session], base: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return sim.solve_oracle_lp(sessions, base)


def collect_public_training_data(
    dataset: PublicDataset,
    days: Iterable[pd.Timestamp],
    local_only: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    per_station_x = [[] for _ in range(sim.N_STATIONS)]
    per_station_y = [[] for _ in range(sim.N_STATIONS)]
    for idx, day in enumerate(days):
        base = sim.base_load_profile(3000 + idx)
        sessions = dataset.sessions_by_day[day]
        oracle_power, session_power = solve_public_oracle(sessions, base)
        remaining = np.array([s.energy_kwh for s in sessions])
        for t in range(sim.N_SLOTS):
            for station in range(sim.N_STATIONS):
                x = sim.active_state_features(sessions, remaining, station, t, base, local_only)
                per_station_x[station].append(x)
                per_station_y[station].append(oracle_power[station, t])
            remaining -= session_power[:, t] * sim.DT_HOURS
            remaining = np.maximum(remaining, 0.0)
    xs = [np.vstack(v) for v in per_station_x]
    ys = [np.array(v) for v in per_station_y]
    return xs, ys


def collect_public_training_data_pair(
    dataset: PublicDataset,
    days: Iterable[pd.Timestamp],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    per_station_x_local = [[] for _ in range(sim.N_STATIONS)]
    per_station_x_graph = [[] for _ in range(sim.N_STATIONS)]
    per_station_y = [[] for _ in range(sim.N_STATIONS)]
    for idx, day in enumerate(days):
        base = sim.base_load_profile(3000 + idx)
        sessions = dataset.sessions_by_day[day]
        oracle_power, session_power = solve_public_oracle(sessions, base)
        remaining = np.array([s.energy_kwh for s in sessions])
        for t_slot in range(sim.N_SLOTS):
            for station in range(sim.N_STATIONS):
                per_station_x_local[station].append(
                    sim.active_state_features(sessions, remaining, station, t_slot, base, local_only=True)
                )
                per_station_x_graph[station].append(
                    sim.active_state_features(sessions, remaining, station, t_slot, base, local_only=False)
                )
                per_station_y[station].append(oracle_power[station, t_slot])
            remaining -= session_power[:, t_slot] * sim.DT_HOURS
            remaining = np.maximum(remaining, 0.0)
    xs_local = [np.vstack(v) for v in per_station_x_local]
    xs_graph = [np.vstack(v) for v in per_station_x_graph]
    ys_local = [np.array(v) for v in per_station_y]
    ys_graph = [np.array(v) for v in per_station_y]
    return xs_local, ys_local, xs_graph, ys_graph


def train_public_models(dataset: PublicDataset, train_days: Iterable[pd.Timestamp]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs_local, ys_local, xs_graph, ys_graph = collect_public_training_data_pair(dataset, train_days)
    return (
        sim.federated_ridge(xs_local, ys_local, network_aware=False),
        sim.federated_ridge(xs_graph, ys_graph, network_aware=False),
        sim.federated_sufficient_stat_ridge(xs_graph, ys_graph),
    )


def public_window_starts(dataset: PublicDataset, window_count: int) -> List[int]:
    total_days = TRAIN_DAYS + TEST_DAYS
    max_start = len(dataset.days) - total_days
    if max_start < 0:
        raise RuntimeError("Not enough public EV-session days for a validation window")
    if window_count <= 1:
        return [0]
    starts = sorted({int(round(v)) for v in np.linspace(0, max_start, window_count)})
    candidate = 0
    while len(starts) < window_count and candidate <= max_start:
        if candidate not in starts:
            starts.append(candidate)
        candidate += 1
    return sorted(starts[:window_count])


def run_public_window(dataset: PublicDataset, window_id: int, start_index: int) -> pd.DataFrame:
    train_days = dataset.days[start_index : start_index + TRAIN_DAYS]
    test_days = dataset.days[start_index + TRAIN_DAYS : start_index + TRAIN_DAYS + TEST_DAYS]
    w_local, w_graph, w_prop = train_public_models(dataset, train_days)
    rows = [
        evaluate_public_policy(dataset, scenario, multiplier, policy, test_days, w_local, w_graph, w_prop)
        for scenario, multiplier in POWER_MULTIPLIERS.items()
        for policy in POLICIES
    ]
    metrics = pd.DataFrame(rows)
    metrics.insert(0, "test_end", test_days[-1])
    metrics.insert(0, "test_start", test_days[0])
    metrics.insert(0, "train_end", train_days[-1])
    metrics.insert(0, "train_start", train_days[0])
    metrics.insert(0, "window_start_index", start_index)
    metrics.insert(0, "window_id", window_id)
    return metrics


def ci95(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    n = len(values)
    if n <= 1:
        return 0.0
    return float(student_t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n))


def summarize_window_metrics(metrics: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    numeric_cols = [
        "public_service_rate_percent",
        "public_unmet_kwh",
        "public_charging_cost",
        "public_peak_load_kw",
        "public_voltage_violation_rate_percent",
    ]
    grouped = metrics.groupby(["scenario", "policy"])[numeric_cols]
    aggregate = grouped.agg(["mean", "std", ci95]).reset_index()
    aggregate.columns = ["_".join(col).strip("_") for col in aggregate.columns.to_flat_index()]

    pivot = metrics.pivot_table(
        index=["window_id", "scenario"],
        columns="policy",
        values=[
            "public_service_rate_percent",
            "public_charging_cost",
            "public_voltage_violation_rate_percent",
        ],
    )
    comparisons = []
    prop = "grid_fedgat_pilot"
    graph = "fedavg_graph_safe"
    no_guard = "grid_fedgat_no_guard"
    uncontrolled = "uncontrolled"
    for idx in pivot.index:
        row = {"window_id": idx[0], "scenario": idx[1]}
        row["prop_service_minus_graph"] = (
            pivot.loc[idx, ("public_service_rate_percent", prop)]
            - pivot.loc[idx, ("public_service_rate_percent", graph)]
        )
        row["prop_service_minus_no_guard"] = (
            pivot.loc[idx, ("public_service_rate_percent", prop)]
            - pivot.loc[idx, ("public_service_rate_percent", no_guard)]
        )
        row["prop_cost_minus_graph_percent"] = 100.0 * (
            pivot.loc[idx, ("public_charging_cost", prop)]
            - pivot.loc[idx, ("public_charging_cost", graph)]
        ) / max(pivot.loc[idx, ("public_charging_cost", graph)], 1e-9)
        row["prop_violation_rate"] = pivot.loc[idx, ("public_voltage_violation_rate_percent", prop)]
        row["uncontrolled_violation_rate"] = pivot.loc[
            idx, ("public_voltage_violation_rate_percent", uncontrolled)
        ]
        comparisons.append(row)
    comparisons_df = pd.DataFrame(comparisons)
    gates = {
        "window_count": float(metrics["window_id"].nunique()),
        "scenario_count": float(len(comparisons_df)),
        "zero_violation_rate": float((comparisons_df["prop_violation_rate"] <= 1e-9).mean()),
        "service_win_vs_graph_rate": float((comparisons_df["prop_service_minus_graph"] > 0.0).mean()),
        "service_win_vs_no_guard_rate": float((comparisons_df["prop_service_minus_no_guard"] > 0.0).mean()),
        "mean_service_gain_vs_graph": float(comparisons_df["prop_service_minus_graph"].mean()),
        "mean_service_gain_vs_no_guard": float(comparisons_df["prop_service_minus_no_guard"].mean()),
        "mean_cost_delta_vs_graph_percent": float(comparisons_df["prop_cost_minus_graph_percent"].mean()),
        "cost_lower_than_graph_rate": float((comparisons_df["prop_cost_minus_graph_percent"] < 0.0).mean()),
        "mean_uncontrolled_violation_rate": float(comparisons_df["uncontrolled_violation_rate"].mean()),
    }
    return aggregate, comparisons_df, gates


def fmt_mean_ci(aggregate: pd.DataFrame, scenario: str, policy: str, metric: str, digits: int = 2) -> str:
    row = aggregate[(aggregate["scenario"] == scenario) & (aggregate["policy"] == policy)].iloc[0]
    return f"{row[f'{metric}_mean']:.{digits}f} +/- {row[f'{metric}_ci95']:.{digits}f}"


def write_window_summary(dataset: PublicDataset, metrics: pd.DataFrame) -> None:
    aggregate, comparisons, gates = summarize_window_metrics(metrics)
    aggregate.to_csv(RESULTS_DIR / "public_ev_window_aggregate.csv", index=False)
    comparisons.to_csv(RESULTS_DIR / "public_ev_window_comparisons.csv", index=False)

    summary = f"""# Repeated Public EV-Session Window Validation Summary

Generated by `simulations/public_ev_validation.py`.

Dataset: City of Palo Alto ChargePoint public CSV cached at
`data/public_ev/palo_alto_chargepoint.csv`.

Eligible mapped days with at least `{MIN_SESSIONS_PER_DAY}` sessions: `{len(dataset.days)}`.
Validation windows: `{int(gates['window_count'])}`, each with `{TRAIN_DAYS}` training days and `{TEST_DAYS}` held-out test days.

Values are mean +/- 95% confidence interval across validation windows.

## Aggregate Metrics

| Scenario | Policy | Service rate (%) | Unmet energy (kWh) | Charging cost | Peak load (kW) | Voltage violations (%) |
|---|---|---:|---:|---:|---:|---:|
"""
    for scenario in POWER_MULTIPLIERS:
        for policy in POLICIES:
            summary += (
                f"| {scenario} | {policy} | "
                f"{fmt_mean_ci(aggregate, scenario, policy, 'public_service_rate_percent')} | "
                f"{fmt_mean_ci(aggregate, scenario, policy, 'public_unmet_kwh')} | "
                f"{fmt_mean_ci(aggregate, scenario, policy, 'public_charging_cost')} | "
                f"{fmt_mean_ci(aggregate, scenario, policy, 'public_peak_load_kw')} | "
                f"{fmt_mean_ci(aggregate, scenario, policy, 'public_voltage_violation_rate_percent', 3)} |\n"
            )

    summary += f"""
## Acceptance-Gate Check

| Gate | Result |
|---|---:|
| Proposed zero-voltage-violation window-scenarios | {100.0 * gates['zero_violation_rate']:.1f}% |
| Proposed service win rate vs graph FedAvg safe | {100.0 * gates['service_win_vs_graph_rate']:.1f}% |
| Proposed service win rate vs no-guard ablation | {100.0 * gates['service_win_vs_no_guard_rate']:.1f}% |
| Mean proposed service gain vs graph FedAvg safe | {gates['mean_service_gain_vs_graph']:.2f} percentage points |
| Mean proposed service gain vs no-guard ablation | {gates['mean_service_gain_vs_no_guard']:.2f} percentage points |
| Mean proposed cost delta vs graph FedAvg safe | {gates['mean_cost_delta_vs_graph_percent']:.2f}% |
| Proposed cost lower than graph FedAvg safe | {100.0 * gates['cost_lower_than_graph_rate']:.1f}% |
| Mean uncontrolled voltage violation rate | {gates['mean_uncontrolled_violation_rate']:.3f}% |

## Window-Level Comparison

| Window | Scenario | Service gain vs graph | Service gain vs no guard | Cost delta vs graph (%) | Proposed violation (%) | Uncontrolled violation (%) |
|---:|---|---:|---:|---:|---:|---:|
"""
    for _, row in comparisons.sort_values(["window_id", "scenario"]).iterrows():
        summary += (
            f"| {int(row['window_id'])} | {row['scenario']} | "
            f"{row['prop_service_minus_graph']:.2f} | "
            f"{row['prop_service_minus_no_guard']:.2f} | "
            f"{row['prop_cost_minus_graph_percent']:.2f} | "
            f"{row['prop_violation_rate']:.3f} | "
            f"{row['uncontrolled_violation_rate']:.3f} |\n"
        )
    (RESULTS_DIR / "public_ev_window_summary.md").write_text(summary, encoding="utf-8")


def evaluate_public_policy(
    dataset: PublicDataset,
    scenario: str,
    power_multiplier: float,
    policy: str,
    days: Iterable[pd.Timestamp],
    w_local: np.ndarray,
    w_graph: np.ndarray,
    w_prop: np.ndarray,
) -> Dict[str, float | str]:
    total_required = 0.0
    total_unmet = 0.0
    total_cost = 0.0
    peak_load = 0.0
    voltage_violations = 0
    observed_points = 0
    for idx, day in enumerate(days):
        base = sim.base_load_profile(4000 + idx)
        sessions = dataset.sessions_by_day[day]
        total_required += sum(s.energy_kwh for s in sessions)
        remaining = np.array([s.energy_kwh for s in sessions])
        caps = sim.transformer_cap(base)
        margins = sim.voltage_margin(base)
        for t in range(sim.N_SLOTS):
            ub = sim.upper_bound_power(sessions, remaining, t)
            raw = sim.raw_policy(policy, sessions, remaining, t, base, w_local, w_graph, w_prop)
            if policy == "uncontrolled":
                station_power = np.minimum(raw * power_multiplier, ub * power_multiplier)
            else:
                station_power = sim.project_power(raw * power_multiplier, ub * power_multiplier, base[t], caps[t])
            total_cost += sim.PRICE[t] * float(station_power.sum()) * sim.DT_HOURS
            peak_load = max(peak_load, float(base[t] * 90.0 + station_power.sum()))
            drops = sim.SENS @ station_power
            voltage_violations += int(np.sum(drops > margins[:, t] + 1e-6))
            observed_points += sim.N_STATIONS
            sim.allocate_edf(sessions, remaining, station_power / power_multiplier, t)
        total_unmet += float(np.sum(np.maximum(remaining, 0.0)))
    service = 100.0 * (1.0 - total_unmet / max(total_required, 1e-9))
    return {
        "scenario": scenario,
        "power_multiplier": power_multiplier,
        "policy": policy,
        "public_energy_required_kwh": total_required,
        "public_unmet_kwh": total_unmet,
        "public_service_rate_percent": service,
        "public_charging_cost": total_cost,
        "public_peak_load_kw": peak_load,
        "public_voltage_violation_rate_percent": 100.0 * voltage_violations / max(observed_points, 1),
    }


def write_summary(dataset: PublicDataset, metrics: pd.DataFrame, train_days: List[pd.Timestamp], test_days: List[pd.Timestamp]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    nominal = metrics[metrics["scenario"] == "nominal_public"]
    stressed = metrics[metrics["scenario"] == "stressed_public_4x"]
    best = nominal.loc[nominal["policy"] == "grid_fedgat_pilot"].iloc[0]
    graph = nominal.loc[nominal["policy"] == "fedavg_graph_safe"].iloc[0]
    no_guard = nominal.loc[nominal["policy"] == "grid_fedgat_no_guard"].iloc[0]
    uncontrolled = nominal.loc[nominal["policy"] == "uncontrolled"].iloc[0]
    stressed_best = stressed.loc[stressed["policy"] == "grid_fedgat_pilot"].iloc[0]
    stressed_graph = stressed.loc[stressed["policy"] == "fedavg_graph_safe"].iloc[0]
    stressed_no_guard = stressed.loc[stressed["policy"] == "grid_fedgat_no_guard"].iloc[0]
    stressed_uncontrolled = stressed.loc[stressed["policy"] == "uncontrolled"].iloc[0]
    service_gain_graph = best["public_service_rate_percent"] - graph["public_service_rate_percent"]
    service_gain_guard = best["public_service_rate_percent"] - no_guard["public_service_rate_percent"]
    stressed_service_gain_graph = stressed_best["public_service_rate_percent"] - stressed_graph["public_service_rate_percent"]
    stressed_service_gain_guard = stressed_best["public_service_rate_percent"] - stressed_no_guard["public_service_rate_percent"]
    violation_drop = uncontrolled["public_voltage_violation_rate_percent"] - best["public_voltage_violation_rate_percent"]
    comm = sim.communication_accounting(11, 17)
    rel_map = dict(zip(comm["policy"], comm["relative_bytes"]))
    rel_map["grid_fedgat_no_guard"] = rel_map["grid_fedgat_pilot"]

    summary = f"""# Public EV-Session Validation Summary

Generated by `simulations/public_ev_validation.py`.

Dataset: City of Palo Alto ChargePoint "Electric Vehicle Charging Station Usage
(July 2011 - Dec 2020)" public CSV cached at `data/public_ev/palo_alto_chargepoint.csv`.

Rows in raw CSV: `{dataset.raw_rows}`.
Usable mapped rows in selected train/test days: `{dataset.usable_rows}`.
Training days: `{len(train_days)}` from `{train_days[0].date()}` to `{train_days[-1].date()}`.
Test days: `{len(test_days)}` from `{test_days[0].date()}` to `{test_days[-1].date()}`.

## Aggregate Metrics

| Scenario | Policy | Service rate (%) | Unmet energy (kWh) | Charging cost | Peak load (kW) | Voltage violations (%) | Relative training bytes |
|---|---|---:|---:|---:|---:|---:|---:|
"""
    for _, row in metrics.iterrows():
        rel = rel_map.get(row["policy"], 0.0)
        rel_text = f"{rel:.3f}" if rel > 0 else "-"
        summary += (
            f"| {row['scenario']} | {row['policy']} | {row['public_service_rate_percent']:.2f} | "
            f"{row['public_unmet_kwh']:.2f} | {row['public_charging_cost']:.2f} | "
            f"{row['public_peak_load_kw']:.2f} | {row['public_voltage_violation_rate_percent']:.2f} | {rel_text} |\n"
        )
    stress_drop = (
        stressed_uncontrolled["public_voltage_violation_rate_percent"]
        - stressed_best["public_voltage_violation_rate_percent"]
    )
    summary += f"""
## Interpretation

- Grid-FedGAT-Safe service gain vs graph FedAvg safe: {service_gain_graph:.2f} percentage points.
- Grid-FedGAT-Safe service gain vs no-guard ablation: {service_gain_guard:.2f} percentage points.
- Voltage-violation reduction vs uncontrolled: {violation_drop:.2f} percentage points.
- Under the 4x public-session stress replay, service gain vs graph FedAvg safe is {stressed_service_gain_graph:.2f} percentage points.
- Under the 4x public-session stress replay, service gain vs no-guard ablation is {stressed_service_gain_guard:.2f} percentage points.
- Under the 4x public-session stress replay, voltage-violation reduction vs uncontrolled is {stress_drop:.2f} percentage points.
- This replay uses real public EV session arrivals, departures, station IDs, and delivered energy, while retaining the same synthetic feeder and communication-accounting setup as the robust validation.
"""
    (RESULTS_DIR / "public_ev_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    dataset = load_public_sessions()
    window_starts = public_window_starts(dataset, PUBLIC_WINDOW_COUNT)
    window_metrics = pd.concat(
        [run_public_window(dataset, window_id, start) for window_id, start in enumerate(window_starts)],
        ignore_index=True,
    )
    window_metrics.to_csv(RESULTS_DIR / "public_ev_window_metrics.csv", index=False)
    write_window_summary(dataset, window_metrics)

    metrics = window_metrics[window_metrics["window_id"] == 0].drop(
        columns=["window_id", "window_start_index", "train_start", "train_end", "test_start", "test_end"]
    )
    metrics.to_csv(RESULTS_DIR / "public_ev_metrics.csv", index=False)
    pd.DataFrame(
        [{"station_name": name, "mapped_station": idx} for name, idx in dataset.station_map.items()]
    ).to_csv(RESULTS_DIR / "public_ev_station_map.csv", index=False)
    train_days = dataset.days[window_starts[0] : window_starts[0] + TRAIN_DAYS]
    test_days = dataset.days[window_starts[0] + TRAIN_DAYS : window_starts[0] + TRAIN_DAYS + TEST_DAYS]
    write_summary(dataset, metrics, train_days, test_days)
    print(metrics.to_string(index=False))
    print(window_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
