"""MATPOWER public-feeder AC replay for Grid-FedGAT-Safe schedules.

This script fetches public MATPOWER radial distribution cases and replays the
learned EV charging schedules with a backward-forward-sweep AC solver.

Cases:
  - case33bw: Baran-Wu 33-bus distribution system
  - case69: 69-bus distribution system

Outputs:
  results/matpower_validation_metrics.csv
  results/matpower_validation_aggregate.csv
  results/matpower_validation_summary.md
"""

from __future__ import annotations

import re
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Tuple
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

import pilot_grid_fedgat as sim
import robust_grid_fedgat as robust


PROJECT_ROOT = Path(__file__).resolve().parents[1]


CASE_URLS = {
    "case33bw": "https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case33bw.m",
    "case69": "https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case69.m",
}
POLICIES = [
    "uncontrolled",
    "edf_safe",
    "fedavg_local_safe",
    "fedavg_graph_safe",
    "grid_fedgat_pilot",
]
AC_TOL = 1e-8
AC_MAX_ITER = 100
V_MIN_AC = 0.95
BASE_TARGET_MIN_VOLTAGE = 0.952


def fetch_case(case_name: str) -> str:
    cache_path = PROJECT_ROOT / "data" / f"{case_name}.m"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="ignore")
    req = Request(CASE_URLS[case_name], headers={"User-Agent": "Mozilla/5.0"})
    text = urlopen(req, timeout=60).read().decode("utf-8", "ignore")
    cache_path.parent.mkdir(exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    return text


def parse_scalar(text: str, name: str) -> float:
    match = re.search(rf"mpc\.{name}\s*=\s*([0-9.]+)\s*;", text)
    if not match:
        raise ValueError(f"Could not parse mpc.{name}")
    return float(match.group(1))


def parse_matrix(text: str, name: str) -> np.ndarray:
    match = re.search(rf"mpc\.{name}\s*=\s*\[(.*?)\];", text, flags=re.S)
    if not match:
        raise ValueError(f"Could not parse mpc.{name}")
    rows = []
    for raw in match.group(1).splitlines():
        line = raw.split("%", 1)[0].strip()
        if not line:
            continue
        line = line.rstrip(";")
        vals = [float(v) for v in line.split()]
        if vals:
            rows.append(vals)
    return np.array(rows, dtype=float)


def load_case(case_name: str) -> Dict[str, np.ndarray | float]:
    text = fetch_case(case_name)
    base_mva = parse_scalar(text, "baseMVA")
    bus = parse_matrix(text, "bus")
    branch = parse_matrix(text, "branch")
    if "case33bw" in case_name and bus[:, 2].max() > 10:
        # MATPOWER converts this case from kW/kVAr to MW/MVAr later in the file.
        bus[:, 2:4] /= 1000.0
    if "case69" in case_name and bus[:, 2].max() > 10:
        bus[:, 2:4] /= 1000.0
    # Distribution cases store branch impedances in ohms and convert to p.u.
    # at the end of the MATPOWER file. Replicate that conversion here.
    base_kv = bus[0, 9]
    z_base_ohm = (base_kv * 1e3) ** 2 / (base_mva * 1e6)
    branch[:, 2:4] /= z_base_ohm
    return {"base_mva": base_mva, "bus": bus, "branch": branch}


def active_radial_branches(branch: np.ndarray) -> np.ndarray:
    if branch.shape[1] >= 11:
        branch = branch[branch[:, 10] != 0]
    return branch


def orient_radial_tree(branch: np.ndarray, bus_count: int) -> Tuple[List[List[int]], np.ndarray, np.ndarray]:
    graph: List[List[Tuple[int, complex]]] = [[] for _ in range(bus_count)]
    for row in active_radial_branches(branch):
        i = int(row[0]) - 1
        j = int(row[1]) - 1
        z = complex(row[2], row[3])
        graph[i].append((j, z))
        graph[j].append((i, z))

    parent = np.full(bus_count, -1, dtype=int)
    z_to_parent = np.zeros(bus_count, dtype=complex)
    children: List[List[int]] = [[] for _ in range(bus_count)]
    queue = [0]
    parent[0] = 0
    for node in queue:
        for nbr, z in graph[node]:
            if parent[nbr] != -1:
                continue
            parent[nbr] = node
            z_to_parent[nbr] = z
            children[node].append(nbr)
            queue.append(nbr)
    if np.any(parent == -1):
        raise ValueError("Case is not connected from slack bus")
    return children, parent, z_to_parent


def bfs_power_flow(
    p_mw: np.ndarray,
    q_mvar: np.ndarray,
    base_mva: float,
    children: List[List[int]],
    z_to_parent: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    n = len(p_mw)
    s = (p_mw + 1j * q_mvar) / base_mva
    v = np.ones(n, dtype=complex)
    converged = False
    for _ in range(AC_MAX_ITER):
        old = v.copy()
        i_load = np.conj(s / np.maximum(v, 1e-6))
        subtree = np.zeros(n, dtype=complex)
        branch_current = np.zeros(n, dtype=complex)
        for node in range(n - 1, -1, -1):
            total = i_load[node]
            for child in children[node]:
                total += subtree[child]
            subtree[node] = total
            if node != 0:
                branch_current[node] = total

        v[0] = 1.0 + 0.0j
        stack = [0]
        while stack:
            node = stack.pop()
            for child in children[node]:
                v[child] = v[node] - z_to_parent[child] * branch_current[child]
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


def station_buses(bus_count: int) -> np.ndarray:
    if bus_count >= 69:
        return np.array([10, 20, 30, 40, 50, 55, 60, bus_count]) - 1
    return np.array([7, 10, 14, 18, 22, 25, 30, min(33, bus_count)]) - 1


def base_scale_for_case(
    bus: np.ndarray,
    base_mva: float,
    children: List[List[int]],
    z_to_parent: np.ndarray,
) -> float:
    p0 = bus[:, 2].copy()
    q0 = bus[:, 3].copy()

    def feasible(scale: float) -> bool:
        worst = 2.0
        for day in range(robust.TEST_DAYS):
            base = sim.base_load_profile(day + 1000)
            profile = 0.65 + 0.35 * base / max(float(np.max(base)), 1e-9)
            for factor in profile:
                vmag, conv = bfs_power_flow(p0 * scale * factor, q0 * scale * factor, base_mva, children, z_to_parent)
                if not conv:
                    return False
                worst = min(worst, float(np.min(vmag)))
        return worst >= BASE_TARGET_MIN_VOLTAGE

    lo, hi = 0.05, 1.0
    for _ in range(28):
        mid = (lo + hi) / 2.0
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return lo


def ac_feasibility_filter(
    p_base: np.ndarray,
    q_base: np.ndarray,
    ev_kw: np.ndarray,
    buses: np.ndarray,
    base_mva: float,
    children: List[List[int]],
    z_to_parent: np.ndarray,
) -> Tuple[np.ndarray, float, float, bool]:
    """Scale EV charging until nonlinear AC voltage constraints are met.

    The no-EV base case is calibrated to be feasible. Therefore alpha=0 is
    feasible, and bisection over alpha in [0, 1] finds the largest feasible
    fraction of the proposed EV charging vector.
    """

    def evaluate(alpha: float) -> Tuple[float, bool]:
        p = p_base.copy()
        p[buses] += alpha * ev_kw / 1000.0
        vmag, conv = bfs_power_flow(p, q_base, base_mva, children, z_to_parent)
        if not conv:
            return -np.inf, False
        return float(np.min(vmag)), True

    min_v_full, conv_full = evaluate(1.0)
    if conv_full and min_v_full >= V_MIN_AC - 1e-6:
        return ev_kw, 1.0, min_v_full, True

    lo, hi = 0.0, 1.0
    best_min_v, best_conv = evaluate(0.0)
    for _ in range(28):
        mid = (lo + hi) / 2.0
        min_v, conv = evaluate(mid)
        if conv and min_v >= V_MIN_AC - 1e-6:
            lo = mid
            best_min_v = min_v
            best_conv = conv
        else:
            hi = mid
    return lo * ev_kw, lo, best_min_v, best_conv


def validate_policy(
    case_name: str,
    case: Dict[str, np.ndarray | float],
    policy: str,
    test_days: Iterable[int],
    w_local: np.ndarray,
    w_graph: np.ndarray,
    w_prop: np.ndarray,
    load_scale: float,
) -> Dict[str, float | str]:
    bus = np.array(case["bus"], dtype=float)
    branch = np.array(case["branch"], dtype=float)
    base_mva = float(case["base_mva"])
    children, _parent, z_to_parent = orient_radial_tree(branch, len(bus))
    buses = station_buses(len(bus))
    p0 = bus[:, 2].copy()
    q0 = bus[:, 3].copy()
    min_v = []
    violations = 0
    nonconverged = 0
    samples = 0
    delivered_before_filter = 0.0
    delivered_after_filter = 0.0
    filter_alpha_sum = 0.0
    filter_count = 0
    filter_runtime_ms = 0.0
    for day in test_days:
        base_scalar, schedule = schedule_policy_day(policy, day, w_local, w_graph, w_prop)
        profile = 0.65 + 0.35 * base_scalar / max(float(np.max(base_scalar)), 1e-9)
        for t in range(sim.N_SLOTS):
            p = p0 * load_scale * profile[t]
            q = q0 * load_scale * profile[t]
            ev_kw = schedule[t].copy()
            delivered_before_filter += float(ev_kw.sum()) * sim.DT_HOURS
            if policy != "uncontrolled":
                start = perf_counter()
                ev_kw, alpha, _filter_min_v, _filter_conv = ac_feasibility_filter(
                    p,
                    q,
                    ev_kw,
                    buses,
                    base_mva,
                    children,
                    z_to_parent,
                )
                filter_runtime_ms += 1000.0 * (perf_counter() - start)
                filter_alpha_sum += alpha
                filter_count += 1
            delivered_after_filter += float(ev_kw.sum()) * sim.DT_HOURS
            p[buses] += ev_kw / 1000.0
            vmag, conv = bfs_power_flow(p, q, base_mva, children, z_to_parent)
            min_v.append(float(np.min(vmag)))
            violations += int(np.sum(vmag < V_MIN_AC - 1e-6))
            nonconverged += int(not conv)
            samples += len(vmag)
    return {
        "case": case_name,
        "policy": policy,
        "matpower_min_voltage": float(np.min(min_v)),
        "matpower_mean_min_voltage": float(np.mean(min_v)),
        "matpower_violation_rate_percent": 100.0 * violations / max(samples, 1),
        "matpower_nonconverged_slots": nonconverged,
        "load_scale": load_scale,
        "ac_filter_energy_retained_percent": 100.0 * delivered_after_filter / max(delivered_before_filter, 1e-9),
        "ac_filter_mean_alpha": filter_alpha_sum / max(filter_count, 1),
        "ac_filter_calls": filter_count,
        "ac_filter_runtime_ms_total": filter_runtime_ms,
        "ac_filter_runtime_ms_mean": filter_runtime_ms / max(filter_count, 1),
    }


def run_one(case_name: str, case: Dict[str, np.ndarray | float], stress: str, cfg: Dict[str, float], seed: int) -> pd.DataFrame:
    robust.configure(seed, cfg)
    branch = np.array(case["branch"], dtype=float)
    bus = np.array(case["bus"], dtype=float)
    children, _parent, z_to_parent = orient_radial_tree(branch, len(bus))
    load_scale = base_scale_for_case(bus, float(case["base_mva"]), children, z_to_parent)
    w_local, w_graph, w_prop = train_models(range(robust.TRAIN_DAYS))
    rows = [
        validate_policy(case_name, case, policy, range(robust.TEST_DAYS), w_local, w_graph, w_prop, load_scale)
        for policy in POLICIES
    ]
    df = pd.DataFrame(rows)
    df.insert(0, "seed", seed)
    df.insert(0, "stress_profile", stress)
    return df


def aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "matpower_min_voltage",
        "matpower_mean_min_voltage",
        "matpower_violation_rate_percent",
        "matpower_nonconverged_slots",
        "ac_filter_energy_retained_percent",
        "ac_filter_mean_alpha",
        "ac_filter_runtime_ms_mean",
    ]
    out = metrics.groupby(["case", "policy"])[cols].agg(["mean", "std", "min", "max"]).reset_index()
    out.columns = ["_".join(c).strip("_") for c in out.columns.to_flat_index()]
    return out


def fmt(row: pd.Series, metric: str, digits: int = 4) -> str:
    return f"{row[f'{metric}_mean']:.{digits}f} +/- {row[f'{metric}_std']:.{digits}f}"


def write_summary(metrics: pd.DataFrame, agg: pd.DataFrame) -> None:
    out = PROJECT_ROOT / "results"
    summary = """# MATPOWER Public-Feeder AC Replay Summary

Generated by `simulations/matpower_validation.py`.

The validation fetches public MATPOWER radial distribution cases `case33bw` and
`case69`, maps eight EV charging stations to downstream buses, calibrates each
case so the no-EV base operating point satisfies the hard-safety assumption, and
replays schedules with a backward-forward-sweep AC solver.

## Aggregate Metrics

| Case | Policy | Minimum voltage | Voltage violations (%) | Energy retained after AC filter (%) | Mean AC filter alpha | AC filter ms/call |
|---|---|---:|---:|---:|---:|---:|
"""
    for _, row in agg.iterrows():
        runtime = "-"
        if row["policy"] != "uncontrolled":
            runtime = fmt(row, "ac_filter_runtime_ms_mean", 3)
        summary += (
            f"| {row['case']} | {row['policy']} | "
            f"{fmt(row, 'matpower_min_voltage')} | "
            f"{fmt(row, 'matpower_violation_rate_percent', 3)} | "
            f"{fmt(row, 'ac_filter_energy_retained_percent', 2)} | "
            f"{fmt(row, 'ac_filter_mean_alpha', 4)} | "
            f"{runtime} |\n"
        )
    gates = []
    for case in sorted(metrics["case"].unique()):
        prop = metrics[(metrics["case"] == case) & (metrics["policy"] == "grid_fedgat_pilot")]
        gates.append((case, 100.0 * float((prop["matpower_violation_rate_percent"] <= 1e-9).mean())))
    summary += "\n## Proposed-Method Gate\n\n"
    for case, rate in gates:
        summary += f"- Proposed policy zero-violation scenario rate on `{case}`: {rate:.1f}%.\n"
    summary += """
## Interpretation

This still uses synthetic EV sessions, but it replaces synthetic-only feeder
topology with public MATPOWER radial distribution cases. The AC feasibility
filter scales EV charging toward zero only when nonlinear AC replay would
violate the voltage floor. Since the no-EV base case is calibrated feasible,
the filter always has a feasible fallback.
"""
    (out / "matpower_validation_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    out = PROJECT_ROOT / "results"
    out.mkdir(exist_ok=True)
    cases = {name: load_case(name) for name in CASE_URLS}
    frames = []
    for case_name, case in cases.items():
        for stress, cfg in robust.FEEDERS.items():
            for seed in robust.SEEDS:
                print(f"MATPOWER validating case={case_name} profile={stress} seed={seed}", flush=True)
                frames.append(run_one(case_name, case, stress, cfg, seed))
    metrics = pd.concat(frames, ignore_index=True)
    agg = aggregate(metrics)
    metrics.to_csv(out / "matpower_validation_metrics.csv", index=False)
    agg.to_csv(out / "matpower_validation_aggregate.csv", index=False)
    write_summary(metrics, agg)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
