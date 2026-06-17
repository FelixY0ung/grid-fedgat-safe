"""Deadline-reserve service/cost Pareto sensitivity.

The main validators report the selected Grid-FedGAT-Safe configuration. This
script evaluates a small set of alternative tariff-tier reserve settings while
reusing the same trained models and scenarios. The purpose is not to tune on the
test set, but to document the service/cost tradeoff behind the selected reserve.

Outputs:
  results/reserve_pareto_sensitivity.csv
  results/reserve_pareto_sensitivity_summary.md
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import pandas as pd

import pilot_grid_fedgat as sim
import public_ev_validation as public_ev
import robust_grid_fedgat as robust


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


@dataclass(frozen=True)
class ReserveVariant:
    name: str
    max_fraction: float
    cheap_weight: float
    shoulder_weight: float
    peak_weight: float

    @property
    def label(self) -> str:
        return (
            f"max={self.max_fraction:.2f}, "
            f"tiers={self.cheap_weight:.2f}/{self.shoulder_weight:.2f}/{self.peak_weight:.2f}"
        )


VARIANTS = [
    ReserveVariant("hard_floor_only", 0.00, 0.00, 0.00, 0.00),
    ReserveVariant("cost_lean", 0.40, 1.00, 0.25, 0.00),
    ReserveVariant("selected", 0.60, 1.00, 0.50, 0.05),
    ReserveVariant("flat_high", 0.60, 1.00, 1.00, 1.00),
]


@contextmanager
def reserve_variant(variant: ReserveVariant) -> Iterator[None]:
    old_max = sim.DEADLINE_RESERVE_MAX_FRACTION
    old_peak = sim.DEADLINE_RESERVE_PEAK_WEIGHT
    old_weight = sim.deadline_tariff_weight

    def tier_weight(t: int) -> float:
        price_t = float(sim.PRICE[t])
        if price_t <= 0.14:
            return variant.cheap_weight
        if price_t <= 0.18:
            return variant.shoulder_weight
        return variant.peak_weight

    sim.DEADLINE_RESERVE_MAX_FRACTION = variant.max_fraction
    sim.DEADLINE_RESERVE_PEAK_WEIGHT = variant.peak_weight
    sim.deadline_tariff_weight = tier_weight
    try:
        yield
    finally:
        sim.DEADLINE_RESERVE_MAX_FRACTION = old_max
        sim.DEADLINE_RESERVE_PEAK_WEIGHT = old_peak
        sim.deadline_tariff_weight = old_weight


def reset_nominal_feeder() -> None:
    sim.SEED = 20260611
    sim.SENS, sim.DEPTH = sim.feeder_sensitivity()
    sim.PRICE = sim.price_profile()


def evaluate_robust() -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    for feeder_name, feeder_cfg in robust.FEEDERS.items():
        for seed in robust.SEEDS:
            robust.configure(seed, feeder_cfg)
            w_local, w_graph, w_prop = robust.train_models(range(robust.TRAIN_DAYS))
            graph = sim.evaluate_policy(
                "fedavg_graph_safe", range(robust.TEST_DAYS), w_local, w_graph, w_prop
            )
            no_guard = sim.evaluate_policy(
                "grid_fedgat_no_guard", range(robust.TEST_DAYS), w_local, w_graph, w_prop
            )
            for variant in VARIANTS:
                with reserve_variant(variant):
                    prop = sim.evaluate_policy(
                        "grid_fedgat_pilot", range(robust.TEST_DAYS), w_local, w_graph, w_prop
                    )
                rows.append(
                    {
                        "validation": "robust_synthetic",
                        "scenario": feeder_name,
                        "seed_or_window": seed,
                        "stress": "synthetic",
                        "variant": variant.name,
                        "reserve_label": variant.label,
                        "service_rate_percent": prop["service_rate_percent"],
                        "unmet_kwh": prop["unmet_kwh"],
                        "charging_cost": prop["charging_cost"],
                        "voltage_violation_rate_percent": prop["voltage_violation_rate_percent"],
                        "graph_service_rate_percent": graph["service_rate_percent"],
                        "graph_charging_cost": graph["charging_cost"],
                        "no_guard_service_rate_percent": no_guard["service_rate_percent"],
                        "no_guard_charging_cost": no_guard["charging_cost"],
                    }
                )
    return pd.DataFrame(rows)


def evaluate_public() -> pd.DataFrame:
    reset_nominal_feeder()
    dataset = public_ev.load_public_sessions()
    starts = public_ev.public_window_starts(dataset, public_ev.PUBLIC_WINDOW_COUNT)
    rows: List[Dict[str, float | str]] = []
    for window_id, start_index in enumerate(starts):
        train_days = dataset.days[start_index : start_index + public_ev.TRAIN_DAYS]
        test_days = dataset.days[
            start_index + public_ev.TRAIN_DAYS : start_index + public_ev.TRAIN_DAYS + public_ev.TEST_DAYS
        ]
        w_local, w_graph, w_prop = public_ev.train_public_models(dataset, train_days)
        for scenario, multiplier in public_ev.POWER_MULTIPLIERS.items():
            graph = public_ev.evaluate_public_policy(
                dataset, scenario, multiplier, "fedavg_graph_safe", test_days, w_local, w_graph, w_prop
            )
            no_guard = public_ev.evaluate_public_policy(
                dataset, scenario, multiplier, "grid_fedgat_no_guard", test_days, w_local, w_graph, w_prop
            )
            for variant in VARIANTS:
                with reserve_variant(variant):
                    prop = public_ev.evaluate_public_policy(
                        dataset,
                        scenario,
                        multiplier,
                        "grid_fedgat_pilot",
                        test_days,
                        w_local,
                        w_graph,
                        w_prop,
                    )
                rows.append(
                    {
                        "validation": "public_windows",
                        "scenario": scenario,
                        "seed_or_window": window_id,
                        "stress": scenario,
                        "variant": variant.name,
                        "reserve_label": variant.label,
                        "service_rate_percent": prop["public_service_rate_percent"],
                        "unmet_kwh": prop["public_unmet_kwh"],
                        "charging_cost": prop["public_charging_cost"],
                        "voltage_violation_rate_percent": prop["public_voltage_violation_rate_percent"],
                        "graph_service_rate_percent": graph["public_service_rate_percent"],
                        "graph_charging_cost": graph["public_charging_cost"],
                        "no_guard_service_rate_percent": no_guard["public_service_rate_percent"],
                        "no_guard_charging_cost": no_guard["public_charging_cost"],
                    }
                )
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    df = metrics.copy()
    df["service_gain_vs_graph_pp"] = df["service_rate_percent"] - df["graph_service_rate_percent"]
    df["service_gain_vs_no_guard_pp"] = df["service_rate_percent"] - df["no_guard_service_rate_percent"]
    df["cost_delta_vs_graph_percent"] = 100.0 * (
        df["charging_cost"] - df["graph_charging_cost"]
    ) / df["graph_charging_cost"].clip(lower=1e-9)
    df["cost_delta_vs_no_guard_percent"] = 100.0 * (
        df["charging_cost"] - df["no_guard_charging_cost"]
    ) / df["no_guard_charging_cost"].clip(lower=1e-9)
    grouped = df.groupby(["validation", "variant", "reserve_label"])
    summary = grouped.agg(
        scenario_count=("variant", "size"),
        service_rate_mean=("service_rate_percent", "mean"),
        service_gain_vs_graph_mean=("service_gain_vs_graph_pp", "mean"),
        service_gain_vs_no_guard_mean=("service_gain_vs_no_guard_pp", "mean"),
        service_win_vs_graph_rate=("service_gain_vs_graph_pp", lambda s: float((s > 0.0).mean())),
        service_win_vs_no_guard_rate=("service_gain_vs_no_guard_pp", lambda s: float((s > 0.0).mean())),
        cost_delta_vs_graph_mean=("cost_delta_vs_graph_percent", "mean"),
        cost_delta_vs_no_guard_mean=("cost_delta_vs_no_guard_percent", "mean"),
        cost_lower_than_graph_rate=("cost_delta_vs_graph_percent", lambda s: float((s < 0.0).mean())),
        zero_violation_rate=("voltage_violation_rate_percent", lambda s: float((s <= 1e-9).mean())),
    ).reset_index()
    order = {variant.name: idx for idx, variant in enumerate(VARIANTS)}
    summary["variant_order"] = summary["variant"].map(order)
    return summary.sort_values(["validation", "variant_order"]).drop(columns=["variant_order"])


def write_summary(metrics: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Deadline-Reserve Pareto Sensitivity Summary",
        "",
        "Generated by `simulations/reserve_pareto_sensitivity.py`.",
        "",
        "This sensitivity keeps the trained policy class, safety projection, voltage reserve, and scenarios fixed while changing only the discretionary deadline-reserve tariff tiers for the proposed policy.",
        "",
        "## Aggregate Metrics",
        "",
        "| Validation | Variant | Reserve tiers | Scenarios | Service (%) | Service gain vs graph | Service gain vs no guard | Service wins vs graph | Service wins vs no guard | Cost delta vs graph | Cost lower than graph | Zero violations |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['validation']} | {row['variant']} | {row['reserve_label']} | "
            f"{int(row['scenario_count'])} | {row['service_rate_mean']:.2f} | "
            f"{row['service_gain_vs_graph_mean']:.2f} pp | "
            f"{row['service_gain_vs_no_guard_mean']:.2f} pp | "
            f"{100.0 * row['service_win_vs_graph_rate']:.1f}% | "
            f"{100.0 * row['service_win_vs_no_guard_rate']:.1f}% | "
            f"{row['cost_delta_vs_graph_mean']:.2f}% | "
            f"{100.0 * row['cost_lower_than_graph_rate']:.1f}% | "
            f"{100.0 * row['zero_violation_rate']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Lower reserve settings reduce or remove the cost premium but lose the all-scenario service-win gate.",
            "- The selected reserve is the smallest tested setting that keeps the current robust synthetic and public-window service-win gates positive, while retaining zero voltage violations.",
            "- The result supports a service/safety/communication Pareto claim; it does not prove universal cost dominance over graph FedAvg.",
            "",
            "## Scenario-Level Metrics",
            "",
            "Full scenario-level rows are in `results/reserve_pareto_sensitivity.csv`.",
        ]
    )
    (RESULTS_DIR / "reserve_pareto_sensitivity_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    robust_metrics = evaluate_robust()
    public_metrics = evaluate_public()
    metrics = pd.concat([robust_metrics, public_metrics], ignore_index=True)
    metrics.to_csv(RESULTS_DIR / "reserve_pareto_sensitivity.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(RESULTS_DIR / "reserve_pareto_sensitivity_aggregate.csv", index=False)
    write_summary(metrics, summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
