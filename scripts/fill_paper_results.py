#!/usr/bin/env python3
"""

Usage:
    python scripts/fill_paper_results.py [--dry-run]
"""

import json
import os
import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

# Paths
BASE_DIR = Path(__file__).parent.parent
PAPER_DIR = BASE_DIR / "paper"
OUTPUTS = Path("/path/to/bamf-eco/outputs")


def load_optimizer_results():
    """Load BAMF-Eco optimizer results."""
    ckpt_file = OUTPUTS / "bamf_eco_main" / "checkpoint.json"
    if not ckpt_file.exists():
        print(f"  [SKIP] No optimizer checkpoint at {ckpt_file}")
        return None
    with open(ckpt_file) as f:
        return json.load(f)


def load_promotion_results():
    """Load promotion (Phase 2) results."""
    results_file = OUTPUTS / "bamf_eco_promotion" / "promotion_results.json"
    if not results_file.exists():
        print(f"  [SKIP] No promotion results at {results_file}")
        return None
    with open(results_file) as f:
        return json.load(f)


def load_baseline_results():
    """Load all baseline results."""
    baselines_dir = OUTPUTS / "baselines"
    results = {}
    for fname in baselines_dir.glob("*_checkpoint.json"):
        name = fname.stem.replace("_seed42_checkpoint", "")
        with open(fname) as f:
            data = json.load(f)
        results[name] = data
    return results


def load_benchmark_results():
    """Load EcoDetBench inference benchmark."""
    bench_file = OUTPUTS / "benchmark" / "ecodetbench_results.json"
    if not bench_file.exists():
        print(f"  [SKIP] No benchmark at {bench_file}")
        return None
    with open(bench_file) as f:
        return json.load(f)


def compute_summary(opt_ckpt, promo_results, baselines):
    """Compute summary statistics for the paper."""
    summary = {}

    # BAMF-Eco Phase 1 (low-fidelity exploration)
    state = opt_ckpt["state"]
    summary["bamf_phase1_evals"] = state["iteration"]
    summary["bamf_phase1_energy"] = state["total_energy_kwh"]
    summary["bamf_phase1_co2e"] = state["total_co2e_kg"]
    summary["bamf_phase1_water"] = state["total_water_liters"]
    summary["bamf_phase1_best_map"] = state["best_map"]

    # BAMF-Eco Phase 2 (full-fidelity promotion)
    if promo_results and len(promo_results) > 0:
        maps = [r["map50_95"] for r in promo_results if r.get("map50_95", 0) > 0]
        energies = [r["energy_kwh"] for r in promo_results if r.get("energy_kwh", 0) > 0]
        total_promo_energy = sum(energies)
        summary["bamf_phase2_evals"] = len(promo_results)
        summary["bamf_phase2_energy"] = total_promo_energy
        summary["bamf_total_evals"] = state["iteration"] + len(promo_results)
        summary["bamf_total_energy"] = state["total_energy_kwh"] + total_promo_energy
        summary["bamf_total_co2e"] = summary["bamf_total_energy"] * 0.79
        summary["bamf_total_water"] = summary["bamf_total_energy"] * 3.5
        summary["bamf_best_map"] = max(maps) if maps else state["best_map"]

        # Pareto front from promotion results
        # Simple Pareto: non-dominated in (mAP↑, energy↓)
        pareto = []
        valid = [(r["map50_95"], r["energy_kwh"], r) for r in promo_results
                 if r.get("map50_95", 0) > 0 and r.get("energy_kwh", 0) > 0]
        valid.sort(key=lambda x: (-x[0], x[1]))
        best_energy = float("inf")
        for mAP, energy, r in valid:
            if energy < best_energy:
                pareto.append(r)
                best_energy = energy
        summary["bamf_pareto_size"] = len(pareto)

    # Baselines
    for name, data in baselines.items():
        n_evals = data.get("n_evaluations", len(data.get("results", [])))
        results_list = data.get("results", [])
        if results_list:
            maps = [r.get("map50_95", r.get("mAP", 0)) for r in results_list]
            energies = [r.get("energy_kwh", 0) for r in results_list]
            summary[f"{name}_evals"] = n_evals
            summary[f"{name}_best_map"] = max(maps) if maps else 0
            summary[f"{name}_total_energy"] = sum(energies)
            summary[f"{name}_total_co2e"] = sum(energies) * 0.79
            summary[f"{name}_total_water"] = sum(energies) * 3.5
        else:
            summary[f"{name}_evals"] = n_evals
            summary[f"{name}_best_map"] = 0
            summary[f"{name}_total_energy"] = 0

    return summary


def generate_comparison_table(summary):
    """Generate LaTeX comparison table."""
    rows = []

    def _fmt(key, default="--"):
        val = summary.get(key)
        if val is None:
            return default
        if isinstance(val, float):
            if val < 0.001:
                return f"{val:.6f}"
            elif val < 1:
                return f"{val:.4f}"
            else:
                return f"{val:.2f}"
        return str(val)

    # BAMF-Eco row
    bamf_evals = summary.get("bamf_total_evals", summary.get("bamf_phase1_evals", "--"))
    rows.append(
        f"\\bamfeco{{}} & {bamf_evals} & {_fmt('bamf_best_map')} "
        f"& {_fmt('bamf_pareto_size')} & -- "
        f"& {_fmt('bamf_total_energy')} & {_fmt('bamf_total_co2e')} "
        f"& {_fmt('bamf_total_water')} \\\\"
    )

    # Baselines
    baseline_names = [
        ("random", "Random Search"),
        ("hyperband", "Hyperband"),
        ("single_obj_accuracy", "Single-Obj (Accuracy)"),
        ("single_obj_efficiency", "Single-Obj (Efficiency)"),
        ("manual_expert", "Manual Expert"),
    ]
    for key, display in baseline_names:
        evals = summary.get(f"{key}_evals", "--")
        rows.append(
            f"{display} & {evals} & {_fmt(f'{key}_best_map')} "
            f"& -- & -- "
            f"& {_fmt(f'{key}_total_energy')} & {_fmt(f'{key}_total_co2e')} "
            f"& {_fmt(f'{key}_total_water')} \\\\"
        )

    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print summary but don't modify files")
    args = parser.parse_args()

    print("=" * 60)
    print("BAMF-Eco Paper Results Fill")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    opt_ckpt = load_optimizer_results()
    promo_results = load_promotion_results()
    baselines = load_baseline_results()
    benchmark = load_benchmark_results()

    if opt_ckpt is None:
        print("ERROR: No optimizer checkpoint found. Cannot proceed.")
        sys.exit(1)

    # Compute summary
    print("\nComputing summary...")
    summary = compute_summary(opt_ckpt, promo_results, baselines)

    print("\n--- Summary ---")
    for k, v in sorted(summary.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    # Save summary
    summary_file = PAPER_DIR / "tables" / "results_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary to {summary_file}")

    # Generate comparison table
    table_content = generate_comparison_table(summary)
    table_file = PAPER_DIR / "tables" / "comparison_table_rows.tex"
    with open(table_file, "w") as f:
        f.write(table_content)
    print(f"Saved table rows to {table_file}")

    if args.dry_run:
        print("\n[DRY RUN] Not modifying main.tex")
        return

    print("\nDone! Run 'make' in paper/ to recompile.")


if __name__ == "__main__":
    main()
