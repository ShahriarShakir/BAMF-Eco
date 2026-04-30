#!/usr/bin/env python3
"""
BAMF-Eco: Complete Analysis Pipeline


Generates all figures and tables...............


Outputs:
  - paper/figures/*.pdf  (publication-quality figures)
  - paper/tables/*.tex   (LaTeX tables)
  - outputs/analysis/    (raw analysis data)

Usage:
    python scripts/run_analysis.py
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from loguru import logger

# Setup paths
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from bamf_eco.analysis import PaperFigureGenerator, ResultsAggregator
from bamf_eco.optimizer import ParetoFront
from bamf_eco.utils import OUTPUT_DIR

PAPER_FIGURES_DIR = BASE_DIR / "paper" / "figures"
PAPER_TABLES_DIR = BASE_DIR / "paper" / "tables"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"

# Ensure dirs exist
for d in [PAPER_FIGURES_DIR, PAPER_TABLES_DIR, ANALYSIS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_optimizer_results():
    """Load BAMF-Eco optimizer results (Phase 1 + Phase 2 promotion)."""
    results = {
        "phase1_history": [],  # Low-fidelity exploration
        "promotion_results": [],  # Full-fidelity promotion
        "pareto_front": None,
        "correction_analysis": None,
    }

    # Phase 1: Load checkpoint history
    checkpoint_path = OUTPUT_DIR / "bamf_eco_main" / "checkpoint.json"
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        results["phase1_history"] = checkpoint.get("history", [])
        results["phase1_state"] = checkpoint.get("state", {})
        results["phase1_settings"] = checkpoint.get("settings", {})
        logger.info(f"Phase 1: {len(results['phase1_history'])} evaluations loaded")
    else:
        logger.warning("Phase 1 checkpoint not found!")

    # Phase 2: Load promotion results
    promotion_path = OUTPUT_DIR / "bamf_eco_promotion" / "promotion_results.json"
    if promotion_path.exists():
        with open(promotion_path) as f:
            results["promotion_results"] = json.load(f)
        logger.info(f"Phase 2: {len(results['promotion_results'])} promotion results")
    else:
        logger.warning("Promotion results not found — run promotion pass first")

    # Pareto front from promotion
    pareto_path = OUTPUT_DIR / "bamf_eco_promotion" / "promotion_pareto.json"
    if pareto_path.exists():
        with open(pareto_path) as f:
            results["pareto_front"] = json.load(f)
        logger.info(f"Pareto front: {len(results['pareto_front'].get('pareto_points', []))} points")

    # Correction analysis
    correction_path = OUTPUT_DIR / "bamf_eco_promotion" / "correction_analysis.json"
    if correction_path.exists():
        with open(correction_path) as f:
            results["correction_analysis"] = json.load(f)

    return results


def load_baseline_results():
    """Load all baseline results."""
    baselines = {}
    baselines_dir = OUTPUT_DIR / "baselines"

    # Load checkpoint files for each baseline
    for ckpt_file in baselines_dir.glob("*_seed42_checkpoint.json"):
        name = ckpt_file.name.replace("_seed42_checkpoint.json", "")
        with open(ckpt_file) as f:
            data = json.load(f)

        results_list = data.get("results", [])
        if not results_list:
            continue

        # Compute totals
        total_energy = sum(r.get("energy_kwh", 0) for r in results_list)
        total_co2e = sum(r.get("co2e_kg", 0) for r in results_list)
        total_water = sum(r.get("water_liters", 0) for r in results_list)
        maps = [r.get("map50_95", 0) for r in results_list]
        energies = [r.get("energy_kwh", float("inf")) for r in results_list]

        # Build Pareto front
        pareto = ParetoFront(
            objectives=["mAP", "energy_kwh", "latency_ms"],
            directions=["maximize", "minimize", "minimize"],
        )
        for r in results_list:
            objectives = {
                "mAP": r.get("map50_95", 0),
                "energy_kwh": r.get("energy_kwh", float("inf")),
                "latency_ms": r.get("latency_ms", float("inf")),
            }
            pareto.update(objectives, r.get("config", {}))

        baselines[name] = {
            "n_evaluations": len(results_list),
            "total_energy_kwh": total_energy,
            "total_co2e_kg": total_co2e,
            "total_water_liters": total_water,
            "best_map": max(maps) if maps else 0,
            "best_energy": min(energies) if energies else float("inf"),
            "pareto_size": len(pareto.points),
            "hypervolume": pareto.hypervolume,
            "hypervolume_history": pareto.hypervolume_history,
            "pareto_points": pareto.points,
            "results": results_list,
        }
        logger.info(
            f"Baseline '{name}': {len(results_list)} evals, "
            f"best mAP={max(maps):.4f}, energy={total_energy:.4f}kWh, "
            f"pareto={len(pareto.points)}"
        )

    return baselines


def load_benchmark_results():
    """Load EcoDetBench inference benchmark results."""
    bench_path = OUTPUT_DIR / "benchmark" / "ecodetbench_results.json"
    if not bench_path.exists():
        logger.warning("Benchmark results not found!")
        return None

    with open(bench_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    logger.info(f"Benchmark: {len(df)} model configurations")
    return df


def compute_bamf_eco_summary(optimizer_results):
    """Compute summary metrics for BAMF-Eco."""
    phase1 = optimizer_results.get("phase1_state", {})
    promotion = optimizer_results.get("promotion_results", [])
    pareto = optimizer_results.get("pareto_front") or {}

    # Total energy: Phase 1 (exploration) + Phase 2 (promotion)
    phase1_energy = phase1.get("total_energy_kwh", 0)
    phase2_energy = sum(r.get("energy_kwh", 0) for r in promotion)
    phase1_co2e = phase1.get("total_co2e_kg", 0)
    phase2_co2e = sum(r.get("co2e_kg", 0) for r in promotion)
    phase1_water = phase1.get("total_water_liters", 0)
    phase2_water = sum(r.get("water_liters", 0) for r in promotion)

    # Best results from promotion (full-fidelity)
    best_map = max([r.get("map50_95", 0) for r in promotion]) if promotion else phase1.get("best_map", 0)

    # Build Pareto front from promotion results if not pre-computed
    if not pareto and promotion:
        pf = ParetoFront(
            objectives=["mAP", "energy_kwh", "latency_ms"],
            directions=["maximize", "minimize", "minimize"],
        )
        for r in promotion:
            objectives = {
                "mAP": r.get("map50_95", 0),
                "energy_kwh": r.get("energy_kwh", float("inf")),
                "latency_ms": r.get("latency_ms", float("inf")),
            }
            pf.update(objectives, r.get("config", {}))
        pareto_size = len(pf.points)
        hypervolume = pf.hypervolume
    else:
        pareto_size = len(pareto.get("pareto_points", []))
        hypervolume = pareto.get("hypervolume", 0)

    return {
        "n_evaluations": phase1.get("iteration", 0) + len(promotion),
        "n_phase1_evals": phase1.get("iteration", 0),
        "n_phase2_evals": len(promotion),
        "total_energy_kwh": phase1_energy + phase2_energy,
        "phase1_energy_kwh": phase1_energy,
        "phase2_energy_kwh": phase2_energy,
        "total_co2e_kg": phase1_co2e + phase2_co2e,
        "total_water_liters": phase1_water + phase2_water,
        "best_map": best_map,
        "pareto_size": pareto_size,
        "hypervolume": hypervolume,
    }


def generate_figure_1_pareto_comparison(optimizer_results, baselines, fig_gen):
    """
    Figure 1: Pareto Front Comparison (BAMF-Eco vs all baselines).
    This is the key result figure showing BAMF-Eco's superior tradeoff.
    """
    logger.info("Generating Figure 1: Pareto Front Comparison")

    all_data = []

    # BAMF-Eco promotion results (full-fidelity)
    for r in optimizer_results.get("promotion_results", []):
        all_data.append({
            "model_name": r.get("model_name", "unknown"),
            "family": r.get("model_name", "")[:6],
            "size": r.get("model_name", "")[-1:],
            "map50_95": r.get("map50_95", 0),
            "energy_kwh": r.get("energy_kwh", 0),
            "latency_ms": r.get("latency_ms", 0),
            "optimizer": "BAMF-Eco",
        })

    # Baselines
    for baseline_name, bdata in baselines.items():
        for r in bdata.get("results", []):
            config = r.get("config", {})
            model_name = config.get("model_name", r.get("model_name", "unknown"))
            all_data.append({
                "model_name": model_name,
                "family": model_name[:6],
                "size": model_name[-1:],
                "map50_95": r.get("map50_95", 0),
                "energy_kwh": r.get("energy_kwh", 0),
                "latency_ms": r.get("latency_ms", 0),
                "optimizer": baseline_name,
            })

    if not all_data:
        logger.warning("No data for Pareto comparison — skipping Figure 1")
        return

    df = pd.DataFrame(all_data)

    # Plot BAMF-Eco points with Pareto front highlighted
    bamf_data = df[df["optimizer"] == "BAMF-Eco"]
    pareto_data = optimizer_results.get("pareto_front") or {}
    pareto_points = pareto_data.get("pareto_points", [])

    if pareto_points:
        pareto_df = pd.DataFrame(pareto_points)
        pareto_df.rename(columns={"mAP": "map50_95"}, inplace=True)
    else:
        pareto_df = None

    fig_gen.plot_pareto_front_2d(
        results=bamf_data,
        pareto_points=pareto_df,
        title="BAMF-Eco: Accuracy-Energy Pareto Front",
        filename="fig1_pareto_front.pdf",
    )

    # Also generate combined comparison
    if len(df) > 0 and HAS_MATPLOTLIB:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 7))

        optimizer_colors = {
            "BAMF-Eco": "#2ca02c",
            "random": "#1f77b4",
            "hyperband": "#ff7f0e",
            "single_obj_accuracy": "#d62728",
            "single_obj_efficiency": "#9467bd",
            "manual_expert": "#8c564b",
        }

        for opt_name in df["optimizer"].unique():
            subset = df[df["optimizer"] == opt_name]
            color = optimizer_colors.get(opt_name, "#333333")
            marker = "*" if opt_name == "BAMF-Eco" else "o"
            size = 150 if opt_name == "BAMF-Eco" else 60

            ax.scatter(
                subset["energy_kwh"], subset["map50_95"],
                c=color, marker=marker, s=size, alpha=0.7,
                label=opt_name, edgecolors="white", linewidths=0.5,
            )

        # Draw BAMF-Eco Pareto front
        if pareto_df is not None and len(pareto_df) > 0:
            sorted_p = pareto_df.sort_values("energy_kwh")
            ax.plot(sorted_p["energy_kwh"], sorted_p["map50_95"],
                    "k--", linewidth=2, alpha=0.6, label="BAMF-Eco Pareto")

        ax.set_xlabel("Energy (kWh)")
        ax.set_ylabel("mAP@50-95")
        ax.set_title("Optimizer Comparison: Accuracy vs Energy")
        ax.legend(loc="lower right")

        path = PAPER_FIGURES_DIR / "fig1_optimizer_comparison.pdf"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        logger.info(f"Saved: {path}")


def generate_figure_2_convergence(optimizer_results, baselines, fig_gen):
    """
    Figure 2: Convergence comparison showing BAMF-Eco's sample efficiency.
    Shows cumulative energy (x-axis) vs best mAP found (y-axis).
    """
    logger.info("Generating Figure 2: Convergence Comparison")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    optimizer_colors = {
        "BAMF-Eco": "#2ca02c",
        "random": "#1f77b4",
        "hyperband": "#ff7f0e",
        "single_obj_accuracy": "#d62728",
        "single_obj_efficiency": "#9467bd",
    }

    # Left: Best mAP vs number of evaluations
    # BAMF-Eco Phase 1 + Phase 2
    phase1 = optimizer_results.get("phase1_history", [])
    promotion = optimizer_results.get("promotion_results", [])

    if phase1:
        cum_best_map = []
        best_so_far = 0
        for rec in phase1:
            if rec.get("result"):
                best_so_far = max(best_so_far, rec["result"].get("map50_95", 0))
            cum_best_map.append(best_so_far)

        # Add promotion phase
        for r in promotion:
            best_so_far = max(best_so_far, r.get("map50_95", 0))
            cum_best_map.append(best_so_far)

        ax1.plot(range(len(cum_best_map)), cum_best_map,
                 color="#2ca02c", linewidth=2, label="BAMF-Eco")

        # Vertical line showing Phase 1 → Phase 2 transition
        ax1.axvline(x=len(phase1), color="#2ca02c", linestyle=":", alpha=0.5)
        ax1.text(len(phase1), best_so_far * 1.01, "→ Full fidelity",
                 fontsize=8, color="#2ca02c", ha="center")

    # Baselines
    for name, bdata in baselines.items():
        results_list = bdata.get("results", [])
        if not results_list:
            continue

        cum_best = []
        best_so_far = 0
        for r in results_list:
            best_so_far = max(best_so_far, r.get("map50_95", 0))
            cum_best.append(best_so_far)

        color = optimizer_colors.get(name, "#333")
        ax1.plot(range(len(cum_best)), cum_best,
                 color=color, linewidth=1.5, label=name, alpha=0.8)

    ax1.set_xlabel("Number of Evaluations")
    ax1.set_ylabel("Best mAP@50-95 Found")
    ax1.set_title("(a) Sample Efficiency")
    ax1.legend(fontsize=8)

    # Right: Best mAP vs cumulative energy
    if phase1:
        cum_energy = []
        cum_best_map = []
        total_e = 0
        best_so_far = 0
        for rec in phase1:
            if rec.get("result"):
                total_e += rec["result"].get("energy_kwh", 0)
                best_so_far = max(best_so_far, rec["result"].get("map50_95", 0))
            cum_energy.append(total_e)
            cum_best_map.append(best_so_far)

        for r in promotion:
            total_e += r.get("energy_kwh", 0)
            best_so_far = max(best_so_far, r.get("map50_95", 0))
            cum_energy.append(total_e)
            cum_best_map.append(best_so_far)

        ax2.plot(cum_energy, cum_best_map,
                 color="#2ca02c", linewidth=2, label="BAMF-Eco")

    for name, bdata in baselines.items():
        results_list = bdata.get("results", [])
        if not results_list:
            continue

        cum_energy = []
        cum_best = []
        total_e = 0
        best_so_far = 0
        for r in results_list:
            total_e += r.get("energy_kwh", 0)
            best_so_far = max(best_so_far, r.get("map50_95", 0))
            cum_energy.append(total_e)
            cum_best.append(best_so_far)

        color = optimizer_colors.get(name, "#333")
        ax2.plot(cum_energy, cum_best,
                 color=color, linewidth=1.5, label=name, alpha=0.8)

    ax2.set_xlabel("Cumulative Energy (kWh)")
    ax2.set_ylabel("Best mAP@50-95 Found")
    ax2.set_title("(b) Energy Efficiency of Search")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    path = PAPER_FIGURES_DIR / "fig2_convergence.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def generate_figure_3_sustainability(bamf_summary, baselines, fig_gen):
    """
    Figure 3: Sustainability comparison waterfall chart.
    Shows CO₂e and water savings of BAMF-Eco vs baselines.
    """
    logger.info("Generating Figure 3: Sustainability Comparison")

    optimizer_costs = {"BAMF-Eco": bamf_summary}
    for name, bdata in baselines.items():
        optimizer_costs[name] = {
            "co2e_kg": bdata.get("total_co2e_kg", 0),
            "water_liters": bdata.get("total_water_liters", 0),
            "total_energy_kwh": bdata.get("total_energy_kwh", 0),
        }

    fig_gen.plot_sustainability_waterfall(optimizer_costs)


def generate_figure_4_fidelity_correction(optimizer_results, fig_gen):
    """
    Figure 4: Fidelity correction model analysis.
    Shows how well low-fidelity results predict full-fidelity performance.
    """
    logger.info("Generating Figure 4: Fidelity Correction Analysis")

    phase1 = optimizer_results.get("phase1_history", [])
    promotion = optimizer_results.get("promotion_results", [])

    if not phase1 or not promotion:
        logger.warning("Need both Phase 1 and Phase 2 data for correction analysis")
        return

    # Create pairs: match by model family
    low_vals = []
    high_vals = []

    promotion_by_model = defaultdict(list)
    for r in promotion:
        promotion_by_model[r.get("model_name", "")].append(r)

    for rec in phase1:
        if not rec.get("result"):
            continue
        model = rec["config"].get("model_name", "")
        if model in promotion_by_model:
            # Average the full-fidelity results for this model
            high_maps = [r["map50_95"] for r in promotion_by_model[model]]
            low_vals.append(rec["result"]["map50_95"])
            high_vals.append(np.mean(high_maps))

    if len(low_vals) < 3:
        logger.warning(f"Not enough paired data ({len(low_vals)}) for correction plot")
        return

    low_vals = np.array(low_vals)
    high_vals = np.array(high_vals)

    # Simple linear prediction for the plot
    from numpy.polynomial import polynomial as P
    coeffs = P.polyfit(low_vals, high_vals, 1)
    predicted = P.polyval(low_vals, coeffs)

    fig_gen.plot_fidelity_correction(
        low_fidelity_vals=low_vals,
        high_fidelity_vals=high_vals,
        predicted_vals=predicted,
        metric_name="mAP@50-95",
        filename="fig4_fidelity_correction.pdf",
    )


def generate_table_1_comparison(bamf_summary, baselines, aggregator):
    """
    Table 1: Main comparison table (BAMF-Eco vs baselines).
    """
    logger.info("Generating Table 1: Optimizer Comparison")

    all_results = {"BAMF-Eco": bamf_summary}
    for name, bdata in baselines.items():
        all_results[name] = bdata

    aggregator.generate_comparison_table(all_results, filename="tab1_comparison.tex")


def generate_table_2_benchmark(benchmark_df, aggregator):
    """
    Table 2: EcoDetBench inference benchmark rankings.
    """
    if benchmark_df is None:
        return

    logger.info("Generating Table 2: EcoDetBench Rankings")

    # Compute efficiency scores
    scored = aggregator.compute_efficiency_scores(benchmark_df)

    # Sort by eco_score
    if "eco_score" in scored.columns:
        scored = scored.sort_values("eco_score", ascending=False)

    aggregator.generate_latex_table(
        scored,
        caption="EcoDetBench: Sustainable Object Detection Benchmark",
        label="tab:ecodetbench",
        filename="tab2_ecodetbench.tex",
    )


def main():
    logger.info("=" * 60)
    logger.info("BAMF-Eco: Complete Analysis Pipeline")
    logger.info("=" * 60)

    # Initialize generators
    fig_gen = PaperFigureGenerator(output_dir=str(PAPER_FIGURES_DIR))
    aggregator = ResultsAggregator(output_dir=str(PAPER_TABLES_DIR))

    # Load all results
    logger.info("\n--- Loading Results ---")
    optimizer_results = load_optimizer_results()
    baselines = load_baseline_results()
    benchmark_df = load_benchmark_results()

    # Compute BAMF-Eco summary
    bamf_summary = compute_bamf_eco_summary(optimizer_results)

    # Save summary
    summary = {
        "bamf_eco": bamf_summary,
        "baselines": {k: {kk: vv for kk, vv in v.items() if kk != "results"}
                      for k, v in baselines.items()},
    }
    with open(ANALYSIS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary saved to {ANALYSIS_DIR / 'summary.json'}")

    # Print summary table
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Optimizer':<25} {'Evals':>6} {'Best mAP':>10} {'Energy(kWh)':>12} {'CO₂e(kg)':>10} {'Pareto':>8}")
    logger.info("-" * 75)
    logger.info(
        f"{'BAMF-Eco':<25} {bamf_summary['n_evaluations']:>6} "
        f"{bamf_summary['best_map']:>10.4f} {bamf_summary['total_energy_kwh']:>12.4f} "
        f"{bamf_summary['total_co2e_kg']:>10.4f} {bamf_summary['pareto_size']:>8}"
    )
    for name, bdata in baselines.items():
        logger.info(
            f"{name:<25} {bdata['n_evaluations']:>6} "
            f"{bdata['best_map']:>10.4f} {bdata['total_energy_kwh']:>12.4f} "
            f"{bdata['total_co2e_kg']:>10.4f} {bdata['pareto_size']:>8}"
        )

    # Generate all figures and tables
    logger.info("\n--- Generating Figures ---")
    try:
        generate_figure_1_pareto_comparison(optimizer_results, baselines, fig_gen)
    except Exception as e:
        logger.error(f"Figure 1 failed: {e}")

    try:
        generate_figure_2_convergence(optimizer_results, baselines, fig_gen)
    except Exception as e:
        logger.error(f"Figure 2 failed: {e}")

    try:
        generate_figure_3_sustainability(bamf_summary, baselines, fig_gen)
    except Exception as e:
        logger.error(f"Figure 3 failed: {e}")

    try:
        generate_figure_4_fidelity_correction(optimizer_results, fig_gen)
    except Exception as e:
        logger.error(f"Figure 4 failed: {e}")

    logger.info("\n--- Generating Tables ---")
    try:
        generate_table_1_comparison(bamf_summary, baselines, aggregator)
    except Exception as e:
        logger.error(f"Table 1 failed: {e}")

    try:
        generate_table_2_benchmark(benchmark_df, aggregator)
    except Exception as e:
        logger.error(f"Table 2 failed: {e}")

    # Generate EcoDetBench plots if benchmark data available
    if benchmark_df is not None:
        try:
            # Benchmark uses 'ap' not 'map50_95'
            y_col = "ap" if "ap" in benchmark_df.columns else "map50_95"
            fig_gen.plot_pareto_front_2d(
                results=benchmark_df,
                x_col="energy_kwh",
                y_col=y_col,
                family_col="family" if "family" in benchmark_df.columns else "model_name",
                size_col="size" if "size" in benchmark_df.columns else "model_name",
                title="EcoDetBench: Inference Efficiency Frontier",
                ylabel=f"{'AP' if y_col == 'ap' else 'mAP'}@50-95",
                filename="fig5_ecodetbench.pdf",
            )
        except Exception as e:
            logger.error(f"Benchmark figure failed: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("Analysis complete!")
    logger.info(f"Figures: {PAPER_FIGURES_DIR}")
    logger.info(f"Tables:  {PAPER_TABLES_DIR}")
    logger.info("=" * 60)


try:
    import matplotlib
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


if __name__ == "__main__":
    main()
