"""
Analysis & Visualization Pipeline
=====================================

Publication-quality plots and analysis for the BAMF-Eco paper:

1. Pareto front visualization (accuracy vs energy vs latency)
2. Hypervolume convergence curves
3. Regret plots (simple vs cumulative)
4. Ablation component contribution
5. EcoDetBench leaderboard (radar/spider charts)
6. Sustainability waterfall charts
7. Fidelity correction accuracy plots
8. Per-model family comparison
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for HPC
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.2)
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib/seaborn not available — plots disabled")


# NeurIPS-style formatting
NEURIPS_RC = {
    "figure.figsize": (6, 4),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
}

# Color palette for model families
FAMILY_COLORS = {
    "yolov8": "#1f77b4",
    "yolov9": "#ff7f0e",
    "yolov10": "#2ca02c",
    "yolo11": "#d62728",
    "yolo26": "#9467bd",
    "rtdetr": "#8c564b",
}

# Markers for model sizes
SIZE_MARKERS = {
    "n": "o", "s": "s", "m": "D", "l": "^",
    "t": "v", "c": "P", "e": "X", "x": "*",
}


class PaperFigureGenerator:
    """Generate all figures for the BAMF-Eco paper."""

    def __init__(self, output_dir: str = "paper/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if HAS_MATPLOTLIB:
            plt.rcParams.update(NEURIPS_RC)

    def plot_pareto_front_2d(
        self,
        results: pd.DataFrame,
        x_col: str = "energy_kwh",
        y_col: str = "map50_95",
        family_col: str = "family",
        size_col: str = "size",
        pareto_points: Optional[pd.DataFrame] = None,
        title: str = "Accuracy-Energy Pareto Front",
        xlabel: str = "Energy (kWh)",
        ylabel: str = "mAP@50-95",
        filename: str = "pareto_front_2d.pdf",
    ):
        """Plot 2D Pareto front with family coloring."""
        if not HAS_MATPLOTLIB:
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot all points by family
        for family in results[family_col].unique():
            mask = results[family_col] == family
            family_data = results[mask]
            color = FAMILY_COLORS.get(family, "#333333")

            for size in family_data[size_col].unique():
                size_mask = family_data[size_col] == size
                subset = family_data[size_mask]
                marker = SIZE_MARKERS.get(size, "o")

                ax.scatter(
                    subset[x_col], subset[y_col],
                    c=color, marker=marker, s=80, alpha=0.7,
                    label=f"{family}-{size}",
                    edgecolors="white", linewidths=0.5,
                )

        # Highlight Pareto front
        if pareto_points is not None and len(pareto_points) > 0:
            sorted_pareto = pareto_points.sort_values(x_col)
            ax.plot(
                sorted_pareto[x_col], sorted_pareto[y_col],
                "k--", alpha=0.5, linewidth=1.5, label="Pareto Front",
            )
            ax.scatter(
                sorted_pareto[x_col], sorted_pareto[y_col],
                c="gold", marker="*", s=200, zorder=5,
                edgecolors="black", linewidths=1,
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7, ncol=2)

        path = self.output_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")

    def plot_pareto_front_3d(
        self,
        results: pd.DataFrame,
        x_col: str = "latency_ms",
        y_col: str = "energy_kwh",
        z_col: str = "map50_95",
        family_col: str = "family",
        filename: str = "pareto_front_3d.pdf",
    ):
        """Plot 3D Pareto front (accuracy × energy × latency)."""
        if not HAS_MATPLOTLIB:
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        for family in results[family_col].unique():
            mask = results[family_col] == family
            subset = results[mask]
            color = FAMILY_COLORS.get(family, "#333333")

            ax.scatter(
                subset[x_col], subset[y_col], subset[z_col],
                c=color, s=60, alpha=0.7, label=family,
                edgecolors="white", linewidths=0.3,
            )

        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Energy (kWh)")
        ax.set_zlabel("mAP@50-95")
        ax.legend(loc="upper left", fontsize=8)

        path = self.output_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")

    def plot_hypervolume_convergence(
        self,
        hv_histories: Dict[str, List[float]],
        filename: str = "hypervolume_convergence.pdf",
    ):
        """Plot hypervolume over evaluations for BAMF-Eco vs baselines."""
        if not HAS_MATPLOTLIB:
            return

        fig, ax = plt.subplots(figsize=(8, 5))

        colors = plt.cm.tab10(np.linspace(0, 1, len(hv_histories)))
        for (name, hv_curve), color in zip(hv_histories.items(), colors):
            ax.plot(range(len(hv_curve)), hv_curve, label=name, color=color, linewidth=2)

        ax.set_xlabel("Number of Evaluations")
        ax.set_ylabel("Hypervolume")
        ax.set_title("Hypervolume Convergence")
        ax.legend()

        path = self.output_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")

    def plot_sustainability_waterfall(
        self,
        optimizer_costs: Dict[str, Dict[str, float]],
        filename: str = "sustainability_waterfall.pdf",
    ):
        """
        Waterfall chart comparing total CO₂e & water across optimizers.
        Shows BAMF-Eco savings relative to baselines.
        """
        if not HAS_MATPLOTLIB:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        names = list(optimizer_costs.keys())
        co2e = [optimizer_costs[n].get("co2e_kg", 0) for n in names]
        water = [optimizer_costs[n].get("water_liters", 0) for n in names]

        # CO₂e bar chart
        colors_co2 = ["#2ca02c" if "bamf" in n.lower() else "#1f77b4" for n in names]
        bars1 = ax1.bar(names, co2e, color=colors_co2, alpha=0.8, edgecolor="white")
        ax1.set_ylabel("Total CO₂e (kg)")
        ax1.set_title("Carbon Footprint Comparison")
        ax1.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars1, co2e):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        # Water bar chart
        colors_h2o = ["#2ca02c" if "bamf" in n.lower() else "#1f77b4" for n in names]
        bars2 = ax2.bar(names, water, color=colors_h2o, alpha=0.8, edgecolor="white")
        ax2.set_ylabel("Total Water (liters)")
        ax2.set_title("Water Footprint Comparison")
        ax2.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars2, water):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")

    def plot_ablation_bar(
        self,
        ablation_results: Dict[str, Dict[str, float]],
        filename: str = "ablation_study.pdf",
    ):
        """
        Bar chart showing contribution of each BAMF-Eco component.
        """
        if not HAS_MATPLOTLIB:
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrics = ["hypervolume", "total_co2e_kg", "best_map"]
        titles = ["Hypervolume ↑", "Total CO₂e (kg) ↓", "Best mAP ↑"]

        for ax, metric, title in zip(axes, metrics, titles):
            names = list(ablation_results.keys())
            values = [ablation_results[n].get(metric, 0) for n in names]

            colors = ["#2ca02c" if "full" in n.lower() else "#aaaaaa" for n in names]
            ax.barh(names, values, color=colors, edgecolor="white")
            ax.set_xlabel(title)
            ax.invert_yaxis()

        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")

    def plot_ecodetbench_radar(
        self,
        model_metrics: Dict[str, Dict[str, float]],
        metrics: Optional[List[str]] = None,
        filename: str = "ecodetbench_radar.pdf",
    ):
        """
        Radar/spider chart for EcoDetBench model comparison.
        """
        if not HAS_MATPLOTLIB:
            return

        metrics = metrics or ["mAP", "FPS", "1/Energy", "1/CO2e", "1/Latency"]
        n_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for model_name, model_vals in model_metrics.items():
            values = [model_vals.get(m, 0) for m in metrics]
            values += values[:1]

            family = model_name.split("-")[0] if "-" in model_name else model_name[:6]
            color = FAMILY_COLORS.get(family, "#333333")

            ax.plot(angles, values, "o-", linewidth=1.5, label=model_name, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)
        ax.set_title("EcoDetBench Model Comparison")

        path = self.output_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")

    def plot_fidelity_correction(
        self,
        low_fidelity_vals: np.ndarray,
        high_fidelity_vals: np.ndarray,
        predicted_vals: np.ndarray,
        metric_name: str = "mAP",
        filename: str = "fidelity_correction.pdf",
    ):
        """Plot fidelity correction: predicted vs actual high-fidelity."""
        if not HAS_MATPLOTLIB:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Left: Low vs High fidelity scatter
        ax1.scatter(low_fidelity_vals, high_fidelity_vals,
                     c="#1f77b4", alpha=0.6, s=40, label="Actual")
        lims = [
            min(low_fidelity_vals.min(), high_fidelity_vals.min()) - 0.05,
            max(low_fidelity_vals.max(), high_fidelity_vals.max()) + 0.05,
        ]
        ax1.plot(lims, lims, "k--", alpha=0.3, label="y=x")
        ax1.set_xlabel(f"Low-Fidelity {metric_name}")
        ax1.set_ylabel(f"High-Fidelity {metric_name}")
        ax1.set_title("Low vs High Fidelity")
        ax1.legend()

        # Right: Predicted vs Actual
        ax2.scatter(high_fidelity_vals, predicted_vals,
                     c="#2ca02c", alpha=0.6, s=40)
        lims2 = [
            min(high_fidelity_vals.min(), predicted_vals.min()) - 0.05,
            max(high_fidelity_vals.max(), predicted_vals.max()) + 0.05,
        ]
        ax2.plot(lims2, lims2, "k--", alpha=0.3, label="Perfect prediction")
        ax2.set_xlabel(f"Actual High-Fidelity {metric_name}")
        ax2.set_ylabel(f"Predicted High-Fidelity {metric_name}")
        ax2.set_title("Correction Model Accuracy")
        ax2.legend()

        # Compute and show R²
        ss_res = np.sum((high_fidelity_vals - predicted_vals) ** 2)
        ss_tot = np.sum((high_fidelity_vals - high_fidelity_vals.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        ax2.text(0.05, 0.95, f"R² = {r2:.3f}",
                 transform=ax2.transAxes, fontsize=12,
                 verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")

    def plot_model_family_comparison(
        self,
        results: pd.DataFrame,
        filename: str = "family_comparison.pdf",
    ):
        """Box plots comparing metrics across model families."""
        if not HAS_MATPLOTLIB:
            return

        metrics = ["map50_95", "energy_kwh", "latency_ms"]
        titles = ["mAP@50-95 ↑", "Energy (kWh) ↓", "Latency (ms) ↓"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        palette = [FAMILY_COLORS.get(f, "#333") for f in sorted(results["family"].unique())]

        for ax, metric, title in zip(axes, metrics, titles):
            sns.boxplot(
                data=results, x="family", y=metric,
                palette=palette, ax=ax,
            )
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=45)

        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")


class ResultsAggregator:
    """Aggregate and analyze results from all experiments."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)

    def load_benchmark_results(self, path: str) -> pd.DataFrame:
        """Load EcoDetBench results from CSV/JSON."""
        if path.endswith(".csv"):
            return pd.read_csv(path)
        elif path.endswith(".json"):
            with open(path) as f:
                data = json.load(f)
            return pd.DataFrame(data)
        raise ValueError(f"Unsupported format: {path}")

    def compute_efficiency_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute composite efficiency scores for EcoDetBench ranking."""
        df = df.copy()

        # Normalize each metric to [0, 1]
        for col in ["map50_95", "energy_kwh", "latency_ms", "co2e_kg"]:
            if col in df.columns:
                col_min = df[col].min()
                col_max = df[col].max()
                if col_max > col_min:
                    df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min)
                else:
                    df[f"{col}_norm"] = 0.5

        # Composite score (higher = better)
        if all(f"{c}_norm" in df.columns for c in ["map50_95", "energy_kwh", "latency_ms"]):
            df["eco_score"] = (
                df["map50_95_norm"] * 0.4 +
                (1 - df["energy_kwh_norm"]) * 0.3 +
                (1 - df["latency_ms_norm"]) * 0.3
            )

        return df

    def generate_latex_table(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        caption: str = "EcoDetBench Results",
        label: str = "tab:ecodetbench",
        filename: str = "ecodetbench_table.tex",
    ) -> str:
        """Generate LaTeX table for the paper."""
        columns = columns or [
            "model_name", "params_m", "map50_95",
            "latency_ms_mean", "energy_kwh",
            "co2e_kg", "water_liters",
        ]

        available = [c for c in columns if c in df.columns]
        subset = df[available].copy()

        # Format numbers
        formatters = {
            "params_m": lambda x: f"{x:.1f}",
            "map50_95": lambda x: f"{x:.3f}",
            "latency_ms_mean": lambda x: f"{x:.1f}",
            "energy_kwh": lambda x: f"{x:.6f}",
            "co2e_kg": lambda x: f"{x:.5f}",
            "water_liters": lambda x: f"{x:.3f}",
        }

        latex = subset.to_latex(
            index=False,
            caption=caption,
            label=label,
            formatters={k: v for k, v in formatters.items() if k in available},
            escape=False,
            column_format="l" + "r" * (len(available) - 1),
        )

        path = self.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(latex)

        logger.info(f"LaTeX table saved to {path}")
        return latex

    def generate_comparison_table(
        self,
        optimizer_results: Dict[str, Dict[str, float]],
        filename: str = "optimizer_comparison.tex",
    ) -> str:
        """Generate baseline comparison table."""
        rows = []
        for name, metrics in optimizer_results.items():
            rows.append({
                "Optimizer": name,
                "HV ↑": f"{metrics.get('hypervolume', 0):.4f}",
                "Best mAP ↑": f"{metrics.get('best_map', 0):.3f}",
                "Total CO₂e ↓": f"{metrics.get('total_co2e_kg', 0):.4f}",
                "Total kWh ↓": f"{metrics.get('total_energy_kwh', 0):.4f}",
                "Pareto |P|": str(metrics.get("pareto_size", 0)),
            })

        df = pd.DataFrame(rows)
        latex = df.to_latex(index=False, escape=False)

        path = self.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(latex)

        logger.info(f"Comparison table saved to {path}")
        return latex
