#!/usr/bin/env python3
"""
BAMF-Eco Phase 2:

This implements the Successive Halving promotion strategy:
- Phase 1: 193 cheap evals at fidelity=0.05 (~1.46 kWh total)
- Phase 2: Top ~30 configs promoted to fidelity=1.0 (full training)
- Result: Pareto front built from reliable full-fidelity measurements
           + Correction model trained from paired (low, high) fidelity data

Usage:
    python scripts/run_promotion.py [--top-k 30]
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from loguru import logger

# Setup paths
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from bamf_eco.optimizer import BAMFEcoOptimizer, ParetoFront, EvaluationRecord
from bamf_eco.optimizer.acquisition import EcoConstraints
from bamf_eco.optimizer.config_space import ConfigEncoder
from bamf_eco.training import TrainingRunner, TrainingConfig, TrainingResult
from bamf_eco.sustainability import SustainabilityAccountant
from bamf_eco.utils import OUTPUT_DIR


def select_diverse_top_configs(checkpoint_path: str, top_k: int = 30):
    """
    Select top K diverse configs from checkpoint history.

    Strategy:
    1. Group by model architecture (ensure diversity across models)
    2. Within each model, rank by low-fidelity mAP
    3. Select top configs ensuring at least 1 per model family
    4. Fill remaining slots with overall best performers
    """
    with open(checkpoint_path) as f:
        checkpoint = json.load(f)

    history = checkpoint.get("history", [])
    if not history:
        logger.error("No history found in checkpoint!")
        return []

    logger.info(f"Analyzing {len(history)} evaluations from Phase 1")

    # Group by model name
    model_groups = defaultdict(list)
    for rec in history:
        if rec.get("result") is None:
            continue
        model_name = rec["config"].get("model_name", "unknown")
        map_val = rec["result"].get("map50_95", 0)
        energy = rec["result"].get("energy_kwh", float("inf"))
        model_groups[model_name].append({
            "config": rec["config"],
            "fidelity": rec["fidelity"],
            "map50_95": map_val,
            "energy_kwh": energy,
        })

    logger.info(f"Models explored: {list(model_groups.keys())}")
    for model, recs in model_groups.items():
        maps = [r["map50_95"] for r in recs]
        logger.info(
            f"  {model}: {len(recs)} evals, "
            f"best mAP={max(maps):.4f}, mean={np.mean(maps):.4f}"
        )

    # Selection strategy: diverse + top performers
    selected = []
    seen_keys = set()

    def _config_key(cfg):
        return (
            cfg.get("model_name", ""),
            cfg.get("image_size", 0),
            cfg.get("batch_size", 0),
            round(cfg.get("lr0", 0), 5),
        )

    # Phase A: Top 1 from each model (ensure diversity)
    for model, recs in sorted(model_groups.items()):
        best = max(recs, key=lambda r: r["map50_95"])
        key = _config_key(best["config"])
        if key not in seen_keys:
            selected.append(best["config"])
            seen_keys.add(key)

    # Phase B: Top performers across all models (fill remaining)
    all_recs = []
    for model, recs in model_groups.items():
        all_recs.extend(recs)
    all_recs.sort(key=lambda r: r["map50_95"], reverse=True)

    for rec in all_recs:
        if len(selected) >= top_k:
            break
        key = _config_key(rec["config"])
        if key not in seen_keys:
            selected.append(rec["config"])
            seen_keys.add(key)

    # Phase C: Add some efficiency-focused configs (low energy, decent mAP)
    # Filter for configs with mAP > median
    median_map = np.median([r["map50_95"] for r in all_recs])
    efficient_recs = [r for r in all_recs if r["map50_95"] > median_map]
    efficient_recs.sort(key=lambda r: r["energy_kwh"])

    for rec in efficient_recs:
        if len(selected) >= top_k:
            break
        key = _config_key(rec["config"])
        if key not in seen_keys:
            selected.append(rec["config"])
            seen_keys.add(key)

    logger.info(f"Selected {len(selected)} configs for full-fidelity promotion")
    return selected


def run_promotion(
    configs,
    dataset_path: str,
    output_dir: Path,
    walltime_seconds: float = None,
    safety_margin: float = 1800.0,
):
    """
    Evaluate selected configs at full fidelity (1.0).

    Saves results incrementally for checkpoint/resume.
    """
    job_start = time.time()

    # Setup
    accountant = SustainabilityAccountant(carbon_region="default")
    evaluator = TrainingRunner(
        sustainability_accountant=accountant,
        power_sample_interval_ms=100,
    )

    results_file = output_dir / "promotion_results.json"
    progress_file = output_dir / "promotion_progress.json"

    # Load previous progress (for resume)
    completed_keys = set()
    results = []
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        for r in results:
            cfg = r.get("config", {})
            key = (
                r.get("model_name", ""),
                r.get("image_size", 0),
                r.get("batch_size", 0),
                round(cfg.get("lr0", r.get("lr0", 0)), 5),
            )
            completed_keys.add(key)
        logger.info(f"Resuming: {len(results)} promotions already complete")

    total_energy = sum(r.get("energy_kwh", 0) for r in results)

    for i, config in enumerate(configs):
        # Check walltime
        if walltime_seconds:
            elapsed = time.time() - job_start
            remaining = walltime_seconds - elapsed
            if remaining < safety_margin:
                logger.warning(
                    f"Walltime safety: {remaining/60:.0f}min left. "
                    f"Completed {len(results)}/{len(configs)} promotions."
                )
                break

        # Skip already completed
        key = (
            config.get("model_name", ""),
            config.get("image_size", 0),
            config.get("batch_size", 0),
            round(config.get("lr0", 0), 5),
        )
        if key in completed_keys:
            logger.info(f"[{i+1}/{len(configs)}] Skipping {config.get('model_name')} (already done)")
            continue

        logger.info(
            f"[{i+1}/{len(configs)}] Promoting {config.get('model_name')} "
            f"| imgsz={config.get('image_size')} | lr={config.get('lr0'):.6f} "
            f"| bs={config.get('batch_size')} → fidelity=1.0"
        )

        # Sanitize types — checkpoint stores some ints as strings
        image_size = int(config.get("image_size", 640))
        epochs = int(config.get("epochs", 100))
        batch_size = int(config.get("batch_size", 16))
        lr0 = float(config.get("lr0", 0.01))
        momentum = float(config.get("momentum", 0.937))
        weight_decay = float(config.get("weight_decay", 0.0005))

        # Build training config at FULL fidelity
        train_config = TrainingConfig(
            model_name=config.get("model_name", "yolov8n"),
            precision=config.get("precision", "fp16"),
            image_size=image_size,
            epochs=epochs,
            fidelity=1.0,  # FULL FIDELITY
            optimizer_name=config.get("optimizer", "SGD"),
            lr0=lr0,
            momentum=momentum,
            weight_decay=weight_decay,
            batch_size=batch_size,
            dataset_path=dataset_path,
            seed=42 + i,
        )

        try:
            result = evaluator.evaluate(train_config)

            result_dict = {
                "config": config,
                "model_name": config.get("model_name"),
                "image_size": config.get("image_size"),
                "batch_size": config.get("batch_size"),
                "lr0": config.get("lr0"),
                "fidelity": 1.0,
                "map50_95": result.map50_95,
                "map50": result.map50,
                "energy_kwh": result.energy_kwh,
                "co2e_kg": result.co2e_kg,
                "latency_ms": result.latency_ms,
                "training_time_s": result.training_time_s,
                "water_liters": result.water_liters,
                "precision_val": result.precision_metric,
                "recall_val": result.recall,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            results.append(result_dict)
            total_energy += result.energy_kwh

            logger.info(
                f"  ✓ mAP={result.map50_95:.4f} | "
                f"energy={result.energy_kwh:.4f}kWh | "
                f"latency={result.latency_ms:.1f}ms | "
                f"time={result.training_time_s:.0f}s"
            )

            # Save incrementally
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

            # Update progress
            progress = {
                "completed": len(results),
                "total": len(configs),
                "total_energy_kwh": total_energy,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(progress_file, "w") as f:
                json.dump(progress, f, indent=2)

        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            continue

    # Build Pareto front from full-fidelity results
    logger.info("\n=== Building Pareto Front ===")
    pareto = ParetoFront(
        objectives=["mAP", "energy_kwh", "latency_ms"],
        directions=["maximize", "minimize", "minimize"],
    )

    for r in results:
        objectives = {
            "mAP": r["map50_95"],
            "energy_kwh": r["energy_kwh"],
            "latency_ms": r["latency_ms"],
        }
        pareto.update(objectives, r["config"])

    logger.info(f"Pareto front size: {len(pareto.points)}")
    logger.info(f"Hypervolume: {pareto.hypervolume:.6f}")
    for pt in pareto.points:
        logger.info(f"  mAP={pt['mAP']:.4f} | energy={pt['energy_kwh']:.4f} | latency={pt['latency_ms']:.1f}")

    # Save Pareto front
    pareto_data = {
        "pareto_points": pareto.points,
        "pareto_configs": pareto.configs,
        "hypervolume": pareto.hypervolume,
        "total_promotions": len(results),
        "total_energy_kwh": total_energy,
    }
    with open(output_dir / "promotion_pareto.json", "w") as f:
        json.dump(pareto_data, f, indent=2, default=str)

    # Train correction model from paired data
    logger.info("\n=== Training Fidelity Correction Model ===")
    _train_correction_model(output_dir, results)

    return results, pareto


def _train_correction_model(output_dir: Path, full_fidelity_results: list):
    """
    Train the fidelity correction model from paired (low, high) fidelity data.

    Reads the original checkpoint to get low-fidelity results,
    pairs them with the full-fidelity promotion results.
    """
    from bamf_eco.optimizer.fidelity_correction import MultiFidelityCorrectionManager
    from bamf_eco.optimizer.config_space import ConfigEncoder

    checkpoint_path = OUTPUT_DIR / "bamf_eco_main" / "checkpoint.json"
    if not checkpoint_path.exists():
        logger.warning("No checkpoint found for correction model training")
        return

    with open(checkpoint_path) as f:
        checkpoint = json.load(f)

    encoder = ConfigEncoder()
    correction = MultiFidelityCorrectionManager(
        objectives=["mAP", "energy_kwh", "latency_ms"],
        method="gp",
        n_features=encoder.encoded_dim,
    )

    # Create pairs: for each full-fidelity result, find matching low-fidelity evals
    n_pairs = 0
    for full_result in full_fidelity_results:
        model_name = full_result.get("model_name", "")
        high_metrics = {
            "mAP": full_result["map50_95"],
            "energy_kwh": full_result["energy_kwh"],
            "latency_ms": full_result["latency_ms"],
        }

        for rec in checkpoint.get("history", []):
            if rec.get("result") is None:
                continue
            if rec["config"].get("model_name") != model_name:
                continue

            config_vec = np.array(rec.get("config_vec", encoder.encode(rec["config"])))
            low_metrics = {
                "mAP": rec["result"]["map50_95"],
                "energy_kwh": rec["result"]["energy_kwh"],
                "latency_ms": rec["result"]["latency_ms"],
            }

            correction.add_paired_observation(
                config_vec=config_vec,
                low_fidelity=rec["fidelity"],
                high_fidelity=1.0,
                low_metrics=low_metrics,
                high_metrics=high_metrics,
                model_name=model_name,
            )
            n_pairs += 1

    logger.info(f"Created {n_pairs} correction pairs")

    # Train
    if n_pairs >= 3:
        r2_scores = correction.fit_all()
        logger.info(f"Correction model R²: {r2_scores}")

        # Save
        correction.save(str(output_dir / "correction_model"))
        logger.info("Correction model saved")

        # Save analysis data
        correction_analysis = {
            "n_pairs": n_pairs,
            "r2_scores": r2_scores if r2_scores else {},
            "n_full_fidelity_results": len(full_fidelity_results),
        }
        with open(output_dir / "correction_analysis.json", "w") as f:
            json.dump(correction_analysis, f, indent=2)
    else:
        logger.warning(f"Not enough pairs ({n_pairs}) for correction model training")


def main():
    parser = argparse.ArgumentParser(description="BAMF-Eco Phase 2: Promotion Pass")
    parser.add_argument("--top-k", type=int, default=30,
                        help="Number of configs to promote to full fidelity")
    parser.add_argument("--walltime", type=float, default=48*3600,
                        help="PBS walltime in seconds")
    parser.add_argument("--safety-margin", type=float, default=1800,
                        help="Safety margin before walltime (seconds)")
    parser.add_argument("--dataset", type=str,
                        default="/path/to/bamf-eco/datasets/coco_person/coco_person.yaml")
    args = parser.parse_args()

    # Setup output
    output_dir = OUTPUT_DIR / "bamf_eco_promotion"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_path = output_dir / "progress.log"
    logger.add(str(log_path), rotation="50 MB", level="INFO",
               format="{time:HH:mm:ss} | {message}")

    logger.info("=" * 60)
    logger.info("BAMF-Eco Phase 2: Successive Halving Promotion")
    logger.info("=" * 60)

    # Load checkpoint and select configs
    checkpoint_path = OUTPUT_DIR / "bamf_eco_main" / "checkpoint.json"
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    configs = select_diverse_top_configs(str(checkpoint_path), top_k=args.top_k)
    if not configs:
        logger.error("No configs selected for promotion!")
        sys.exit(1)

    # Run promotion
    results, pareto = run_promotion(
        configs=configs,
        dataset_path=args.dataset,
        output_dir=output_dir,
        walltime_seconds=args.walltime,
        safety_margin=args.safety_margin,
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PROMOTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Configs promoted: {len(results)}/{len(configs)}")
    logger.info(f"Pareto front size: {len(pareto.points)}")
    if results:
        maps = [r["map50_95"] for r in results]
        energies = [r["energy_kwh"] for r in results]
        logger.info(f"mAP range: [{min(maps):.4f}, {max(maps):.4f}]")
        logger.info(f"Energy range: [{min(energies):.4f}, {max(energies):.4f}] kWh")
        logger.info(f"Total promotion energy: {sum(energies):.4f} kWh")


if __name__ == "__main__":
    main()
