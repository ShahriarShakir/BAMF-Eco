"""
Training Experiment Runner
============================

Provides multi-fidelity training of detection models with:
- Energy monitoring throughout training
- Checkpoint management for pause/resume
- Configurable hyperparameters (lr, optimizer, augmentation, etc.)
- Fidelity control via epoch count
- Per-epoch metric logging for fidelity correction model training

This is the "black-box" evaluator that BAMF-Eco calls.
"""

import os
import gc
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime

import numpy as np
import torch
from loguru import logger

from bamf_eco.utils import (
    ModelSpec, MODEL_REGISTRY, MODEL_DIR, OUTPUT_DIR,
    load_model, set_seed, Timer,
)
from bamf_eco.measurement import get_monitor, EnergyMeasurement
from bamf_eco.sustainability import SustainabilityAccountant, SustainabilityEstimate


@dataclass
class TrainingConfig:
    """Hyperparameters for a single training run."""
    # Model
    model_name: str = "yolov8n"
    precision: str = "fp16"
    image_size: int = 640

    # Fidelity (epoch count is the fidelity dimension)
    epochs: int = 100
    fidelity: float = 1.0    # 0.0-1.0 fraction of max epochs

    # Optimizer
    optimizer_name: str = "SGD"
    lr0: float = 0.01
    lrf: float = 0.01        # Final learning rate fraction
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1

    # Data
    batch_size: int = 16
    workers: int = 8
    dataset_path: str = ""

    # Augmentation
    augment: bool = True
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    erasing: float = 0.4

    # Training settings
    patience: int = 50
    close_mosaic: int = 10
    amp: bool = True
    cos_lr: bool = False

    # Output
    project: str = ""
    name: str = ""
    exist_ok: bool = True
    seed: int = 42

    def __post_init__(self):
        """Sanitize types for compatibility — handles numpy scalars and string-encoded ints/floats."""
        for f_name, f_def in self.__dataclass_fields__.items():
            val = getattr(self, f_name)
            expected_type = f_def.type
            if hasattr(val, 'item'):  # numpy scalar
                setattr(self, f_name, val.item())
            elif isinstance(val, np.str_):
                setattr(self, f_name, str(val))
            elif isinstance(val, str) and expected_type is int:
                try:
                    setattr(self, f_name, int(val))
                except (ValueError, TypeError):
                    pass
            elif isinstance(val, str) and expected_type is float:
                try:
                    setattr(self, f_name, float(val))
                except (ValueError, TypeError):
                    pass

    def effective_epochs(self) -> int:
        """Get actual epoch count considering fidelity."""
        return max(1, int(self.epochs * self.fidelity))

    def to_ultralytics_args(self) -> Dict[str, Any]:
        """Convert to ultralytics train() kwargs."""
        args = {
            "data": self.dataset_path,
            "epochs": self.effective_epochs(),
            "imgsz": self.image_size,
            "batch": self.batch_size,
            "workers": self.workers,
            "optimizer": self.optimizer_name,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "warmup_momentum": self.warmup_momentum,
            "warmup_bias_lr": self.warmup_bias_lr,
            "patience": self.patience,
            "close_mosaic": self.close_mosaic,
            "amp": self.amp and self.precision == "fp16",
            "cos_lr": self.cos_lr,
            "seed": self.seed,
            "exist_ok": self.exist_ok,
            "verbose": True,
            # Augmentation
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "copy_paste": self.copy_paste,
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "degrees": self.degrees,
            "translate": self.translate,
            "scale": self.scale,
            "shear": self.shear,
            "perspective": self.perspective,
            "flipud": self.flipud,
            "fliplr": self.fliplr,
            "erasing": self.erasing,
        }
        if self.project:
            args["project"] = self.project
        if self.name:
            args["name"] = self.name
        return args


@dataclass
class TrainingResult:
    """Result of a single training run."""
    # Identity
    config_hash: str = ""
    model_name: str = ""
    family: str = ""
    size: str = ""

    # Fidelity
    epochs_requested: int = 0
    epochs_actual: int = 0
    fidelity: float = 1.0

    # Accuracy (at end of training)
    map50_95: float = 0.0
    map50: float = 0.0
    map75: float = 0.0
    precision_metric: float = 0.0
    recall: float = 0.0

    # Training dynamics (per-epoch)
    epoch_maps: List[float] = field(default_factory=list)
    epoch_losses: List[float] = field(default_factory=list)

    # Latency (post-training inference benchmark)
    latency_ms: float = 0.0

    # Energy & Sustainability
    energy_kwh: float = 0.0
    power_watts_mean: float = 0.0
    co2e_kg: float = 0.0
    water_liters: float = 0.0

    # Resource usage
    peak_memory_mb: float = 0.0
    training_time_s: float = 0.0

    # Metadata
    hardware: str = ""
    timestamp: str = ""
    output_dir: str = ""
    best_weights_path: str = ""
    hyperparams: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class TrainingRunner:
    """
    Runs training experiments with energy monitoring.

    This is the "black-box evaluator" that the BAMF-Eco optimizer calls.
    It trains a model with given hyperparameters and returns a TrainingResult
    containing all objectives (accuracy, energy, latency, CO₂e).
    """

    def __init__(
        self,
        base_output_dir: Optional[str] = None,
        sustainability_accountant: Optional[SustainabilityAccountant] = None,
        power_sample_interval_ms: int = 100,
    ):
        self.base_output_dir = Path(base_output_dir or str(OUTPUT_DIR / "training"))
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.accountant = sustainability_accountant or SustainabilityAccountant()
        self.power_sample_interval_ms = power_sample_interval_ms

    def _build_run_id(self, config: TrainingConfig) -> str:
        """Generate a unique run ID."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{config.model_name}_f{config.fidelity:.2f}_{ts}"

    def evaluate(
        self,
        config: TrainingConfig,
        run_id: Optional[str] = None,
        resume_if_exists: bool = True,
    ) -> TrainingResult:
        """
        Train a model and return all metrics.

        This is the main entry point called by the optimizer.
        Supports resume: if a previous run exists with last.pt weights,
        it will resume training from there (useful across PBS job boundaries).

        Args:
            config: Training configuration / hyperparameters
            run_id: Optional run identifier (auto-generated if not provided)
            resume_if_exists: If True, resume from last.pt if prior run exists

        Returns:
            TrainingResult with accuracy, energy, latency, CO₂e
        """
        run_id = run_id or self._build_run_id(config)
        spec = MODEL_REGISTRY.get(config.model_name)
        if spec is None:
            raise ValueError(f"Unknown model: {config.model_name}")

        set_seed(config.seed)
        run_dir = self.base_output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Check if we already have a completed result
        result_path = run_dir / "training_result.json"
        if result_path.exists() and resume_if_exists:
            try:
                with open(result_path) as f:
                    saved = json.load(f)
                result = TrainingResult.from_dict(saved)
                if result.map50_95 > 0 or result.epochs_actual > 0:
                    logger.info(f"Skipping {spec.name} fid={config.fidelity:.2f} — already completed (mAP={result.map50_95:.4f})")
                    return result
            except Exception:
                pass  # Corrupted result, re-run

        # Check for resumable training (last.pt from interrupted run)
        last_weights = run_dir / "train" / "weights" / "last.pt"
        resume_training = last_weights.exists() and resume_if_exists

        # Check for accumulated energy from previous partial runs
        prev_energy_kwh = 0.0
        prev_training_time = 0.0
        energy_log = run_dir / "energy_log.json"
        if energy_log.exists() and resume_training:
            try:
                with open(energy_log) as f:
                    elog = json.load(f)
                prev_energy_kwh = elog.get("accumulated_energy_kwh", 0.0)
                prev_training_time = elog.get("accumulated_time_s", 0.0)
                logger.info(f"  Resuming with {prev_energy_kwh:.6f}kWh accumulated energy")
            except Exception:
                pass

        if resume_training:
            logger.info(
                f"RESUMING {spec.name} | fidelity={config.fidelity:.2f} "
                f"({config.effective_epochs()} epochs) | from last.pt"
            )
        else:
            logger.info(
                f"Training {spec.name} | fidelity={config.fidelity:.2f} "
                f"({config.effective_epochs()} epochs) | lr={config.lr0} | "
                f"bs={config.batch_size} | imgsz={config.image_size}"
            )

        # Save config
        config_path = run_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(config), f, indent=2, default=str)

        # Load model (from last.pt if resuming, else from pretrained)
        if resume_training:
            from ultralytics import YOLO
            model = YOLO(str(last_weights))
        else:
            model = load_model(spec)

        # Prepare ultralytics args
        config.project = str(run_dir)
        config.name = "train"
        train_args = config.to_ultralytics_args()

        if resume_training:
            train_args["resume"] = True

        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        # Start power monitoring
        monitor = get_monitor(
            gpu_index=0,
            sample_interval_ms=self.power_sample_interval_ms,
            measurement_type="training",
        )
        monitor.start()

        # ---- TRAIN ----
        t_start = time.perf_counter()
        try:
            results = model.train(**train_args)
        except Exception as e:
            # Save partial energy on failure (for resume)
            energy = monitor.stop()
            partial_energy = prev_energy_kwh + energy.energy_kwh
            partial_time = prev_training_time + (time.perf_counter() - t_start)
            with open(energy_log, "w") as f:
                json.dump({
                    "accumulated_energy_kwh": partial_energy,
                    "accumulated_time_s": partial_time,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, f, indent=2)
            logger.error(f"Training failed (energy saved for resume): {e}")
            raise

        training_time = time.perf_counter() - t_start
        energy = monitor.stop()

        # Accumulate energy across resume boundaries
        total_energy_kwh = prev_energy_kwh + energy.energy_kwh
        total_training_time = prev_training_time + training_time

        # ---- Collect metrics ----
        # Peak memory
        peak_memory_mb = 0.0
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        # Extract accuracy from results
        map50_95 = 0.0
        map50 = 0.0
        map75 = 0.0
        precision_metric = 0.0
        recall = 0.0

        if results is not None:
            try:
                metrics = results.results_dict
                map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
                map50 = metrics.get("metrics/mAP50(B)", 0.0)
                precision_metric = metrics.get("metrics/precision(B)", 0.0)
                recall = metrics.get("metrics/recall(B)", 0.0)
            except Exception as e:
                logger.warning(f"Could not extract metrics: {e}")

        # Extract per-epoch metrics from CSV if available
        epoch_maps, epoch_losses = self._parse_training_csv(
            run_dir / "train" / "results.csv"
        )

        # Sustainability (using total accumulated energy)
        sustainability = self.accountant.compute(
            energy_kwh=total_energy_kwh,
            measurement_type="measured" if energy.power_mean_watts > 0 else "derived",
        )

        # Post-training inference latency (quick, 50 iters)
        latency_ms = self._measure_post_training_latency(
            model, config.image_size, device="cuda:0" if torch.cuda.is_available() else "cpu"
        )

        # Build result
        result = TrainingResult(
            config_hash=run_id,
            model_name=spec.name,
            family=spec.family,
            size=spec.size,
            epochs_requested=config.effective_epochs(),
            epochs_actual=len(epoch_maps) if epoch_maps else config.effective_epochs(),
            fidelity=config.fidelity,
            map50_95=map50_95,
            map50=map50,
            map75=map75,
            precision_metric=precision_metric,
            recall=recall,
            epoch_maps=epoch_maps,
            epoch_losses=epoch_losses,
            latency_ms=latency_ms,
            energy_kwh=total_energy_kwh,
            power_watts_mean=energy.power_mean_watts,
            co2e_kg=sustainability.co2e_kg,
            water_liters=sustainability.water_liters,
            peak_memory_mb=peak_memory_mb,
            training_time_s=total_training_time,
            hardware=energy.gpu_name,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            output_dir=str(run_dir),
            best_weights_path=str(run_dir / "train" / "weights" / "best.pt"),
            hyperparams={
                "lr0": config.lr0,
                "optimizer": config.optimizer_name,
                "batch_size": config.batch_size,
                "image_size": config.image_size,
                "precision": config.precision,
                "momentum": config.momentum,
                "weight_decay": config.weight_decay,
            },
        )

        # Save result
        result_path = run_dir / "training_result.json"
        with open(result_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        logger.info(
            f"  Training complete: mAP50-95={result.map50_95:.4f} | "
            f"latency={result.latency_ms:.2f}ms | "
            f"energy={result.energy_kwh:.6f}kWh | "
            f"CO₂e={result.co2e_kg:.6f}kg | "
            f"time={result.training_time_s:.0f}s"
        )

        # Cleanup
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def _parse_training_csv(self, csv_path: Path) -> Tuple[List[float], List[float]]:
        """Extract per-epoch mAP and loss from ultralytics results.csv."""
        epoch_maps = []
        epoch_losses = []
        if not csv_path.exists():
            return epoch_maps, epoch_losses

        try:
            import pandas as pd
            df = pd.read_csv(csv_path, skipinitialspace=True)
            # Column names vary by ultralytics version
            map_col = None
            loss_col = None
            for col in df.columns:
                col_stripped = col.strip()
                if "mAP50-95" in col_stripped:
                    map_col = col
                if "box_loss" in col_stripped or "train/box_loss" in col_stripped:
                    loss_col = col

            if map_col:
                epoch_maps = df[map_col].dropna().tolist()
            if loss_col:
                epoch_losses = df[loss_col].dropna().tolist()
        except Exception as e:
            logger.warning(f"Failed to parse training CSV: {e}")

        return epoch_maps, epoch_losses

    def _measure_post_training_latency(
        self,
        model,
        image_size: int,
        device: str = "cpu",
        n_iters: int = 50,
    ) -> float:
        """Quick latency measurement after training."""
        dummy = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)

        # Warmup
        for _ in range(10):
            model.predict(dummy, imgsz=image_size, verbose=False, device=device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        latencies = []
        for _ in range(n_iters):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.predict(dummy, imgsz=image_size, verbose=False, device=device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)

        return float(np.mean(latencies))

    def evaluate_from_dict(self, params: Dict[str, Any]) -> TrainingResult:
        """
        Evaluate from a dictionary of hyperparameters.
        Convenience method for optimizer integration.
        """
        config = TrainingConfig(
            model_name=params.get("model_name", "yolov8n"),
            precision=params.get("precision", "fp16"),
            image_size=params.get("image_size", 640),
            epochs=params.get("max_epochs", 100),
            fidelity=params.get("fidelity", 1.0),
            optimizer_name=params.get("optimizer", "SGD"),
            lr0=params.get("lr0", 0.01),
            momentum=params.get("momentum", 0.937),
            weight_decay=params.get("weight_decay", 0.0005),
            batch_size=params.get("batch_size", 16),
            dataset_path=params.get("dataset_path", ""),
            seed=params.get("seed", 42),
        )
        return self.evaluate(config)
