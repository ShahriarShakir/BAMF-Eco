"""
Inference Benchmark Runner
============================

Systematic benchmarking of detection models measuring:
- Accuracy (AP, AP50, AP75) on COCO
- Latency (ms) with proper warmup
- Throughput (FPS)
- Peak GPU memory (MB)
- Energy consumption (kWh) via NVML
- Derived CO₂e and water footprint

Supports all 24 model variants across precisions and image sizes.
"""

import os
import time
import gc
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from bamf_eco.utils import (
    ModelSpec, BenchmarkResult, MODEL_REGISTRY, MODEL_DIR,
    get_model_specs, load_model, Timer, set_seed,
)
from bamf_eco.measurement import GPUPowerMonitor, EnergyMeasurement, get_monitor
from bamf_eco.sustainability import SustainabilityAccountant, SustainabilityEstimate


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    model_name: str
    precision: str = "fp16"      # fp32, fp16, int8
    image_size: int = 640
    batch_size: int = 1
    warmup_iters: int = 50
    benchmark_iters: int = 200
    num_images: int = 100        # Number of unique images for accuracy eval
    device: str = "cuda:0"
    seed: int = 42


class InferenceBenchmarkRunner:
    """
    Runs inference benchmarks for detection models.

    Measures latency, throughput, memory, energy, and accuracy
    in a standardized protocol.
    """

    def __init__(
        self,
        sustainability_accountant: Optional[SustainabilityAccountant] = None,
        power_sample_interval_ms: int = 50,
        verbose: bool = True,
    ):
        self.accountant = sustainability_accountant or SustainabilityAccountant()
        self.power_sample_interval_ms = power_sample_interval_ms
        self.verbose = verbose

    def _get_device(self, device: str) -> str:
        """Validate and return device string."""
        if "cuda" in device and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return "cpu"
        return device

    def _prepare_model(self, spec: ModelSpec, config: BenchmarkConfig):
        """Load and prepare model for inference."""
        model = load_model(spec, device=config.device)

        # Set precision via model settings
        if config.precision == "fp16":
            model.overrides["half"] = True
        elif config.precision == "int8":
            # INT8 requires export; for now treat as fp16 fallback
            model.overrides["half"] = True
            logger.warning(f"INT8 not directly supported for {spec.name}, using FP16")

        return model

    def _generate_dummy_input(self, config: BenchmarkConfig) -> np.ndarray:
        """Generate a random input image for latency benchmarking."""
        return np.random.randint(
            0, 255,
            (config.image_size, config.image_size, 3),
            dtype=np.uint8,
        )

    def _warmup(self, model, config: BenchmarkConfig):
        """Run warmup iterations to stabilize GPU clocks and caches."""
        logger.info(f"Running {config.warmup_iters} warmup iterations...")
        dummy = self._generate_dummy_input(config)
        for _ in range(config.warmup_iters):
            model.predict(dummy, imgsz=config.image_size, verbose=False, device=config.device)

        # Force sync
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _measure_latency(
        self,
        model,
        config: BenchmarkConfig,
    ) -> Tuple[List[float], EnergyMeasurement]:
        """
        Measure per-image inference latency with power monitoring.

        Returns:
            latencies: List of per-image latency in milliseconds
            energy: EnergyMeasurement from power monitor
        """
        dummy = self._generate_dummy_input(config)
        latencies = []

        # Start power monitoring
        monitor = get_monitor(
            gpu_index=0,
            sample_interval_ms=self.power_sample_interval_ms,
            measurement_type="inference",
        )
        monitor.start()

        for i in range(config.benchmark_iters):
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_start = time.perf_counter()
            model.predict(dummy, imgsz=config.image_size, verbose=False, device=config.device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000.0)  # ms

        energy = monitor.stop()
        return latencies, energy

    def _measure_memory(self, model, config: BenchmarkConfig) -> float:
        """Measure peak GPU memory usage in MB."""
        if not torch.cuda.is_available():
            return 0.0

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()

        dummy = self._generate_dummy_input(config)
        model.predict(dummy, imgsz=config.image_size, verbose=False, device=config.device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        peak_bytes = torch.cuda.max_memory_allocated()
        return peak_bytes / (1024 * 1024)  # MB

    def _evaluate_accuracy(
        self,
        model,
        config: BenchmarkConfig,
        data_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate detection accuracy on dataset.

        Returns dict with ap, ap50, ap75.
        Falls back to dummy values if no dataset is available.
        """
        if data_path and os.path.exists(data_path):
            try:
                results = model.val(
                    data=data_path,
                    imgsz=config.image_size,
                    batch=config.batch_size,
                    device=config.device,
                    verbose=False,
                )
                return {
                    "ap": float(results.box.map),     # mAP50-95
                    "ap50": float(results.box.map50),
                    "ap75": float(results.box.map75),
                }
            except Exception as e:
                logger.warning(f"Accuracy evaluation failed: {e}")

        # No dataset available — return -1 to indicate "not measured"
        return {"ap": -1.0, "ap50": -1.0, "ap75": -1.0}

    def run_single(
        self,
        config: BenchmarkConfig,
        data_path: Optional[str] = None,
    ) -> BenchmarkResult:
        """
        Run a complete benchmark for a single model configuration.

        Args:
            config: Benchmark configuration
            data_path: Path to COCO-format dataset YAML for accuracy eval

        Returns:
            BenchmarkResult with all metrics
        """
        spec = MODEL_REGISTRY.get(config.model_name)
        if spec is None:
            raise ValueError(f"Unknown model: {config.model_name}")

        device = self._get_device(config.device)
        set_seed(config.seed)

        logger.info(
            f"Benchmarking {spec.name} | precision={config.precision} | "
            f"imgsz={config.image_size} | batch={config.batch_size} | device={device}"
        )

        # Load model
        with Timer("model_load"):
            model = self._prepare_model(spec, config)

        # Warmup
        self._warmup(model, config)

        # Measure latency + energy
        latencies, energy = self._measure_latency(model, config)

        # Measure memory
        peak_memory = self._measure_memory(model, config)

        # Compute sustainability
        sustainability = self.accountant.compute(
            energy_kwh=energy.energy_kwh,
            measurement_type="measured" if energy.power_mean_watts > 0 else "derived",
        )

        # Evaluate accuracy (if dataset available)
        accuracy = self._evaluate_accuracy(model, config, data_path)

        # Compile result
        latency_arr = np.array(latencies)
        result = BenchmarkResult(
            model_name=spec.name,
            family=spec.family,
            size=spec.size,
            precision=config.precision,
            image_size=config.image_size,
            batch_size=config.batch_size,
            ap=accuracy["ap"],
            ap50=accuracy["ap50"],
            ap75=accuracy["ap75"],
            latency_ms_mean=float(np.mean(latency_arr)),
            latency_ms_std=float(np.std(latency_arr)),
            throughput_fps=1000.0 / float(np.mean(latency_arr)) if np.mean(latency_arr) > 0 else 0,
            peak_memory_mb=peak_memory,
            energy_kwh=energy.energy_kwh,
            power_watts_mean=energy.power_mean_watts,
            power_watts_std=energy.power_std_watts,
            co2e_kg=sustainability.co2e_kg,
            co2e_kg_min=sustainability.co2e_kg_min,
            co2e_kg_max=sustainability.co2e_kg_max,
            water_liters=sustainability.water_liters,
            water_liters_min=sustainability.water_liters_min,
            water_liters_max=sustainability.water_liters_max,
            hardware=energy.gpu_name,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            duration_s=energy.duration_seconds,
            num_images=config.benchmark_iters,
            params_m=spec.params_m,
            seed=config.seed,
        )

        # Clean up
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.verbose:
            logger.info(
                f"  Result: latency={result.latency_ms_mean:.2f}±{result.latency_ms_std:.2f}ms | "
                f"FPS={result.throughput_fps:.1f} | mem={result.peak_memory_mb:.0f}MB | "
                f"energy={result.energy_kwh:.8f}kWh | CO₂e={result.co2e_kg:.6f}kg | "
                f"water={result.water_liters:.4f}L"
            )

        return result

    def run_sweep(
        self,
        model_names: Optional[List[str]] = None,
        precisions: Optional[List[str]] = None,
        image_sizes: Optional[List[int]] = None,
        data_path: Optional[str] = None,
        warmup_iters: int = 50,
        benchmark_iters: int = 200,
        device: str = "cuda:0",
        seed: int = 42,
        checkpoint_path: Optional[str] = None,
        walltime_seconds: Optional[float] = None,
        safety_margin_seconds: float = 1800.0,
    ) -> List[BenchmarkResult]:
        """
        Run a full sweep across models, precisions, and image sizes.

        Supports checkpoint/resume: saves results incrementally after each config.
        Can be safely interrupted and resumed across PBS job boundaries.

        Args:
            checkpoint_path: Path to save/load checkpoint. If None, no checkpointing.
            walltime_seconds: PBS walltime limit in seconds. Stops safely before expiry.
            safety_margin_seconds: Stop this many seconds before walltime (default 30 min).
        """
        job_start_time = time.time()

        if model_names is None:
            model_names = list(MODEL_REGISTRY.keys())
        if precisions is None:
            precisions = ["fp32", "fp16"]
        if image_sizes is None:
            image_sizes = [640]

        total = len(model_names) * len(precisions) * len(image_sizes)

        # Load existing results from checkpoint
        completed_keys: set = set()
        results = []
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path) as f:
                    saved = json.load(f)
                results = [BenchmarkResult.from_dict(r) for r in saved]
                completed_keys = {
                    f"{r.model_name}_{r.precision}_{r.image_size}"
                    for r in results
                }
                logger.info(f"Resumed sweep: {len(results)}/{total} already completed")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint, starting fresh: {e}")

        logger.info(f"Sweep: {len(model_names)} models × {len(precisions)} precisions × {len(image_sizes)} sizes = {total} configs ({len(completed_keys)} done)")

        for model_name in model_names:
            for precision in precisions:
                for imgsz in image_sizes:
                    key = f"{model_name}_{precision}_{imgsz}"
                    if key in completed_keys:
                        continue  # Already done in previous job

                    # Check walltime safety
                    if walltime_seconds:
                        elapsed = time.time() - job_start_time
                        remaining = walltime_seconds - elapsed
                        if remaining < safety_margin_seconds:
                            logger.warning(
                                f"Walltime safety: {remaining/60:.0f}min remaining. "
                                f"Saving {len(results)}/{total} results and stopping."
                            )
                            return results

                    config = BenchmarkConfig(
                        model_name=model_name,
                        precision=precision,
                        image_size=imgsz,
                        warmup_iters=warmup_iters,
                        benchmark_iters=benchmark_iters,
                        device=device,
                        seed=seed,
                    )
                    try:
                        result = self.run_single(config, data_path=data_path)
                        results.append(result)

                        # Save checkpoint after each result
                        if checkpoint_path:
                            tmp_path = checkpoint_path + ".tmp"
                            with open(tmp_path, "w") as f:
                                json.dump([r.to_dict() for r in results], f, indent=2)
                            os.rename(tmp_path, checkpoint_path)

                    except Exception as e:
                        logger.error(f"Failed: {model_name}/{precision}/{imgsz}: {e}")

        logger.info(f"Sweep complete: {len(results)}/{total} successful")
        return results


def run_quick_benchmark(
    model_name: str = "yolov8n",
    image_size: int = 640,
    precision: str = "fp16",
    iters: int = 50,
    device: str = "cpu",
) -> BenchmarkResult:
    """
    Quick single-model benchmark for testing.
    Defaults to CPU for login-node compatibility.
    """
    runner = InferenceBenchmarkRunner(verbose=True)
    config = BenchmarkConfig(
        model_name=model_name,
        precision=precision,
        image_size=image_size,
        warmup_iters=10,
        benchmark_iters=iters,
        device=device,
    )
    return runner.run_single(config)
