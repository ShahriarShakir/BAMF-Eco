"""
Utility constants, configuration, and helper functions for BAMF-Eco.
"""

import os
import json
import hashlib
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import yaml
import numpy as np
from loguru import logger


# ===========================================================================
# Path Configuration
# ===========================================================================

# Detect HPC cluster vs local
_ON_HPC = os.environ.get("BAMF_ECO_HPC", "").lower() == "true"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_ROOT = Path(os.environ.get("BAMF_ECO_ROOT", str(PROJECT_ROOT / "scratch"))) if _ON_HPC else PROJECT_ROOT / "scratch"
MODEL_DIR = SCRATCH_ROOT / "models"
OUTPUT_DIR = SCRATCH_ROOT / "outputs"
CACHE_DIR = SCRATCH_ROOT / "cache"
CONFIG_DIR = PROJECT_ROOT / "configs"

# Ensure directories exist
for _d in [MODEL_DIR, OUTPUT_DIR, CACHE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Ultralytics config (avoid polluting home directory)
os.environ.setdefault("YOLO_CONFIG_DIR", str(SCRATCH_ROOT))
os.environ.setdefault("TORCH_HOME", str(SCRATCH_ROOT / "torch_cache"))


# ===========================================================================
# Model Registry
# ===========================================================================

@dataclass
class ModelSpec:
    """Specification for a single model variant."""
    family: str          # e.g., "yolov8", "yolov10", "yolo11", "yolo26", "rtdetr"
    size: str            # e.g., "n", "s", "m", "l", "x", "t", "c", "e"
    name: str            # e.g., "yolov8n"
    weight_file: str     # e.g., "yolov8n.pt"
    params_m: float      # Parameters in millions
    arch_type: str       # "anchor-free", "nms-free", "pgilan", "transformer", "c3k2"
    loader: str          # "YOLO" or "RTDETR"


MODEL_REGISTRY: Dict[str, ModelSpec] = {}

def _register_models():
    """Populate model registry with all supported variants."""
    families = {
        "yolov8": {
            "sizes": {"n": 3.2, "s": 11.2, "m": 25.9, "l": 43.7},
            "arch_type": "anchor-free",
            "loader": "YOLO",
        },
        "yolov9": {
            "sizes": {"t": 2.1, "s": 7.3, "m": 20.2, "c": 25.6, "e": 58.2},
            "arch_type": "pgi-gelan",
            "loader": "YOLO",
        },
        "yolov10": {
            "sizes": {"n": 2.8, "s": 8.1, "m": 16.6, "l": 25.9},
            "arch_type": "nms-free",
            "loader": "YOLO",
        },
        "yolo11": {
            "sizes": {"n": 2.6, "s": 9.5, "m": 20.1, "l": 25.4},
            "arch_type": "c3k2-sppf",
            "loader": "YOLO",
        },
        "yolo26": {
            "sizes": {"n": 2.6, "s": 10.0, "m": 21.9, "l": 26.3},
            "arch_type": "latest-gen",
            "loader": "YOLO",
        },
        "rtdetr": {
            "sizes": {"l": 33.0, "x": 67.5},
            "arch_type": "transformer",
            "loader": "RTDETR",
        },
    }

    for family, info in families.items():
        for size, params in info["sizes"].items():
            if family == "rtdetr":
                name = f"rtdetr-{size}"
            else:
                name = f"{family}{size}"
            spec = ModelSpec(
                family=family,
                size=size,
                name=name,
                weight_file=f"{name}.pt",
                params_m=params,
                arch_type=info["arch_type"],
                loader=info["loader"],
            )
            MODEL_REGISTRY[name] = spec

_register_models()


def get_model_specs(families: Optional[List[str]] = None,
                    sizes: Optional[List[str]] = None) -> List[ModelSpec]:
    """Filter model registry by family and/or size."""
    specs = list(MODEL_REGISTRY.values())
    if families:
        specs = [s for s in specs if s.family in families]
    if sizes:
        specs = [s for s in specs if s.size in sizes]
    return specs


def load_model(spec: ModelSpec, device: str = "cpu"):
    """Load a model from the registry."""
    from ultralytics import YOLO, RTDETR

    weight_path = MODEL_DIR / spec.weight_file
    if not weight_path.exists():
        # Let ultralytics download it
        weight_path = spec.weight_file

    LoaderClass = YOLO if spec.loader == "YOLO" else RTDETR
    model = LoaderClass(str(weight_path))
    return model


# ===========================================================================
# Configuration Management
# ===========================================================================

@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    # Experiment identity
    experiment_id: str = ""
    experiment_name: str = ""
    timestamp: str = ""
    seed: int = 42

    # Model selection
    model_families: List[str] = field(default_factory=lambda: [
        "yolov8", "yolov9", "yolov10", "yolo11", "yolo26", "rtdetr"
    ])
    model_sizes: List[str] = field(default_factory=lambda: ["n", "s", "m", "l"])

    # Precision modes
    precisions: List[str] = field(default_factory=lambda: ["fp32", "fp16"])

    # Image sizes
    image_sizes: List[int] = field(default_factory=lambda: [320, 416, 512, 640])

    # Benchmark settings
    warmup_iters: int = 50
    benchmark_iters: int = 200
    num_repeats: int = 3

    # Dataset
    dataset: str = "coco"
    dataset_path: str = ""
    target_classes: List[int] = field(default_factory=lambda: [0])  # person class

    # Sustainability parameters
    grid_carbon_factor: float = 0.79        # kg CO₂e per kWh (regional avg)
    grid_carbon_factor_min: float = 0.60
    grid_carbon_factor_max: float = 1.00
    water_factor: float = 3.5               # liters per kWh (data center cooling)
    water_factor_min: float = 1.8
    water_factor_max: float = 5.0

    # Hardware
    hardware_name: str = ""
    gpu_name: str = ""

    # Output
    output_dir: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not self.experiment_id:
            self.experiment_id = f"exp_{self.timestamp}_{self._config_hash()[:8]}"
        if not self.output_dir:
            self.output_dir = str(OUTPUT_DIR / self.experiment_id)

    def _config_hash(self) -> str:
        """Deterministic hash of configuration for reproducibility."""
        d = asdict(self)
        d.pop("timestamp", None)
        d.pop("experiment_id", None)
        d.pop("output_dir", None)
        return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()

    def save(self, path: Optional[str] = None):
        """Save config to YAML."""
        path = path or os.path.join(self.output_dir, "config.yaml")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
        logger.info(f"Config saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load config from YAML."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


def load_config(path: str) -> ExperimentConfig:
    """Load experiment config from YAML file."""
    return ExperimentConfig.load(path)


# ===========================================================================
# Reproducibility
# ===========================================================================

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ===========================================================================
# Timing utilities
# ===========================================================================

class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed: float = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        if self.name:
            logger.debug(f"[Timer] {self.name}: {self.elapsed:.4f}s")


# ===========================================================================
# Result storage
# ===========================================================================

@dataclass
class BenchmarkResult:
    """Single benchmark evaluation result."""
    model_name: str
    family: str
    size: str
    precision: str
    image_size: int
    batch_size: int

    # Accuracy
    ap: float = 0.0
    ap50: float = 0.0
    ap75: float = 0.0

    # Latency & throughput
    latency_ms_mean: float = 0.0
    latency_ms_std: float = 0.0
    throughput_fps: float = 0.0

    # Memory
    peak_memory_mb: float = 0.0

    # Energy & sustainability
    energy_kwh: float = 0.0
    power_watts_mean: float = 0.0
    power_watts_std: float = 0.0
    co2e_kg: float = 0.0
    co2e_kg_min: float = 0.0
    co2e_kg_max: float = 0.0
    water_liters: float = 0.0
    water_liters_min: float = 0.0
    water_liters_max: float = 0.0

    # Metadata
    hardware: str = ""
    timestamp: str = ""
    duration_s: float = 0.0
    num_images: int = 0
    params_m: float = 0.0
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def save_results(results: List[BenchmarkResult], path: str):
    """Save results to JSON."""
    import pandas as pd
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([r.to_dict() for r in results])
    df.to_csv(path.replace(".json", ".csv"), index=False)
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    logger.info(f"Saved {len(results)} results to {path}")


def load_results(path: str) -> List[BenchmarkResult]:
    """Load results from JSON."""
    with open(path) as f:
        data = json.load(f)
    return [BenchmarkResult.from_dict(d) for d in data]
