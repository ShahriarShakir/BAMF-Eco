"""
Baseline Optimizers
=====================

Implements comparison baselines for the BAMF-Eco paper:

1. RandomSearch: Uniform random configuration sampling
2. HyperbandOptimizer: Multi-fidelity Successive Halving / Hyperband
3. BOHBOptimizer: Bayesian Optimization + HyperBand
4. DEHBOptimizer: Differential Evolution + HyperBand
5. EHVIOptimizer: Expected Hypervolume Improvement (no eco constraints)
6. SingleObjectiveBO: BO for a single objective (accuracy or efficiency)
7. ManualExpertBaseline: Fixed selection based on common practice

All baselines implement the same interface for fair comparison with BAMF-Eco.
"""

import time
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod

import numpy as np
from loguru import logger

from bamf_eco.optimizer.config_space import ConfigEncoder
from bamf_eco.training import TrainingRunner, TrainingConfig, TrainingResult
from bamf_eco.sustainability import SustainabilityAccountant


@dataclass
class BaselineResult:
    """Result from a baseline optimizer run."""
    optimizer_name: str
    n_evaluations: int = 0
    total_energy_kwh: float = 0.0
    total_co2e_kg: float = 0.0
    total_time_s: float = 0.0
    best_map: float = 0.0
    best_energy: float = float("inf")
    pareto_size: int = 0
    pareto_points: List[Dict[str, float]] = field(default_factory=list)
    hypervolume: float = 0.0
    all_results: List[Dict[str, Any]] = field(default_factory=list)


class BaseOptimizer(ABC):
    """Abstract base class for all optimizers."""

    def __init__(
        self,
        name: str,
        max_evaluations: int = 200,
        seed: int = 42,
    ):
        self.name = name
        self.max_evaluations = max_evaluations
        self.seed = seed
        self.encoder = ConfigEncoder()
        self.rng = np.random.RandomState(seed)
        self.results: List[TrainingResult] = []
        self.configs: List[Dict[str, Any]] = []

    @abstractmethod
    def suggest(self) -> Tuple[Dict[str, Any], float]:
        """Suggest next (config, fidelity) pair."""
        pass

    def observe(self, config: Dict[str, Any], fidelity: float, result: TrainingResult):
        """Record an observation."""
        self.results.append(result)
        self.configs.append(config)

    def run(
        self,
        evaluator: TrainingRunner,
        dataset_path: str = "",
        checkpoint_path: Optional[str] = None,
        walltime_seconds: Optional[float] = None,
        safety_margin_seconds: float = 1800.0,
    ) -> BaselineResult:
        """
        Run the full optimization loop with checkpoint/resume support.

        Args:
            checkpoint_path: Path to save/load checkpoint JSON.
            walltime_seconds: PBS walltime limit in seconds. Stops safely before.
            safety_margin_seconds: Stop this many seconds before walltime (default 30 min).
        """
        job_start_time = time.time()
        start_iter = 0

        # Resume from checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path) as f:
                    ckpt = json.load(f)
                self.results = [TrainingResult.from_dict(r) for r in ckpt.get("results", [])]
                self.configs = ckpt.get("configs", [])
                start_iter = ckpt.get("iteration", len(self.results))
                logger.info(f"[{self.name}] Resumed from iter {start_iter}")
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to load checkpoint: {e}")

        logger.info(f"[{self.name}] Starting from iter {start_iter}, budget={self.max_evaluations}")
        t_start = time.perf_counter()

        for i in range(start_iter, self.max_evaluations):
            # Walltime safety
            if walltime_seconds:
                elapsed = time.time() - job_start_time
                remaining = walltime_seconds - elapsed
                if remaining < safety_margin_seconds:
                    logger.warning(
                        f"[{self.name}] Walltime safety: {remaining/60:.0f}min left. "
                        f"Saving checkpoint at iter {i}."
                    )
                    if checkpoint_path:
                        self._save_baseline_checkpoint(checkpoint_path, i)
                    break

            config, fidelity = self.suggest()

            train_config = TrainingConfig(
                model_name=config.get("model_name", "yolov8n"),
                precision=config.get("precision", "fp16"),
                image_size=config.get("image_size", 640),
                epochs=config.get("epochs", 100),
                fidelity=fidelity,
                optimizer_name=config.get("optimizer", "SGD"),
                lr0=config.get("lr0", 0.01),
                momentum=config.get("momentum", 0.937),
                weight_decay=config.get("weight_decay", 0.0005),
                batch_size=config.get("batch_size", 16),
                dataset_path=dataset_path,
                seed=self.seed + i,
            )

            try:
                result = evaluator.evaluate(train_config)
                self.observe(config, fidelity, result)

                # Save checkpoint after each evaluation
                if checkpoint_path:
                    self._save_baseline_checkpoint(checkpoint_path, i + 1)
            except Exception as e:
                logger.error(f"[{self.name}] Eval {i} failed: {e}")

        total_time = time.perf_counter() - t_start

        return self._compile_results(total_time)

    def _save_baseline_checkpoint(self, path: str, iteration: int):
        """Save baseline checkpoint for resume."""
        ckpt = {
            "optimizer_name": self.name,
            "iteration": iteration,
            "results": [r.to_dict() for r in self.results],
            "configs": self.configs,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(ckpt, f, indent=2, default=str)
        os.rename(tmp_path, path)

    def _compile_results(self, total_time: float) -> BaselineResult:
        """Compile all results into a BaselineResult."""
        total_energy = sum(r.energy_kwh for r in self.results)
        total_co2e = sum(r.co2e_kg for r in self.results)

        best_map = max((r.map50_95 for r in self.results), default=0.0)
        best_energy = min(
            (r.energy_kwh for r in self.results if r.energy_kwh > 0),
            default=float("inf"),
        )

        return BaselineResult(
            optimizer_name=self.name,
            n_evaluations=len(self.results),
            total_energy_kwh=total_energy,
            total_co2e_kg=total_co2e,
            total_time_s=total_time,
            best_map=best_map,
            best_energy=best_energy,
            all_results=[
                {
                    "model": r.model_name,
                    "map": r.map50_95,
                    "energy": r.energy_kwh,
                    "latency": r.latency_ms,
                    "co2e": r.co2e_kg,
                }
                for r in self.results
            ],
        )


class RandomSearch(BaseOptimizer):
    """Uniform random search baseline."""

    def __init__(self, max_evaluations: int = 200, seed: int = 42):
        super().__init__("RandomSearch", max_evaluations, seed)

    def suggest(self) -> Tuple[Dict[str, Any], float]:
        config = self.encoder.random_config(self.rng)
        return config, 1.0  # Always full fidelity


class HyperbandOptimizer(BaseOptimizer):
    """
    Hyperband: Multi-fidelity random with Successive Halving.

    Schedules configs at increasing fidelity levels, promoting top 1/η fraction.
    """

    def __init__(
        self,
        max_evaluations: int = 200,
        eta: int = 3,
        min_fidelity: float = 0.05,
        max_fidelity: float = 1.0,
        seed: int = 42,
    ):
        super().__init__("Hyperband", max_evaluations, seed)
        self.eta = eta
        self.min_fidelity = min_fidelity
        self.max_fidelity = max_fidelity
        self._brackets = self._compute_brackets()
        self._current_bracket = 0
        self._current_rung = 0
        self._rung_configs: List[Dict[str, Any]] = []
        self._rung_results: List[float] = []
        self._eval_count = 0

    def _compute_brackets(self) -> List[List[Tuple[int, float]]]:
        """Compute SH brackets: [(n_configs, fidelity), ...]"""
        import math
        s_max = int(math.log(self.max_fidelity / self.min_fidelity, self.eta))
        brackets = []
        for s in range(s_max + 1):
            bracket = []
            n = int(math.ceil((s_max + 1) / (s + 1) * self.eta ** s))
            for i in range(s + 1):
                n_i = max(1, int(n * self.eta ** (-i)))
                r_i = self.min_fidelity * self.eta ** (s - s + i)
                r_i = min(r_i, self.max_fidelity)
                bracket.append((n_i, r_i))
            brackets.append(bracket)
        return brackets

    def suggest(self) -> Tuple[Dict[str, Any], float]:
        self._eval_count += 1

        if self._current_bracket >= len(self._brackets):
            self._current_bracket = 0  # Restart

        bracket = self._brackets[self._current_bracket]

        if self._current_rung >= len(bracket):
            # Move to next bracket
            self._current_bracket += 1
            self._current_rung = 0
            self._rung_configs = []
            self._rung_results = []
            if self._current_bracket >= len(self._brackets):
                self._current_bracket = 0
            bracket = self._brackets[self._current_bracket]

        n_configs, fidelity = bracket[self._current_rung]

        if len(self._rung_configs) < n_configs:
            # Sample new configs for this rung
            config = self.encoder.random_config(self.rng)
            self._rung_configs.append(config)
            return config, fidelity
        else:
            # Promote top 1/η configs to next rung
            top_k = max(1, len(self._rung_results) // self.eta)
            top_indices = np.argsort(self._rung_results)[-top_k:]
            self._rung_configs = [self._rung_configs[i] for i in top_indices]
            self._rung_results = []
            self._current_rung += 1
            if self._current_rung < len(bracket):
                _, fidelity = bracket[self._current_rung]
                config = self._rung_configs[0] if self._rung_configs else self.encoder.random_config(self.rng)
                return config, fidelity
            else:
                self._current_bracket += 1
                self._current_rung = 0
                self._rung_configs = []
                config = self.encoder.random_config(self.rng)
                return config, self.min_fidelity

    def observe(self, config: Dict[str, Any], fidelity: float, result: TrainingResult):
        super().observe(config, fidelity, result)
        self._rung_results.append(result.map50_95)


class SingleObjectiveBO(BaseOptimizer):
    """
    Single-objective Bayesian Optimization baseline.

    Optimizes one objective (e.g., accuracy OR efficiency) using
    standard BO with GP + Expected Improvement.
    """

    def __init__(
        self,
        objective: str = "mAP",
        direction: str = "maximize",
        max_evaluations: int = 200,
        n_initial: int = 10,
        seed: int = 42,
    ):
        super().__init__(f"SingleObj_{objective}", max_evaluations, seed)
        self.objective = objective
        self.direction = direction
        self.n_initial = n_initial
        self._X = []
        self._y = []

    def suggest(self) -> Tuple[Dict[str, Any], float]:
        if len(self._X) < self.n_initial:
            return self.encoder.random_config(self.rng), 1.0

        # Fit GP and select via EI
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern

            X = np.array(self._X)
            y = np.array(self._y)
            if self.direction == "minimize":
                y = -y

            gpr = GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                normalize_y=True,
                random_state=self.seed,
            )
            gpr.fit(X, y)

            # Generate candidates
            best_ei = -float("inf")
            best_config = None

            for _ in range(500):
                config = self.encoder.random_config(self.rng)
                x = self.encoder.encode(config)
                mu, sigma = gpr.predict(x.reshape(1, -1), return_std=True)

                # Expected improvement
                from scipy.stats import norm
                best_y = y.max()
                z = (mu[0] - best_y) / (sigma[0] + 1e-8)
                ei = (mu[0] - best_y) * norm.cdf(z) + sigma[0] * norm.pdf(z)

                if ei > best_ei:
                    best_ei = ei
                    best_config = config

            return best_config, 1.0

        except Exception as e:
            logger.warning(f"BO suggestion failed, using random: {e}")
            return self.encoder.random_config(self.rng), 1.0

    def observe(self, config: Dict[str, Any], fidelity: float, result: TrainingResult):
        super().observe(config, fidelity, result)
        x = self.encoder.encode(config)
        self._X.append(x)

        obj_map = {
            "mAP": result.map50_95,
            "energy_kwh": result.energy_kwh,
            "latency_ms": result.latency_ms,
        }
        self._y.append(obj_map.get(self.objective, 0))


class ManualExpertBaseline(BaseOptimizer):
    """
    Manual expert baseline: fixed selection of "best practice" configs.

    Represents what a practitioner would do without automated optimization.
    """

    EXPERT_CONFIGS = [
        {"model_name": "yolov8n", "image_size": 640, "precision": "fp16",
         "lr0": 0.01, "batch_size": 16, "optimizer": "SGD", "epochs": 100},
        {"model_name": "yolov8s", "image_size": 640, "precision": "fp16",
         "lr0": 0.01, "batch_size": 16, "optimizer": "SGD", "epochs": 100},
        {"model_name": "yolo11n", "image_size": 640, "precision": "fp16",
         "lr0": 0.01, "batch_size": 16, "optimizer": "AdamW", "epochs": 100},
        {"model_name": "yolo11s", "image_size": 640, "precision": "fp16",
         "lr0": 0.001, "batch_size": 16, "optimizer": "AdamW", "epochs": 100},
        {"model_name": "yolo26n", "image_size": 640, "precision": "fp16",
         "lr0": 0.01, "batch_size": 16, "optimizer": "SGD", "epochs": 100},
        {"model_name": "rtdetr-l", "image_size": 640, "precision": "fp16",
         "lr0": 0.001, "batch_size": 8, "optimizer": "AdamW", "epochs": 100},
    ]

    def __init__(self, seed: int = 42, **kwargs):
        max_evals = len(self.EXPERT_CONFIGS)
        super().__init__("ManualExpert", max_evals, seed)
        self._idx = 0

    def suggest(self) -> Tuple[Dict[str, Any], float]:
        if self._idx < len(self.EXPERT_CONFIGS):
            config = self.EXPERT_CONFIGS[self._idx].copy()
            # Add defaults for missing keys
            config.setdefault("momentum", 0.937)
            config.setdefault("weight_decay", 0.0005)
            config.setdefault("augment_strength", 0.5)
            self._idx += 1
            return config, 1.0
        else:
            return self.encoder.random_config(self.rng), 1.0


def get_baseline(name: str, **kwargs) -> BaseOptimizer:
    """Factory function for baselines."""
    baselines = {
        "random": RandomSearch,
        "hyperband": HyperbandOptimizer,
        "single_obj_accuracy": lambda **kw: SingleObjectiveBO(
            objective="mAP", direction="maximize", **kw
        ),
        "single_obj_efficiency": lambda **kw: SingleObjectiveBO(
            objective="energy_kwh", direction="minimize", **kw
        ),
        "manual_expert": ManualExpertBaseline,
    }

    if name not in baselines:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(baselines.keys())}")

    return baselines[name](**kwargs)
