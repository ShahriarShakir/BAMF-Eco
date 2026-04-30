"""
BAMF-Eco Optimizer
=====================

Budget-Aware Multi-Fidelity Eco-Optimizer for sustainable object detection.

Core loop:
1. Sample/select configuration via eco-acquisition function
2. Choose fidelity level (Successive Halving schedule)
3. Evaluate (train model, measure energy)
4. Update GP surrogates + fidelity correction + feasibility
5. Check early termination (Theorem 3: Pareto dominance)
6. Repeat until budget exhausted

Key components:
- EcoAcquisition: Eco-feasibility EHVI (Theorem 1)
- FidelityCorrectionModel: Low→High fidelity prediction (Theorem 2)
- ParetoEarlyTermination: Stop when no Pareto improvement (Theorem 3)
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime

import numpy as np
from loguru import logger

from bamf_eco.optimizer.config_space import ConfigEncoder, DEFAULT_SEARCH_SPACE
from bamf_eco.optimizer.acquisition import EcoAcquisition, EcoConstraints
from bamf_eco.optimizer.fidelity_correction import (
    MultiFidelityCorrectionManager,
    CorrectionObservation,
    FidelityCorrectionModel,
)
from bamf_eco.training import TrainingRunner, TrainingConfig, TrainingResult
from bamf_eco.sustainability import SustainabilityAccountant


@dataclass
class OptimizationState:
    """Persistent state of the optimization loop."""
    iteration: int = 0
    total_energy_kwh: float = 0.0
    total_co2e_kg: float = 0.0
    total_water_liters: float = 0.0
    total_training_time_s: float = 0.0
    n_full_fidelity: int = 0
    n_low_fidelity: int = 0
    pareto_size: int = 0
    best_map: float = 0.0
    best_efficiency: float = float("inf")
    stagnation_count: int = 0
    terminated_early: bool = False
    termination_reason: str = ""


@dataclass
class EvaluationRecord:
    """Record of a single evaluation."""
    iteration: int
    config: Dict[str, Any]
    config_vec: np.ndarray
    fidelity: float
    result: Optional[TrainingResult] = None
    is_feasible: bool = True
    acquisition_value: float = 0.0
    predicted_objectives: Optional[Dict[str, float]] = None
    prediction_error: Optional[Dict[str, float]] = None
    timestamp: str = ""


class ParetoFront:
    """
    Maintains the Pareto front of evaluated solutions.

    Used for:
    - EHVI reference computation
    - Hypervolume tracking
    - Early termination detection (Theorem 3)
    """

    def __init__(self, objectives: List[str], directions: List[str]):
        self.objectives = objectives
        self.directions = directions
        self.points: List[Dict[str, float]] = []
        self.configs: List[Dict[str, Any]] = []
        self._hypervolumes: List[float] = []

    def update(
        self,
        objectives: Dict[str, float],
        config: Dict[str, Any],
    ) -> bool:
        """
        Add a point and update the Pareto front.

        Returns True if the Pareto front was modified.
        """
        # Check if this point is dominated
        if self._is_dominated(objectives):
            return False

        # Remove points dominated by this one
        surviving = []
        surviving_configs = []
        for i, pt in enumerate(self.points):
            if not self._dominates(objectives, pt):
                surviving.append(pt)
                surviving_configs.append(self.configs[i])

        surviving.append(objectives)
        surviving_configs.append(config)

        old_size = len(self.points)
        self.points = surviving
        self.configs = surviving_configs

        # Track hypervolume
        hv = self._compute_hypervolume()
        self._hypervolumes.append(hv)

        return True

    def _dominates(self, a: Dict[str, float], b: Dict[str, float]) -> bool:
        """Does point a dominate point b?"""
        dominated = True
        strictly_better = False
        for obj, direction in zip(self.objectives, self.directions):
            va, vb = a.get(obj, 0), b.get(obj, 0)
            if direction == "maximize":
                if va < vb:
                    dominated = False
                if va > vb:
                    strictly_better = True
            else:  # minimize
                if va > vb:
                    dominated = False
                if va < vb:
                    strictly_better = True
        return dominated and strictly_better

    def _is_dominated(self, point: Dict[str, float]) -> bool:
        """Is the point dominated by any existing Pareto point?"""
        for pt in self.points:
            if self._dominates(pt, point):
                return True
        return False

    def _compute_hypervolume(self) -> float:
        """Compute hypervolume indicator (2D fast, general fallback)."""
        if not self.points:
            return 0.0

        n_obj = len(self.objectives)
        Y = np.zeros((len(self.points), n_obj))
        for i, pt in enumerate(self.points):
            for j, (obj, direction) in enumerate(zip(self.objectives, self.directions)):
                val = pt.get(obj, 0)
                if direction == "minimize":
                    val = -val  # Flip for HV computation (always maximize)
                Y[i, j] = val

        if n_obj == 2:
            return self._hypervolume_2d(Y)
        else:
            # Approximate using Monte Carlo for higher dimensions
            return self._hypervolume_mc(Y, n_samples=10000)

    def _hypervolume_2d(self, Y: np.ndarray) -> float:
        """Exact 2D hypervolume computation with fixed reference point."""
        # Sort by first objective descending
        order = np.argsort(-Y[:, 0])
        Y_sorted = Y[order]

        # Use fixed reference point for comparable HV across iterations
        ref = self._get_reference_point(Y.shape[1])
        hv = 0.0
        prev_y2 = ref[1]
        for i in range(len(Y_sorted)):
            hv += (Y_sorted[i, 0] - ref[0]) * (Y_sorted[i, 1] - prev_y2)
            prev_y2 = max(prev_y2, Y_sorted[i, 1])

        return abs(hv)

    def _hypervolume_mc(self, Y: np.ndarray, n_samples: int = 10000) -> float:
        """Monte Carlo hypervolume approximation with fixed reference point."""
        ref = self._get_reference_point(Y.shape[1])
        ideal = np.max(Y, axis=0) + 0.1

        samples = np.random.uniform(ref, ideal, size=(n_samples, Y.shape[1]))
        dominated = np.zeros(n_samples, dtype=bool)

        for pt in Y:
            dominated |= np.all(samples <= pt, axis=1)

        volume_ratio = dominated.mean()
        total_volume = np.prod(ideal - ref)
        return volume_ratio * total_volume

    def _get_reference_point(self, n_dim: int) -> np.ndarray:
        """Fixed reference point for hypervolume computation.

        Using fixed values ensures HV is comparable across iterations.
        For maximize objectives (flipped to positive): ref = 0
        For minimize objectives (flipped to negative): ref = -max_bound
        """
        # Conservative fixed reference: worst plausible values
        # After direction-flipping, all are maximized, so ref should be below all points
        defaults = {
            "mAP": 0.0,           # worst accuracy
            "energy_kwh": -0.1,   # after flip: -max_energy
            "latency_ms": -500.0, # after flip: -max_latency
            "co2e_kg": -1.0,      # after flip: -max_co2
            "water_liters": -10.0, # after flip: -max_water
        }
        ref = np.zeros(n_dim)
        for j, (obj, direction) in enumerate(zip(self.objectives, self.directions)):
            if j < n_dim:
                if direction == "minimize":
                    ref[j] = defaults.get(obj, -1.0)
                else:
                    ref[j] = defaults.get(obj, 0.0)
        return ref

    @property
    def hypervolume(self) -> float:
        return self._hypervolumes[-1] if self._hypervolumes else 0.0

    @property
    def hypervolume_history(self) -> List[float]:
        return self._hypervolumes.copy()


class BAMFEcoOptimizer:
    """
    Budget-Aware Multi-Fidelity Eco-Optimizer.

    Main algorithm that orchestrates the optimization loop.
    """

    def __init__(
        self,
        objectives: Optional[List[str]] = None,
        directions: Optional[List[str]] = None,
        reference_point: Optional[Dict[str, float]] = None,
        constraints: Optional[EcoConstraints] = None,
        # Multi-fidelity settings
        fidelity_levels: Optional[List[float]] = None,
        eta: int = 3,
        correction_method: str = "gp",
        correction_start_after: int = 15,
        # Budget
        max_evaluations: int = 200,
        total_budget_ksu: float = 10.0,
        # Early termination
        early_termination: bool = True,
        patience: int = 10,
        min_evaluations: int = 30,
        # Initialization
        n_initial_random: int = 10,
        n_candidates: int = 1000,
        seed: int = 42,
        # Output
        output_dir: Optional[str] = None,
    ):
        # Objectives
        self.objectives = objectives or ["mAP", "energy_kwh", "latency_ms"]
        self.directions = directions or ["maximize", "minimize", "minimize"]
        self.reference_point = reference_point or {
            "mAP": 0.0, "energy_kwh": 0.1, "latency_ms": 500.0
        }

        # Constraints
        self.constraints = constraints or EcoConstraints()

        # Multi-fidelity
        self.fidelity_levels = fidelity_levels or [0.05, 0.1, 0.2, 0.5, 1.0]
        self.eta = eta
        self.correction_start_after = correction_start_after

        # Budget
        self.max_evaluations = max_evaluations
        self.total_budget_ksu = total_budget_ksu

        # Early termination
        self.early_termination_enabled = early_termination
        self.patience = patience
        self.min_evaluations = min_evaluations

        # Initialization
        self.n_initial_random = n_initial_random
        self.n_candidates = n_candidates
        self.seed = seed

        # Components
        self.encoder = ConfigEncoder()
        self.acquisition = EcoAcquisition(
            reference_point=self.reference_point,
            objectives=self.objectives,
            directions=self.directions,
            constraints=self.constraints,
        )
        self.correction_manager = MultiFidelityCorrectionManager(
            objectives=self.objectives,
            method=correction_method,
            n_features=self.encoder.encoded_dim,
        )
        self.pareto_front = ParetoFront(
            objectives=self.objectives,
            directions=self.directions,
        )

        # State
        self.state = OptimizationState()
        self.history: List[EvaluationRecord] = []
        self.rng = np.random.RandomState(seed)

        # Output
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_fidelity_schedule(self) -> List[float]:
        """
        Determine fidelity levels following Successive Halving.

        Early iterations: low fidelity (cheap exploration)
        Later iterations: high fidelity (expensive exploitation)
        """
        n = self.state.iteration
        total = self.max_evaluations

        if n < self.n_initial_random:
            # Initial random phase: lowest fidelity
            return [self.fidelity_levels[0]]
        elif n < total * 0.25:
            return self.fidelity_levels[:2]
        elif n < total * 0.5:
            return self.fidelity_levels[:3]
        elif n < total * 0.75:
            return self.fidelity_levels[:4]
        else:
            return self.fidelity_levels  # All fidelities

    def _should_promote(self) -> Optional[float]:
        """
        Successive Halving promotion schedule (Theorem 2).

        Forces periodic evaluations at increasing fidelity to:
        1. Bootstrap the fidelity correction model with paired observations
        2. Validate low-fidelity discoveries at full fidelity
        3. Populate the Pareto front with reliable measurements

        Returns target fidelity if promotion is needed, None otherwise.
        """
        n = self.state.iteration
        if n < self.n_initial_random:
            return None

        # Offset from end of initial phase
        phase_iter = n - self.n_initial_random

        # Tier 1: Every 5th iteration → fidelity 0.2 (medium-low)
        # Tier 2: Every 10th iteration → fidelity 0.5 (medium-high)
        # Tier 3: Every 20th iteration → fidelity 1.0 (full)
        # This ensures ~15% of evals are at elevated fidelity
        if phase_iter > 0 and phase_iter % 20 == 0:
            return self.fidelity_levels[-1]  # Full fidelity (1.0)
        elif phase_iter > 0 and phase_iter % 10 == 0:
            mid_high = min(len(self.fidelity_levels) - 1, 3)
            return self.fidelity_levels[mid_high]  # 0.5
        elif phase_iter > 0 and phase_iter % 5 == 0:
            mid_low = min(len(self.fidelity_levels) - 1, 2)
            return self.fidelity_levels[mid_low]  # 0.2

        return None

    def _get_promotion_config(self) -> Optional[Dict[str, Any]]:
        """
        Select the most promising low-fidelity config for promotion.

        Picks the config with the highest predicted performance from
        low-fidelity evaluations that hasn't been evaluated at a
        higher fidelity yet. This is the core Successive Halving idea:
        cheap exploration → expensive exploitation of winners.
        """
        if not self.history:
            return None

        # Track which configs have been promoted (by model_name + key params)
        def _config_key(cfg):
            return (
                cfg.get("model_name", ""),
                cfg.get("image_size", 0),
                cfg.get("batch_size", 0),
                round(cfg.get("lr0", 0), 6),
            )

        promoted_keys = set()
        low_fid_candidates = []

        for rec in self.history:
            key = _config_key(rec.config)
            if rec.fidelity >= 0.5:
                promoted_keys.add(key)

        for rec in self.history:
            key = _config_key(rec.config)
            if rec.fidelity < 0.5 and key not in promoted_keys and rec.result:
                low_fid_candidates.append(rec)

        if not low_fid_candidates:
            # All configs already promoted, return None to explore new
            return None

        # Pick the best by primary objective (mAP)
        best_rec = max(low_fid_candidates, key=lambda r: r.result.map50_95)
        return best_rec.config

    def _select_fidelity(self, config_vec: np.ndarray) -> float:
        """
        Select fidelity level for a candidate.

        Uses cost-aware selection: prefers low fidelity unless
        correction model is confident enough.

        Note: Forced promotion schedule is handled in suggest(), not here.
        This method handles the default case for non-promotion iterations.
        """
        available = self._get_fidelity_schedule()

        if not self.correction_manager.models.get(self.objectives[0]):
            # No correction model yet, use lowest available
            return available[0]

        # Use correction uncertainty to decide
        primary_obj = self.objectives[0]  # Usually mAP
        model = self.correction_manager.models[primary_obj]

        if model.is_fitted:
            # Estimate the metric at each fidelity using historical mean
            # (avoids the meaningless low_metric=0.0 placeholder)
            estimated_metric = self._estimate_metric_for_config(config_vec)

            # Predict at each fidelity, pick the one with best cost/info ratio
            best_fidelity = available[0]
            best_ratio = -float("inf")

            for fid in available:
                _, uncertainty = model.predict(config_vec, fid, estimated_metric)
                info_gain = 1.0 / (uncertainty + 0.01)
                cost = fid  # Cost proportional to fidelity
                ratio = info_gain / cost
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_fidelity = fid

            return best_fidelity

        return available[0]

    def _estimate_metric_for_config(self, config_vec: np.ndarray) -> float:
        """
        Estimate the primary metric for a config, used as a reasonable prior
        when selecting fidelity (instead of the meaningless 0.0).

        Uses the nearest observed config's metric, or the global mean.
        """
        if not self.history:
            return 0.3  # Reasonable mAP prior

        # Find the nearest config in encoded space (L2 distance)
        best_dist = float("inf")
        best_metric = 0.3

        for rec in self.history:
            if rec.result is not None and rec.config_vec is not None:
                dist = np.linalg.norm(config_vec - rec.config_vec)
                if dist < best_dist:
                    best_dist = dist
                    best_metric = rec.result.map50_95

        return best_metric

    @staticmethod
    def _configs_similar(config_a: Dict[str, Any], config_b: Dict[str, Any]) -> bool:
        """
        Check if two configs are similar enough for correction pairing.

        Requires matching model_name. If both configs have image_size/batch_size,
        those must also match.
        """
        if config_a.get("model_name") != config_b.get("model_name"):
            return False

        # If both have image_size, they must match
        if "image_size" in config_a and "image_size" in config_b:
            if str(config_a["image_size"]) != str(config_b["image_size"]):
                return False

        # If both have batch_size, they must match
        if "batch_size" in config_a and "batch_size" in config_b:
            if str(config_a["batch_size"]) != str(config_b["batch_size"]):
                return False

        return True

    def _extract_objectives(self, result: TrainingResult) -> Dict[str, float]:
        """Extract objective values from a training result."""
        obj_map = {
            "mAP": result.map50_95,
            "energy_kwh": result.energy_kwh,
            "latency_ms": result.latency_ms,
            "co2e_kg": result.co2e_kg,
            "water_liters": result.water_liters,
        }
        return {obj: obj_map.get(obj, 0.0) for obj in self.objectives}

    def _check_early_termination(self) -> bool:
        """
        Theorem 3: Pareto Dominance Early Termination.

        Terminate if no Pareto improvement for `patience` evaluations
        AND hypervolume improvement rate < ε.
        """
        if not self.early_termination_enabled:
            return False
        if self.state.iteration < self.min_evaluations:
            return False

        hvs = self.pareto_front.hypervolume_history
        if len(hvs) < self.patience:
            return False

        # Check if hypervolume has stagnated
        recent = hvs[-self.patience:]
        if len(set([round(h, 8) for h in recent])) <= 1:
            self.state.terminated_early = True
            self.state.termination_reason = (
                f"Pareto front stagnant for {self.patience} evaluations "
                f"(HV={hvs[-1]:.6f})"
            )
            return True

        # Check improvement rate
        improvement_rate = (recent[-1] - recent[0]) / (abs(recent[0]) + 1e-8)
        if improvement_rate < 1e-4:
            self.state.terminated_early = True
            self.state.termination_reason = (
                f"HV improvement rate < 0.01% ({improvement_rate:.6f})"
            )
            return True

        return False

    def suggest(self) -> Tuple[Dict[str, Any], float]:
        """
        Suggest the next configuration and fidelity to evaluate.

        Implements the BAMF-Eco suggest loop (Algorithm 1):
        1. Check if Successive Halving promotion is due
        2. If promoting: re-evaluate best low-fidelity config at higher fidelity
        3. Otherwise: standard acquisition-based suggestion

        Returns:
            (config_dict, fidelity)
        """
        if self.state.iteration < self.n_initial_random:
            # Random initialization phase
            config = self.encoder.random_config(self.rng)
            fidelity = self.fidelity_levels[0]
            return config, fidelity

        # ===== Successive Halving Promotion Check =====
        promotion_fidelity = self._should_promote()
        if promotion_fidelity is not None:
            # Re-evaluate the best low-fidelity config at higher fidelity
            promotion_config = self._get_promotion_config()
            if promotion_config is not None:
                logger.info(
                    f"  Promotion: {promotion_config.get('model_name')} "
                    f"→ fidelity={promotion_fidelity:.2f}"
                )
                return promotion_config, promotion_fidelity
            # No config to promote — fall through to standard suggestion
            # but still use the higher fidelity for the new config
            config = self.encoder.random_config(self.rng)
            return config, promotion_fidelity

        # ===== Standard acquisition-based suggestion =====
        # Generate random candidates
        candidates = []
        candidate_vecs = []
        for _ in range(self.n_candidates):
            config = self.encoder.random_config(self.rng)
            vec = self.encoder.encode(config)
            candidates.append(config)
            candidate_vecs.append(vec)

        candidate_vecs = np.array(candidate_vecs)

        # Select fidelity for each candidate
        fidelity_costs = np.array([
            self._select_fidelity(v) for v in candidate_vecs
        ])

        # Select best via acquisition function
        selected_indices = self.acquisition.select_next(
            candidate_vecs, fidelity_costs, n_select=1
        )
        idx = selected_indices[0]

        config = candidates[idx]
        fidelity = float(fidelity_costs[idx])

        return config, fidelity

    def observe(
        self,
        config: Dict[str, Any],
        fidelity: float,
        result: TrainingResult,
    ):
        """
        Record an evaluation result and update all models.

        Args:
            config: Configuration that was evaluated
            fidelity: Fidelity level used
            result: Training result
        """
        config_vec = self.encoder.encode(config)
        objectives = self._extract_objectives(result)

        # Check feasibility
        is_feasible = self.acquisition.feasibility_gp.is_feasible({
            "energy_kwh": result.energy_kwh,
            "co2e_kg": result.co2e_kg,
            "latency_ms": result.latency_ms,
        })

        # Create record
        record = EvaluationRecord(
            iteration=self.state.iteration,
            config=config,
            config_vec=config_vec,
            fidelity=fidelity,
            result=result,
            is_feasible=is_feasible,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Update Pareto front
        pareto_updated = False
        if fidelity >= 0.9:
            # Full-fidelity: directly update Pareto front (reliable)
            pareto_updated = self.pareto_front.update(objectives, config)
            self.state.n_full_fidelity += 1

            # Also create correction pairs with past low-fidelity evals of same config
            self._create_correction_pairs_from_promotion(
                config, config_vec, fidelity, objectives
            )
        else:
            # Low-fidelity: use correction model if available
            self.state.n_low_fidelity += 1
            if self.correction_manager.models.get(self.objectives[0]) and \
               self.correction_manager.models[self.objectives[0]].is_fitted:
                corrected, _ = self.correction_manager.predict_all(
                    config_vec, fidelity, objectives
                )
                record.predicted_objectives = corrected
                pareto_updated = self.pareto_front.update(corrected, config)
            # else: no correction model yet, pareto_updated stays False

        # Update stagnation counter
        if pareto_updated:
            self.state.stagnation_count = 0
        else:
            self.state.stagnation_count += 1

        # Accumulate sustainability metrics
        self.state.total_energy_kwh += result.energy_kwh
        self.state.total_co2e_kg += result.co2e_kg
        self.state.total_water_liters += result.water_liters
        self.state.total_training_time_s += result.training_time_s

        # Track best
        if result.map50_95 > self.state.best_map:
            self.state.best_map = result.map50_95
        if result.energy_kwh < self.state.best_efficiency and result.energy_kwh > 0:
            self.state.best_efficiency = result.energy_kwh

        # Store record
        self.history.append(record)
        self.state.iteration += 1
        self.state.pareto_size = len(self.pareto_front.points)

        # Update surrogates (refit every N evaluations)
        if self.state.iteration % 5 == 0 and len(self.history) >= 5:
            self._refit_surrogates()

        # Update correction model: pair low+high fidelity observations
        # Triggered when we have ANY full-fidelity evals (not gated by threshold)
        if fidelity < 0.9 and self.state.n_full_fidelity >= 1:
            self._update_correction_pairs(config_vec, fidelity, objectives, result.model_name)

        logger.info(
            f"  Iter {self.state.iteration}: {result.model_name} "
            f"fid={fidelity:.2f} | mAP={result.map50_95:.4f} | "
            f"energy={result.energy_kwh:.6f}kWh | "
            f"pareto={self.state.pareto_size} | "
            f"stagnation={self.state.stagnation_count}"
        )

    def _refit_surrogates(self):
        """Refit GP surrogates on all data."""
        n = len(self.history)
        d = self.encoder.encoded_dim
        m = len(self.objectives)

        X = np.zeros((n, d))
        Y = np.zeros((n, m))
        feasibility = np.zeros(n)

        for i, rec in enumerate(self.history):
            X[i] = rec.config_vec
            objectives = self._extract_objectives(rec.result) if rec.result else {}
            for j, obj in enumerate(self.objectives):
                Y[i, j] = objectives.get(obj, 0)
            feasibility[i] = 1.0 if rec.is_feasible else 0.0

        self.acquisition.fit_surrogates(X, Y, feasibility)

    def _update_correction_pairs(
        self,
        config_vec: np.ndarray,
        fidelity: float,
        low_metrics: Dict[str, float],
        model_name: str,
    ):
        """Find or create paired observations for fidelity correction."""
        # Find full-fidelity evaluations with similar configs
        # Match by model_name + image_size + batch_size for tighter pairing
        for rec in self.history:
            if rec.fidelity >= 0.9 and rec.result is not None:
                if self._configs_similar(rec.config, {"model_name": model_name}):
                    high_metrics = self._extract_objectives(rec.result)
                    self.correction_manager.add_paired_observation(
                        config_vec=config_vec,
                        low_fidelity=fidelity,
                        high_fidelity=rec.fidelity,
                        low_metrics=low_metrics,
                        high_metrics=high_metrics,
                        model_name=model_name,
                    )
                    break

        # Refit correction models periodically
        if self.state.iteration % 10 == 0:
            r2_scores = self.correction_manager.fit_all()
            if r2_scores:
                logger.info(f"  Correction R²: {r2_scores}")

    def _create_correction_pairs_from_promotion(
        self,
        config: Dict[str, Any],
        config_vec: np.ndarray,
        high_fidelity: float,
        high_metrics: Dict[str, float],
    ):
        """
        When a config is evaluated at high fidelity (promotion), pair it
        with any previous low-fidelity evaluations of the same or similar config.

        This is key for training the fidelity correction model (Theorem 2).
        """
        model_name = config.get("model_name", "")
        pairs_created = 0

        for rec in self.history:
            if rec.fidelity < high_fidelity and rec.result is not None:
                if self._configs_similar(rec.config, config):
                    low_metrics = self._extract_objectives(rec.result)
                    self.correction_manager.add_paired_observation(
                        config_vec=rec.config_vec,
                        low_fidelity=rec.fidelity,
                        high_fidelity=high_fidelity,
                        low_metrics=low_metrics,
                        high_metrics=high_metrics,
                        model_name=model_name,
                    )
                    pairs_created += 1

        # Refit correction models if we have enough pairs
        if pairs_created > 0:
            r2_scores = self.correction_manager.fit_all()
            if r2_scores:
                logger.info(
                    f"  Correction model trained: {pairs_created} pairs, "
                    f"R²={r2_scores}"
                )
                logger.info(f"  Correction R²: {r2_scores}")

    def run(
        self,
        evaluator: TrainingRunner,
        dataset_path: str = "",
        resume: bool = True,
        walltime_seconds: Optional[float] = None,
        safety_margin_seconds: float = 1800.0,
    ) -> OptimizationState:
        """
        Run the full optimization loop with checkpoint/resume support.

        Args:
            evaluator: TrainingRunner to perform evaluations
            dataset_path: Path to dataset YAML
            resume: If True, attempt to load checkpoint and resume
            walltime_seconds: PBS walltime limit in seconds (e.g. 48*3600=172800).
                              If set, will stop safely before walltime expires.
            safety_margin_seconds: Stop this many seconds before walltime (default 30 min).

        Returns:
            Final optimization state
        """
        job_start_time = time.time()

        # Attempt resume from checkpoint
        if resume:
            resumed = self.load_checkpoint()
            if resumed:
                logger.info(f"Resuming from iteration {self.state.iteration}")
            else:
                logger.info("Starting fresh optimization run")

        logger.info(
            f"BAMF-Eco optimization: "
            f"iter={self.state.iteration}/{self.max_evaluations} | "
            f"budget={self.total_budget_ksu}KSU | "
            f"objectives={self.objectives}"
        )
        if walltime_seconds:
            logger.info(
                f"Walltime limit: {walltime_seconds/3600:.1f}h | "
                f"Safety margin: {safety_margin_seconds/60:.0f}min"
            )

        while self.state.iteration < self.max_evaluations:
            # Check walltime safety
            if walltime_seconds:
                elapsed = time.time() - job_start_time
                remaining = walltime_seconds - elapsed
                if remaining < safety_margin_seconds:
                    logger.warning(
                        f"Walltime safety: {remaining/60:.0f}min remaining "
                        f"(< {safety_margin_seconds/60:.0f}min margin). "
                        f"Saving checkpoint and stopping."
                    )
                    if self.output_dir:
                        self.save_checkpoint()
                    self.state.termination_reason = (
                        f"Walltime safety stop at iter {self.state.iteration} "
                        f"({remaining/60:.0f}min remaining)"
                    )
                    return self.state

            # Check early termination
            if self._check_early_termination():
                logger.info(f"Early termination: {self.state.termination_reason}")
                break

            # Suggest next config
            config, fidelity = self.suggest()

            # Build training config
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
                seed=self.seed + self.state.iteration,
            )

            # Evaluate
            try:
                result = evaluator.evaluate(train_config)
                self.observe(config, fidelity, result)
            except Exception as e:
                logger.error(f"Evaluation failed for {config}: {e}")
                continue

            # Save checkpoint after EVERY evaluation (critical for 48h walltime)
            if self.output_dir:
                self.save_checkpoint()

        # Final save
        if self.output_dir:
            self.save_checkpoint()
            self._save_final_report()

        logger.info(
            f"Optimization complete: {self.state.iteration} evals | "
            f"Pareto size={self.state.pareto_size} | "
            f"Total energy={self.state.total_energy_kwh:.4f}kWh | "
            f"Total CO₂e={self.state.total_co2e_kg:.4f}kg"
        )

        return self.state

    def save_checkpoint(self):
        """Save full optimization state to disk for resume across PBS jobs."""
        if not self.output_dir:
            return

        # Serialize history (convert numpy arrays to lists)
        history_data = []
        for rec in self.history:
            rec_dict = {
                "iteration": rec.iteration,
                "config": rec.config,
                "config_vec": rec.config_vec.tolist() if isinstance(rec.config_vec, np.ndarray) else rec.config_vec,
                "fidelity": rec.fidelity,
                "is_feasible": rec.is_feasible,
                "acquisition_value": rec.acquisition_value,
                "predicted_objectives": rec.predicted_objectives,
                "prediction_error": rec.prediction_error,
                "timestamp": rec.timestamp,
            }
            # Serialize TrainingResult
            if rec.result is not None:
                rec_dict["result"] = rec.result.to_dict()
            else:
                rec_dict["result"] = None
            history_data.append(rec_dict)

        checkpoint = {
            "state": asdict(self.state),
            "history": history_data,
            "pareto_points": self.pareto_front.points,
            "pareto_configs": self.pareto_front.configs,
            "hypervolume_history": self.pareto_front.hypervolume_history,
            "rng_state": self.rng.get_state()[1].tolist(),
            "settings": {
                "objectives": self.objectives,
                "directions": self.directions,
                "max_evaluations": self.max_evaluations,
                "fidelity_levels": self.fidelity_levels,
                "seed": self.seed,
                "n_initial_random": self.n_initial_random,
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": 2,
        }

        # Atomic write: write to temp file then rename
        path = self.output_dir / "checkpoint.json"
        tmp_path = self.output_dir / "checkpoint.json.tmp"
        with open(tmp_path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)
        tmp_path.rename(path)

        # Save correction models
        self.correction_manager.save(str(self.output_dir / "correction"))

        logger.info(
            f"  Checkpoint saved: iter={self.state.iteration} | "
            f"pareto={self.state.pareto_size} | "
            f"energy={self.state.total_energy_kwh:.4f}kWh"
        )

    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> bool:
        """
        Load optimization state from a checkpoint file.

        Restores: OptimizationState, history, ParetoFront, RNG state,
        and correction models. Allows resuming across PBS job boundaries.

        Args:
            checkpoint_path: Path to checkpoint JSON. If None, uses output_dir/checkpoint.json.

        Returns:
            True if checkpoint was loaded, False if not found.
        """
        if checkpoint_path:
            path = Path(checkpoint_path)
        elif self.output_dir:
            path = self.output_dir / "checkpoint.json"
        else:
            return False

        if not path.exists():
            logger.info("No checkpoint found, starting fresh.")
            return False

        try:
            with open(path) as f:
                checkpoint = json.load(f)

            # Restore state
            state_dict = checkpoint["state"]
            self.state = OptimizationState(**state_dict)

            # Restore history
            from bamf_eco.training import TrainingResult
            self.history = []
            for rec_dict in checkpoint.get("history", []):
                result = None
                if rec_dict.get("result") is not None:
                    result = TrainingResult.from_dict(rec_dict["result"])
                config_vec = np.array(rec_dict["config_vec"]) if rec_dict.get("config_vec") else np.zeros(self.encoder.encoded_dim)
                record = EvaluationRecord(
                    iteration=rec_dict["iteration"],
                    config=rec_dict["config"],
                    config_vec=config_vec,
                    fidelity=rec_dict["fidelity"],
                    result=result,
                    is_feasible=rec_dict.get("is_feasible", True),
                    acquisition_value=rec_dict.get("acquisition_value", 0.0),
                    predicted_objectives=rec_dict.get("predicted_objectives"),
                    prediction_error=rec_dict.get("prediction_error"),
                    timestamp=rec_dict.get("timestamp", ""),
                )
                self.history.append(record)

            # Restore Pareto front
            self.pareto_front.points = checkpoint.get("pareto_points", [])
            self.pareto_front.configs = checkpoint.get("pareto_configs", [])
            self.pareto_front._hypervolumes = checkpoint.get("hypervolume_history", [])

            # Restore RNG state
            if "rng_state" in checkpoint:
                rng_state = list(self.rng.get_state())
                rng_state[1] = np.array(checkpoint["rng_state"], dtype=np.uint32)
                self.rng.set_state(tuple(rng_state))

            # Restore correction models
            correction_dir = self.output_dir / "correction" if self.output_dir else None
            if correction_dir and correction_dir.exists():
                try:
                    self.correction_manager = MultiFidelityCorrectionManager.load(str(correction_dir))
                except Exception as e:
                    logger.warning(f"Could not load correction models: {e}")

            # Refit GP surrogates from loaded history
            if len(self.history) >= 5:
                self._refit_surrogates()

            logger.info(
                f"Checkpoint loaded: iter={self.state.iteration}/{self.max_evaluations} | "
                f"history={len(self.history)} | pareto={self.state.pareto_size} | "
                f"energy={self.state.total_energy_kwh:.4f}kWh | "
                f"from {checkpoint.get('timestamp', 'unknown')}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def _save_final_report(self):
        """Save final optimization report."""
        report = {
            "state": asdict(self.state),
            "settings": {
                "objectives": self.objectives,
                "directions": self.directions,
                "max_evaluations": self.max_evaluations,
                "total_budget_ksu": self.total_budget_ksu,
                "fidelity_levels": self.fidelity_levels,
                "seed": self.seed,
            },
            "results": {
                "pareto_size": len(self.pareto_front.points),
                "pareto_points": self.pareto_front.points,
                "final_hypervolume": self.pareto_front.hypervolume,
                "hypervolume_history": self.pareto_front.hypervolume_history,
                "total_energy_kwh": self.state.total_energy_kwh,
                "total_co2e_kg": self.state.total_co2e_kg,
                "total_water_liters": self.state.total_water_liters,
            },
            "history_summary": [
                {
                    "iter": rec.iteration,
                    "model": rec.config.get("model_name"),
                    "fidelity": rec.fidelity,
                    "feasible": rec.is_feasible,
                    "mAP": rec.result.map50_95 if rec.result else None,
                    "energy": rec.result.energy_kwh if rec.result else None,
                }
                for rec in self.history
            ],
        }

        path = self.output_dir / "final_report.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Final report saved to {path}")
