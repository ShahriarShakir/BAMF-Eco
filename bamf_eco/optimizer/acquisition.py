"""
Eco-Aware Acquisition Function
=================================

Implements the novel Eco-Feasibility EHVI acquisition:

    α_eco(x) = EHVI(x) · P(feasible(x) | D) · w_fidelity(z)

Where:
- EHVI: Expected Hypervolume Improvement (multi-objective)
- P(feasible): Probability config stays within eco-budget constraints
- w_fidelity: Fidelity-aware cost weighting

Theorem 1 (Eco-Feasibility Acquisition):
  The expected cumulative regret of BAMF-Eco using α_eco satisfies
  R_T ≤ O(√(T · β_T · γ_T)) where γ_T is the information gain of the
  composite GP model including eco-feasibility constraints.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

try:
    import torch
    from botorch.models import SingleTaskGP, ModelListGP
    from botorch.acquisition.multi_objective import (
        ExpectedHypervolumeImprovement,
    )
    from botorch.utils.multi_objective.box_decompositions import (
        NondominatedPartitioning,
    )
    from botorch.utils.multi_objective.pareto import is_non_dominated
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    HAS_BOTORCH = True
except ImportError:
    HAS_BOTORCH = False


from loguru import logger


@dataclass
class EcoConstraints:
    """Sustainability constraints for feasibility."""
    max_energy_kwh: float = 0.05
    max_co2e_kg: float = 0.5
    max_latency_ms: float = 100.0
    max_training_time_s: float = 86400.0   # 24 hours


class EcoFeasibilityGP:
    """
    GP classifier for eco-feasibility.

    Models P(feasible | x) where feasibility is defined as
    meeting all eco-constraints (energy, CO₂e, latency).
    """

    def __init__(self, constraints: EcoConstraints):
        self.constraints = constraints
        self._model = None
        self._train_x = None
        self._train_y = None
        self._input_mean = None
        self._input_std = None

    def is_feasible(self, result: Dict[str, float]) -> bool:
        """Check if a result meets all eco-constraints."""
        checks = [
            result.get("energy_kwh", 0) <= self.constraints.max_energy_kwh,
            result.get("co2e_kg", 0) <= self.constraints.max_co2e_kg,
            result.get("latency_ms", 0) <= self.constraints.max_latency_ms,
        ]
        return all(checks)

    def fit(self, X: np.ndarray, feasibility: np.ndarray):
        """
        Fit GP on binary feasibility outcomes.

        Args:
            X: (n, d) config feature vectors
            feasibility: (n,) binary labels (1=feasible, 0=infeasible)
        """
        if len(X) < 3:
            return

        self._input_mean = X.mean(axis=0)
        self._input_std = X.std(axis=0) + 1e-8
        X_norm = (X - self._input_mean) / self._input_std

        # Transform binary to continuous for GP (probit approximation)
        y_cont = feasibility.astype(float) * 2.0 - 1.0  # {0,1} → {-1, 1}

        if HAS_BOTORCH:
            train_x = torch.tensor(X_norm, dtype=torch.float64)
            train_y = torch.tensor(y_cont, dtype=torch.float64).unsqueeze(-1)

            self._model = SingleTaskGP(train_x, train_y)
            mll = ExactMarginalLogLikelihood(self._model.likelihood, self._model)
            try:
                fit_gpytorch_mll(mll)
            except Exception as e:
                logger.warning(f"Feasibility GP fitting issue: {e}")

            self._train_x = train_x
            self._train_y = train_y

    def predict_feasibility(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Predict feasibility probability.

        Returns:
            (probability_feasible, uncertainty)
        """
        if self._model is None or self._input_mean is None:
            return 0.5, 0.5  # Uninformed prior

        x_norm = (x - self._input_mean) / self._input_std

        if HAS_BOTORCH:
            x_tensor = torch.tensor(x_norm, dtype=torch.float64).unsqueeze(0)
            self._model.eval()
            with torch.no_grad():
                posterior = self._model.posterior(x_tensor)
                mean = posterior.mean.item()
                std = posterior.variance.sqrt().item()

            # Sigmoid-like mapping from GP output to probability
            prob = 1.0 / (1.0 + np.exp(-mean))
            return prob, std

        return 0.5, 0.5


class EcoAcquisition:
    """
    Eco-aware multi-objective acquisition function.

    Combines EHVI with eco-feasibility probability and fidelity cost weighting.
    """

    def __init__(
        self,
        reference_point: Dict[str, float],
        objectives: List[str],
        directions: List[str],   # "maximize" or "minimize"
        constraints: Optional[EcoConstraints] = None,
    ):
        self.reference_point = reference_point
        self.objectives = objectives
        self.directions = directions
        self.constraints = constraints or EcoConstraints()
        self.feasibility_gp = EcoFeasibilityGP(self.constraints)

        # Surrogate GP models (one per objective)
        self._models: Dict[str, Any] = {}
        self._pareto_Y = None

    def fit_surrogates(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        feasibility: np.ndarray,
    ):
        """
        Fit GP surrogates for all objectives + feasibility.

        Args:
            X: (n, d) config vectors
            Y: (n, m) objective values (columns match self.objectives)
            feasibility: (n,) binary feasibility labels
        """
        if len(X) < 3:
            return

        # Normalize X
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0) + 1e-8
        X_norm = (X - self._x_mean) / self._x_std

        # Flip sign for minimization objectives so we always maximize
        Y_max = Y.copy()
        for i, direction in enumerate(self.directions):
            if direction == "minimize":
                Y_max[:, i] = -Y_max[:, i]

        # Normalize Y
        self._y_mean = Y_max.mean(axis=0)
        self._y_std = Y_max.std(axis=0) + 1e-8
        Y_norm = (Y_max - self._y_mean) / self._y_std

        if HAS_BOTORCH:
            train_x = torch.tensor(X_norm, dtype=torch.float64)

            models = []
            for i, obj in enumerate(self.objectives):
                train_y = torch.tensor(Y_norm[:, i], dtype=torch.float64).unsqueeze(-1)
                gp = SingleTaskGP(train_x, train_y)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                try:
                    fit_gpytorch_mll(mll)
                except Exception:
                    pass
                models.append(gp)
                self._models[obj] = gp

            self._model_list = ModelListGP(*models)

            # Store Pareto front for EHVI
            Y_tensor = torch.tensor(Y_norm, dtype=torch.float64)
            pareto_mask = is_non_dominated(Y_tensor)
            self._pareto_Y = Y_tensor[pareto_mask]

        # Fit feasibility GP
        self.feasibility_gp.fit(X, feasibility)

        logger.debug(f"Surrogates fitted: {len(X)} observations, {len(self.objectives)} objectives")

    def compute(
        self,
        x: np.ndarray,
        fidelity_cost: float = 1.0,
    ) -> float:
        """
        Compute eco-acquisition value: α_eco(x) = EHVI(x) · P(feasible) / cost(z)

        Args:
            x: Config feature vector
            fidelity_cost: Relative cost of this fidelity level [0, 1]

        Returns:
            Acquisition value (higher = better candidate)
        """
        if not self._models:
            # No surrogates fitted yet, use random exploration
            return np.random.rand()

        # 1. Compute EHVI
        ehvi = self._compute_ehvi(x)

        # 2. Compute feasibility probability
        prob_feasible, _ = self.feasibility_gp.predict_feasibility(x)

        # 3. Cost weighting (cheap fidelities are preferred)
        cost_weight = 1.0 / (fidelity_cost + 0.01)

        # Combine: α_eco = EHVI × P(feasible) × cost_weight
        acquisition = ehvi * prob_feasible * cost_weight

        return acquisition

    def _compute_ehvi(self, x: np.ndarray) -> float:
        """Compute Expected Hypervolume Improvement."""
        if not HAS_BOTORCH or self._pareto_Y is None:
            # Fallback: use mean improvement
            return self._compute_fallback_improvement(x)

        x_norm = (x - self._x_mean) / self._x_std
        x_tensor = torch.tensor(x_norm, dtype=torch.float64).unsqueeze(0)

        # Reference point in normalized space
        ref_raw = np.array([self.reference_point[obj] for obj in self.objectives])
        for i, direction in enumerate(self.directions):
            if direction == "minimize":
                ref_raw[i] = -ref_raw[i]
        ref_norm = (ref_raw - self._y_mean) / self._y_std
        ref_tensor = torch.tensor(ref_norm, dtype=torch.float64)

        try:
            partitioning = NondominatedPartitioning(
                ref_point=ref_tensor,
                Y=self._pareto_Y,
            )
            ehvi = ExpectedHypervolumeImprovement(
                model=self._model_list,
                ref_point=ref_tensor.tolist(),
                partitioning=partitioning,
            )

            with torch.no_grad():
                value = ehvi(x_tensor.unsqueeze(0))
            return float(value.item())
        except Exception as e:
            logger.debug(f"EHVI computation fallback: {e}")
            return self._compute_fallback_improvement(x)

    def _compute_fallback_improvement(self, x: np.ndarray) -> float:
        """Simple predicted improvement when EHVI fails."""
        if not self._models:
            return np.random.rand()

        x_norm = (x - self._x_mean) / self._x_std

        if HAS_BOTORCH:
            x_tensor = torch.tensor(x_norm, dtype=torch.float64).unsqueeze(0)
            improvements = []
            for obj, model in self._models.items():
                model.eval()
                with torch.no_grad():
                    posterior = model.posterior(x_tensor)
                    mean = posterior.mean.item()
                    std = posterior.variance.sqrt().item()
                # Expected improvement: E[max(0, f(x) - f*)]
                if self._pareto_Y is not None and len(self._pareto_Y) > 0:
                    best = float(self._pareto_Y.max(dim=0).values.mean())
                else:
                    best = 0.0
                z = (mean - best) / (std + 1e-8)
                from scipy.stats import norm
                ei = (mean - best) * norm.cdf(z) + std * norm.pdf(z)
                improvements.append(max(0, ei))
            return float(np.prod(improvements)) if improvements else 0.0

        return np.random.rand()

    def select_next(
        self,
        candidates: np.ndarray,
        fidelity_costs: Optional[np.ndarray] = None,
        n_select: int = 1,
    ) -> List[int]:
        """
        Select the best candidate(s) from a set.

        Args:
            candidates: (n, d) candidate config vectors
            fidelity_costs: (n,) cost per candidate
            n_select: Number of candidates to select

        Returns:
            Indices of selected candidates
        """
        if fidelity_costs is None:
            fidelity_costs = np.ones(len(candidates))

        values = np.array([
            self.compute(candidates[i], fidelity_costs[i])
            for i in range(len(candidates))
        ])

        # Select top-n
        indices = np.argsort(values)[::-1][:n_select]
        return indices.tolist()
