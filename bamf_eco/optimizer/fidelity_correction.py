"""
Multi-Fidelity Correction Model
==================================

Learns a mapping from low-fidelity metrics to high-fidelity metrics,
enabling cheap evaluations (few epochs) to predict full-training outcomes.

Two approaches implemented:
1. **GP Correction** (primary): Uses Gaussian Process regression to model
   the residual f_high - f_low as a function of config + f_low.
   Provides calibrated uncertainty for acquisition function.
2. **Linear Correction** (baseline): Simple linear scaling f_high ≈ α·f_low + β.

Theorem 2 (Multi-Fidelity Correction Convergence):
  Under Lipschitz continuity of the true correction function, the GP
  posterior mean converges to the true correction at rate O(n^{-1/(d+2)})
  where n is the number of paired observations.

Usage:
    corrector = FidelityCorrectionModel(method="gp")
    corrector.fit(low_fidelity_metrics, high_fidelity_metrics, configs)
    predicted_high, uncertainty = corrector.predict(new_low_metric, new_config)
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict

import numpy as np
from loguru import logger

try:
    import torch
    import gpytorch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    HAS_BOTORCH = True
except ImportError:
    HAS_BOTORCH = False
    logger.warning("BoTorch not available — GP correction will use sklearn fallback")


@dataclass
class CorrectionObservation:
    """Paired low/high fidelity observation for learning correction."""
    config_vec: np.ndarray            # Encoded configuration vector
    low_fidelity: float                # Low-fidelity fidelity level (e.g., 0.1)
    high_fidelity: float               # High-fidelity fidelity level (e.g., 1.0)
    low_metric: float                  # Metric at low fidelity
    high_metric: float                 # Metric at high fidelity (ground truth)
    metric_name: str = "mAP"           # Which metric this is for
    model_name: str = ""               # Which model was used


class FidelityCorrectionModel:
    """
    Learns correction from low-fidelity to high-fidelity metrics.

    The correction models: f_high = f_low + δ(config, f_low, z_low)
    where δ is the residual learned by a GP or linear model.
    """

    def __init__(
        self,
        method: str = "gp",
        n_features: int = 10,     # Config vector dimensionality
        min_observations: int = 5,
    ):
        """
        Args:
            method: "gp" for GP correction, "linear" for linear scaling
            n_features: Dimensionality of config feature vector
            min_observations: Minimum paired observations before fitting
        """
        self.method = method
        self.n_features = n_features
        self.min_observations = min_observations

        self.observations: List[CorrectionObservation] = []
        self.is_fitted = False

        # GP model components
        self._gp_model = None
        self._gp_likelihood = None
        self._train_x = None
        self._train_y = None
        self._input_mean = None
        self._input_std = None
        self._output_mean = None
        self._output_std = None

        # Linear model components
        self._linear_alpha = 1.0
        self._linear_beta = 0.0
        self._linear_r2 = 0.0

    def add_observation(self, obs: CorrectionObservation):
        """Add a paired low/high fidelity observation."""
        self.observations.append(obs)
        logger.debug(
            f"Correction obs added: {obs.model_name} | "
            f"low({obs.low_fidelity:.2f})={obs.low_metric:.4f} → "
            f"high({obs.high_fidelity:.2f})={obs.high_metric:.4f}"
        )

    @property
    def n_observations(self) -> int:
        return len(self.observations)

    @property
    def can_fit(self) -> bool:
        return self.n_observations >= self.min_observations

    def _build_feature_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build input features X and target residuals y.

        X = [config_vec, low_fidelity_level, low_metric]
        y = high_metric - low_metric (residual)
        """
        n = len(self.observations)
        d = self.n_features + 2  # config_vec + fidelity_level + low_metric

        X = np.zeros((n, d))
        y = np.zeros(n)

        for i, obs in enumerate(self.observations):
            config = obs.config_vec
            if len(config) < self.n_features:
                # Pad with zeros
                config = np.pad(config, (0, self.n_features - len(config)))
            elif len(config) > self.n_features:
                config = config[:self.n_features]

            X[i, :self.n_features] = config
            X[i, self.n_features] = obs.low_fidelity
            X[i, self.n_features + 1] = obs.low_metric
            y[i] = obs.high_metric - obs.low_metric

        return X, y

    def fit(self) -> float:
        """
        Fit the correction model on collected observations.

        Returns:
            fit_quality: R² or negative log likelihood
        """
        if not self.can_fit:
            logger.warning(
                f"Not enough observations to fit ({self.n_observations}/{self.min_observations})"
            )
            return 0.0

        X, y = self._build_feature_matrix()

        if self.method == "gp":
            return self._fit_gp(X, y)
        elif self.method == "linear":
            return self._fit_linear(X, y)
        else:
            raise ValueError(f"Unknown correction method: {self.method}")

    def _fit_gp(self, X: np.ndarray, y: np.ndarray) -> float:
        """Fit GP correction model using BoTorch."""
        # Normalize inputs
        self._input_mean = X.mean(axis=0)
        self._input_std = X.std(axis=0) + 1e-8
        X_norm = (X - self._input_mean) / self._input_std

        self._output_mean = y.mean()
        self._output_std = y.std() + 1e-8
        y_norm = (y - self._output_mean) / self._output_std

        if HAS_BOTORCH:
            # Use BoTorch SingleTaskGP
            train_x = torch.tensor(X_norm, dtype=torch.float64)
            train_y = torch.tensor(y_norm, dtype=torch.float64).unsqueeze(-1)

            self._gp_model = SingleTaskGP(train_x, train_y)
            mll = ExactMarginalLogLikelihood(
                self._gp_model.likelihood, self._gp_model
            )

            try:
                fit_gpytorch_mll(mll)
            except Exception as e:
                logger.warning(f"GP fitting encountered issue: {e}")

            self._train_x = train_x
            self._train_y = train_y
            self.is_fitted = True

            # Compute in-sample R²
            self._gp_model.eval()
            with torch.no_grad():
                posterior = self._gp_model.posterior(train_x)
                pred = posterior.mean.squeeze().numpy()

            ss_res = np.sum((y_norm - pred) ** 2)
            ss_tot = np.sum((y_norm - y_norm.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)

            logger.info(f"GP correction fitted: R²={r2:.4f}, n={len(y)}")
            return r2

        else:
            # Fallback: sklearn GP
            return self._fit_sklearn_gp(X_norm, y_norm)

    def _fit_sklearn_gp(self, X_norm: np.ndarray, y_norm: np.ndarray) -> float:
        """Fallback GP using sklearn."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel

        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=0.01)
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=False,
            random_state=42,
        )
        gpr.fit(X_norm, y_norm)

        self._sklearn_gp = gpr
        self.is_fitted = True

        r2 = gpr.score(X_norm, y_norm)
        logger.info(f"sklearn GP correction fitted: R²={r2:.4f}, n={len(y_norm)}")
        return r2

    def _fit_linear(self, X: np.ndarray, y: np.ndarray) -> float:
        """Fit simple linear correction: high = alpha * low + beta."""
        # Only use the low_metric feature
        low_metrics = X[:, -1]  # last column is low_metric
        high_metrics = low_metrics + y

        # Least squares: high = alpha * low + beta
        A = np.vstack([low_metrics, np.ones(len(low_metrics))]).T
        result = np.linalg.lstsq(A, high_metrics, rcond=None)
        self._linear_alpha, self._linear_beta = result[0]

        # R²
        pred = self._linear_alpha * low_metrics + self._linear_beta
        ss_res = np.sum((high_metrics - pred) ** 2)
        ss_tot = np.sum((high_metrics - high_metrics.mean()) ** 2)
        self._linear_r2 = 1 - ss_res / (ss_tot + 1e-8)

        self.is_fitted = True
        logger.info(
            f"Linear correction fitted: α={self._linear_alpha:.4f}, "
            f"β={self._linear_beta:.4f}, R²={self._linear_r2:.4f}"
        )
        return self._linear_r2

    def predict(
        self,
        config_vec: np.ndarray,
        low_fidelity: float,
        low_metric: float,
    ) -> Tuple[float, float]:
        """
        Predict high-fidelity metric from low-fidelity observation.

        Args:
            config_vec: Configuration feature vector
            low_fidelity: Fidelity level of the low-fidelity evaluation
            low_metric: Metric value at low fidelity

        Returns:
            (predicted_high_metric, uncertainty_std)
        """
        if not self.is_fitted:
            # Before fitting, just return the low-fidelity metric with high uncertainty
            return low_metric, 0.5

        if self.method == "gp":
            return self._predict_gp(config_vec, low_fidelity, low_metric)
        elif self.method == "linear":
            return self._predict_linear(low_metric)
        else:
            return low_metric, 0.5

    def _predict_gp(
        self,
        config_vec: np.ndarray,
        low_fidelity: float,
        low_metric: float,
    ) -> Tuple[float, float]:
        """Predict using GP model."""
        # Build feature vector
        if len(config_vec) < self.n_features:
            config_vec = np.pad(config_vec, (0, self.n_features - len(config_vec)))
        elif len(config_vec) > self.n_features:
            config_vec = config_vec[:self.n_features]

        x = np.concatenate([config_vec, [low_fidelity, low_metric]])
        x_norm = (x - self._input_mean) / self._input_std

        if HAS_BOTORCH and self._gp_model is not None:
            x_tensor = torch.tensor(x_norm, dtype=torch.float64).unsqueeze(0)
            self._gp_model.eval()

            with torch.no_grad():
                posterior = self._gp_model.posterior(x_tensor)
                residual_norm = posterior.mean.item()
                std_norm = posterior.variance.sqrt().item()

            # Denormalize
            residual = residual_norm * self._output_std + self._output_mean
            std = std_norm * self._output_std

            predicted_high = low_metric + residual
            return predicted_high, std

        elif hasattr(self, "_sklearn_gp"):
            residual_norm, std_norm = self._sklearn_gp.predict(
                x_norm.reshape(1, -1), return_std=True
            )
            residual = residual_norm[0] * self._output_std + self._output_mean
            std = std_norm[0] * self._output_std
            return low_metric + residual, std

        return low_metric, 0.5

    def _predict_linear(self, low_metric: float) -> Tuple[float, float]:
        """Predict using linear model."""
        predicted_high = self._linear_alpha * low_metric + self._linear_beta
        # Rough uncertainty from R²
        std = (1 - self._linear_r2) * 0.1
        return predicted_high, std

    def get_correction_stats(self) -> Dict[str, Any]:
        """Get statistics about the correction model."""
        if not self.observations:
            return {"n_observations": 0, "fitted": False}

        residuals = [obs.high_metric - obs.low_metric for obs in self.observations]
        return {
            "n_observations": len(self.observations),
            "fitted": self.is_fitted,
            "method": self.method,
            "residual_mean": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals)),
            "residual_min": float(np.min(residuals)),
            "residual_max": float(np.max(residuals)),
        }

    def save(self, path: str):
        """Save correction model to disk."""
        state = {
            "method": self.method,
            "n_features": self.n_features,
            "min_observations": self.min_observations,
            "is_fitted": self.is_fitted,
            "n_observations": len(self.observations),
            "observations": [asdict(obs) for obs in self.observations],
        }

        if self.method == "linear":
            state["linear_alpha"] = self._linear_alpha
            state["linear_beta"] = self._linear_beta
            state["linear_r2"] = self._linear_r2

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save JSON metadata
        with open(path + ".json", "w") as f:
            json.dump(state, f, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))

        # Save GP model state if applicable
        if self.is_fitted and self.method == "gp":
            gp_state = {
                "input_mean": self._input_mean,
                "input_std": self._input_std,
                "output_mean": self._output_mean,
                "output_std": self._output_std,
            }
            if HAS_BOTORCH and self._gp_model is not None:
                gp_state["gp_state_dict"] = self._gp_model.state_dict()
                gp_state["train_x"] = self._train_x
                gp_state["train_y"] = self._train_y
            with open(path + ".pkl", "wb") as f:
                pickle.dump(gp_state, f)

        logger.info(f"Correction model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "FidelityCorrectionModel":
        """Load correction model from disk."""
        with open(path + ".json") as f:
            state = json.load(f)

        model = cls(
            method=state["method"],
            n_features=state["n_features"],
            min_observations=state["min_observations"],
        )
        model.is_fitted = state["is_fitted"]

        # Restore observations
        for obs_dict in state.get("observations", []):
            obs_dict["config_vec"] = np.array(obs_dict["config_vec"])
            model.observations.append(CorrectionObservation(**obs_dict))

        if state["method"] == "linear":
            model._linear_alpha = state.get("linear_alpha", 1.0)
            model._linear_beta = state.get("linear_beta", 0.0)
            model._linear_r2 = state.get("linear_r2", 0.0)

        # Load GP state
        if model.is_fitted and model.method == "gp":
            pkl_path = path + ".pkl"
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    gp_state = pickle.load(f)
                model._input_mean = gp_state["input_mean"]
                model._input_std = gp_state["input_std"]
                model._output_mean = gp_state["output_mean"]
                model._output_std = gp_state["output_std"]

                if HAS_BOTORCH and "gp_state_dict" in gp_state:
                    train_x = gp_state["train_x"]
                    train_y = gp_state["train_y"]
                    model._train_x = train_x
                    model._train_y = train_y
                    model._gp_model = SingleTaskGP(train_x, train_y)
                    model._gp_model.load_state_dict(gp_state["gp_state_dict"])

        logger.info(f"Correction model loaded from {path}")
        return model


class MultiFidelityCorrectionManager:
    """
    Manages correction models for multiple objectives.

    Each objective (mAP, energy, latency) gets its own correction model
    since the low→high fidelity relationship differs per metric.
    """

    def __init__(
        self,
        objectives: Optional[List[str]] = None,
        method: str = "gp",
        n_features: int = 10,
    ):
        self.objectives = objectives or ["mAP", "energy_kwh", "latency_ms"]
        self.models: Dict[str, FidelityCorrectionModel] = {}
        for obj in self.objectives:
            self.models[obj] = FidelityCorrectionModel(
                method=method,
                n_features=n_features,
            )

    def add_paired_observation(
        self,
        config_vec: np.ndarray,
        low_fidelity: float,
        high_fidelity: float,
        low_metrics: Dict[str, float],
        high_metrics: Dict[str, float],
        model_name: str = "",
    ):
        """Add paired observation for all objectives."""
        for obj in self.objectives:
            if obj in low_metrics and obj in high_metrics:
                obs = CorrectionObservation(
                    config_vec=config_vec.copy(),
                    low_fidelity=low_fidelity,
                    high_fidelity=high_fidelity,
                    low_metric=low_metrics[obj],
                    high_metric=high_metrics[obj],
                    metric_name=obj,
                    model_name=model_name,
                )
                self.models[obj].add_observation(obs)

    def fit_all(self) -> Dict[str, float]:
        """Fit all correction models. Returns per-objective R²."""
        r2_scores = {}
        for obj, model in self.models.items():
            if model.can_fit:
                r2 = model.fit()
                r2_scores[obj] = r2
            else:
                r2_scores[obj] = 0.0
        return r2_scores

    def predict_all(
        self,
        config_vec: np.ndarray,
        low_fidelity: float,
        low_metrics: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Predict high-fidelity metrics for all objectives.

        Returns:
            means: Dict of predicted high-fidelity values
            stds: Dict of prediction uncertainties
        """
        means = {}
        stds = {}
        for obj in self.objectives:
            if obj in low_metrics:
                mean, std = self.models[obj].predict(
                    config_vec, low_fidelity, low_metrics[obj]
                )
                means[obj] = mean
                stds[obj] = std
        return means, stds

    def save(self, dir_path: str):
        """Save all correction models."""
        os.makedirs(dir_path, exist_ok=True)
        for obj, model in self.models.items():
            model.save(os.path.join(dir_path, f"correction_{obj}"))

    @classmethod
    def load(cls, dir_path: str) -> "MultiFidelityCorrectionManager":
        """Load all correction models."""
        manager = cls.__new__(cls)
        manager.models = {}
        manager.objectives = []

        for f in sorted(os.listdir(dir_path)):
            if f.startswith("correction_") and f.endswith(".json"):
                obj = f.replace("correction_", "").replace(".json", "")
                manager.objectives.append(obj)
                model_path = os.path.join(dir_path, f"correction_{obj}")
                manager.models[obj] = FidelityCorrectionModel.load(model_path)

        return manager
