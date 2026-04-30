"""
BAMF-Eco Configuration Encoding
==================================

Converts the mixed-type search space (categorical model names, ordinal
image sizes, continuous LR, etc.) into a numeric feature vector suitable
for GP surrogate modeling.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from bamf_eco.utils import MODEL_REGISTRY


@dataclass
class SearchSpaceDim:
    """A single dimension of the search space."""
    name: str
    dim_type: str          # "categorical", "ordinal", "continuous"
    choices: Optional[List[Any]] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    log_scale: bool = False
    is_fidelity: bool = False


# Default BAMF-Eco search space
DEFAULT_SEARCH_SPACE: List[SearchSpaceDim] = [
    SearchSpaceDim("model_name", "categorical",
                   choices=list(MODEL_REGISTRY.keys())),
    SearchSpaceDim("image_size", "ordinal",
                   choices=[320, 416, 512, 640]),
    SearchSpaceDim("precision", "categorical",
                   choices=["fp32", "fp16"]),
    SearchSpaceDim("epochs", "ordinal",
                   choices=[5, 10, 20, 50, 100], is_fidelity=True),
    SearchSpaceDim("lr0", "continuous",
                   lower=1e-4, upper=0.1, log_scale=True),
    SearchSpaceDim("batch_size", "ordinal",
                   choices=[8, 16, 32]),
    SearchSpaceDim("optimizer", "categorical",
                   choices=["SGD", "AdamW"]),
    SearchSpaceDim("weight_decay", "continuous",
                   lower=1e-4, upper=0.01, log_scale=True),
    SearchSpaceDim("momentum", "continuous",
                   lower=0.8, upper=0.99),
    SearchSpaceDim("augment_strength", "continuous",
                   lower=0.0, upper=1.0),
]


class ConfigEncoder:
    """
    Encode/decode configurations to/from numeric vectors.

    Categorical → one-hot
    Ordinal → normalized index [0, 1]
    Continuous → normalized to [0, 1] (optionally log-transformed)
    """

    def __init__(self, dims: Optional[List[SearchSpaceDim]] = None):
        self.dims = dims or DEFAULT_SEARCH_SPACE
        self._build_index()

    def _build_index(self):
        """Compute the total encoded dimensionality."""
        self._dim_slices = {}
        idx = 0
        for dim in self.dims:
            if dim.dim_type == "categorical":
                n = len(dim.choices)
                self._dim_slices[dim.name] = (idx, idx + n, "onehot")
                idx += n
            else:
                self._dim_slices[dim.name] = (idx, idx + 1, dim.dim_type)
                idx += 1
        self.encoded_dim = idx

    def encode(self, config: Dict[str, Any]) -> np.ndarray:
        """Encode a config dict into a numeric vector."""
        vec = np.zeros(self.encoded_dim)
        for dim in self.dims:
            start, end, dtype = self._dim_slices[dim.name]
            val = config.get(dim.name)
            if val is None:
                continue

            if dtype == "onehot":
                if val in dim.choices:
                    idx = dim.choices.index(val)
                    vec[start + idx] = 1.0
            elif dim.dim_type == "ordinal":
                idx = dim.choices.index(val) if val in dim.choices else 0
                vec[start] = idx / max(len(dim.choices) - 1, 1)
            elif dim.dim_type == "continuous":
                if dim.log_scale:
                    val = np.log(val)
                    lo = np.log(dim.lower)
                    hi = np.log(dim.upper)
                else:
                    lo = dim.lower
                    hi = dim.upper
                vec[start] = (val - lo) / (hi - lo + 1e-8)

        return vec

    def decode(self, vec: np.ndarray) -> Dict[str, Any]:
        """Decode a numeric vector back to a config dict."""
        config = {}
        for dim in self.dims:
            start, end, dtype = self._dim_slices[dim.name]

            if dtype == "onehot":
                idx = int(np.argmax(vec[start:end]))
                config[dim.name] = dim.choices[idx]
            elif dim.dim_type == "ordinal":
                idx = int(round(vec[start] * (len(dim.choices) - 1)))
                idx = max(0, min(idx, len(dim.choices) - 1))
                config[dim.name] = dim.choices[idx]
            elif dim.dim_type == "continuous":
                if dim.log_scale:
                    lo = np.log(dim.lower)
                    hi = np.log(dim.upper)
                    val = np.exp(vec[start] * (hi - lo) + lo)
                else:
                    val = vec[start] * (dim.upper - dim.lower) + dim.lower
                config[dim.name] = float(val)

        return config

    def random_config(self, rng: Optional[np.random.RandomState] = None) -> Dict[str, Any]:
        """Sample a random configuration."""
        rng = rng or np.random.RandomState()
        config = {}
        for dim in self.dims:
            if dim.dim_type == "categorical":
                config[dim.name] = rng.choice(dim.choices)
            elif dim.dim_type == "ordinal":
                config[dim.name] = rng.choice(dim.choices)
            elif dim.dim_type == "continuous":
                if dim.log_scale:
                    log_val = rng.uniform(np.log(dim.lower), np.log(dim.upper))
                    config[dim.name] = float(np.exp(log_val))
                else:
                    config[dim.name] = float(rng.uniform(dim.lower, dim.upper))
        return config

    def get_fidelity_dim(self) -> Optional[SearchSpaceDim]:
        """Get the fidelity dimension if defined."""
        for dim in self.dims:
            if dim.is_fidelity:
                return dim
        return None
