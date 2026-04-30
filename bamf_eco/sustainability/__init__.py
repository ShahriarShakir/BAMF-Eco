"""
Sustainability Accounting Module
=================================

Converts measured energy (kWh) to:
- CO₂e emissions (kg) using regional grid carbon intensity factors
- Water footprint (liters) using data center cooling factors

Supports:
- Point estimates with uncertainty ranges (min/max)
- Configurable regional factors with provenance tracking
- Parameterized assumptions for sensitivity analysis

References:
    - Patterson et al. (2021): Carbon Emissions and Large Neural Network Training
    - Li et al. (2023): Making AI Less "Thirsty"
    - IEA (2024): CO₂ Emissions per kWh by country
    - Shehabi et al. (2016): US Data Center Energy Report (water usage)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple
from loguru import logger


# ===========================================================================
# Regional Carbon Intensity Factors (kg CO₂e per kWh)
# ===========================================================================
# Source: IEA 2024 edition, various country reports

CARBON_FACTORS: Dict[str, Dict[str, float]] = {
    # Region: {mean, min, max} in kg CO₂e per kWh
    "default": {"mean": 0.79, "min": 0.60, "max": 1.00},
    "australia": {"mean": 0.79, "min": 0.60, "max": 1.00},
    "australia_nsw": {"mean": 0.73, "min": 0.55, "max": 0.90},
    "australia_vic": {"mean": 0.85, "min": 0.65, "max": 1.05},
    "australia_qld": {"mean": 0.80, "min": 0.60, "max": 1.00},
    "usa_avg": {"mean": 0.39, "min": 0.25, "max": 0.55},
    "usa_california": {"mean": 0.21, "min": 0.10, "max": 0.35},
    "usa_virginia": {"mean": 0.33, "min": 0.20, "max": 0.45},
    "usa_texas": {"mean": 0.40, "min": 0.25, "max": 0.55},
    "europe_avg": {"mean": 0.28, "min": 0.15, "max": 0.45},
    "france": {"mean": 0.06, "min": 0.04, "max": 0.10},
    "germany": {"mean": 0.35, "min": 0.20, "max": 0.50},
    "uk": {"mean": 0.23, "min": 0.12, "max": 0.35},
    "china_avg": {"mean": 0.58, "min": 0.40, "max": 0.75},
    "india_avg": {"mean": 0.72, "min": 0.55, "max": 0.90},
    "japan": {"mean": 0.45, "min": 0.30, "max": 0.60},
    "canada_avg": {"mean": 0.12, "min": 0.05, "max": 0.20},
    "nordic": {"mean": 0.03, "min": 0.01, "max": 0.08},
}


# ===========================================================================
# Water Usage Factors (liters per kWh)
# ===========================================================================
# Source: Li et al. (2023), Shehabi et al. (2016)
# Cooling water consumption varies with PUE, climate, cooling method

WATER_FACTORS: Dict[str, Dict[str, float]] = {
    # Cooling type: {mean, min, max} in liters per kWh of IT load
    "evaporative_arid": {"mean": 5.0, "min": 3.5, "max": 7.0},     # Hot/dry climate
    "evaporative_temperate": {"mean": 3.5, "min": 2.0, "max": 5.0}, # Moderate climate
    "air_cooled": {"mean": 1.8, "min": 0.5, "max": 3.0},            # Air cooling only
    "hybrid": {"mean": 2.5, "min": 1.5, "max": 4.0},                # Mixed cooling
    "free_cooling": {"mean": 0.5, "min": 0.1, "max": 1.5},          # Free-air (cold climate)
    "default": {"mean": 3.5, "min": 1.8, "max": 5.0},               # General estimate
}


@dataclass
class SustainabilityEstimate:
    """Sustainability impact estimate with uncertainty."""
    # CO₂ emissions
    co2e_kg: float = 0.0
    co2e_kg_min: float = 0.0
    co2e_kg_max: float = 0.0

    # Water footprint
    water_liters: float = 0.0
    water_liters_min: float = 0.0
    water_liters_max: float = 0.0

    # Provenance
    energy_kwh: float = 0.0
    carbon_region: str = ""
    carbon_factor: float = 0.0
    water_cooling_type: str = ""
    water_factor: float = 0.0
    measurement_type: str = ""  # "measured" or "derived"

    def to_dict(self) -> dict:
        return asdict(self)


class SustainabilityAccountant:
    """
    Converts energy measurements to environmental impact estimates.

    Usage:
        accountant = SustainabilityAccountant(
            carbon_region="default",
            water_cooling_type="evaporative_temperate"
        )
        estimate = accountant.compute(energy_kwh=0.5)
        print(f"CO₂e: {estimate.co2e_kg:.4f} kg")
        print(f"Water: {estimate.water_liters:.2f} liters")
    """

    def __init__(
        self,
        carbon_region: str = "default",
        water_cooling_type: str = "default",
        custom_carbon_factor: Optional[float] = None,
        custom_carbon_min: Optional[float] = None,
        custom_carbon_max: Optional[float] = None,
        custom_water_factor: Optional[float] = None,
        custom_water_min: Optional[float] = None,
        custom_water_max: Optional[float] = None,
    ):
        # Carbon settings
        self.carbon_region = carbon_region
        if custom_carbon_factor is not None:
            self.carbon_factor = custom_carbon_factor
            self.carbon_factor_min = custom_carbon_min or custom_carbon_factor * 0.75
            self.carbon_factor_max = custom_carbon_max or custom_carbon_factor * 1.25
        elif carbon_region in CARBON_FACTORS:
            cf = CARBON_FACTORS[carbon_region]
            self.carbon_factor = cf["mean"]
            self.carbon_factor_min = cf["min"]
            self.carbon_factor_max = cf["max"]
        else:
            logger.warning(f"Unknown region '{carbon_region}', using default region")
            cf = CARBON_FACTORS["default"]
            self.carbon_factor = cf["mean"]
            self.carbon_factor_min = cf["min"]
            self.carbon_factor_max = cf["max"]

        # Water settings
        self.water_cooling_type = water_cooling_type
        if custom_water_factor is not None:
            self.water_factor = custom_water_factor
            self.water_factor_min = custom_water_min or custom_water_factor * 0.5
            self.water_factor_max = custom_water_max or custom_water_factor * 1.5
        elif water_cooling_type in WATER_FACTORS:
            wf = WATER_FACTORS[water_cooling_type]
            self.water_factor = wf["mean"]
            self.water_factor_min = wf["min"]
            self.water_factor_max = wf["max"]
        else:
            logger.warning(f"Unknown cooling type '{water_cooling_type}', using default")
            wf = WATER_FACTORS["default"]
            self.water_factor = wf["mean"]
            self.water_factor_min = wf["min"]
            self.water_factor_max = wf["max"]

        logger.info(
            f"SustainabilityAccountant: carbon={self.carbon_factor:.2f} "
            f"[{self.carbon_factor_min:.2f}-{self.carbon_factor_max:.2f}] "
            f"kg CO₂e/kWh ({carbon_region}), "
            f"water={self.water_factor:.1f} "
            f"[{self.water_factor_min:.1f}-{self.water_factor_max:.1f}] "
            f"L/kWh ({water_cooling_type})"
        )

    def compute(
        self,
        energy_kwh: float,
        measurement_type: str = "measured",
    ) -> SustainabilityEstimate:
        """
        Compute sustainability estimate from energy consumption.

        Args:
            energy_kwh: Measured or estimated energy in kWh
            measurement_type: "measured" (from NVML) or "derived" (estimated)

        Returns:
            SustainabilityEstimate with point estimate and uncertainty ranges
        """
        return SustainabilityEstimate(
            co2e_kg=energy_kwh * self.carbon_factor,
            co2e_kg_min=energy_kwh * self.carbon_factor_min,
            co2e_kg_max=energy_kwh * self.carbon_factor_max,
            water_liters=energy_kwh * self.water_factor,
            water_liters_min=energy_kwh * self.water_factor_min,
            water_liters_max=energy_kwh * self.water_factor_max,
            energy_kwh=energy_kwh,
            carbon_region=self.carbon_region,
            carbon_factor=self.carbon_factor,
            water_cooling_type=self.water_cooling_type,
            water_factor=self.water_factor,
            measurement_type=measurement_type,
        )

    def assumptions_snapshot(self) -> dict:
        """Return current assumptions for reproducibility."""
        return {
            "carbon_region": self.carbon_region,
            "carbon_factor_mean": self.carbon_factor,
            "carbon_factor_min": self.carbon_factor_min,
            "carbon_factor_max": self.carbon_factor_max,
            "water_cooling_type": self.water_cooling_type,
            "water_factor_mean": self.water_factor,
            "water_factor_min": self.water_factor_min,
            "water_factor_max": self.water_factor_max,
            "carbon_factor_source": "IEA 2024 / Patterson et al. 2021",
            "water_factor_source": "Li et al. 2023 / Shehabi et al. 2016",
        }


def compute_sustainability(
    energy_kwh: float,
    carbon_region: str = "default",
    water_cooling_type: str = "default",
) -> SustainabilityEstimate:
    """Convenience function for quick sustainability computation."""
    accountant = SustainabilityAccountant(
        carbon_region=carbon_region,
        water_cooling_type=water_cooling_type,
    )
    return accountant.compute(energy_kwh)
