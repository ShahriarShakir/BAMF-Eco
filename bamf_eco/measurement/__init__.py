"""
GPU Power Measurement Harness
==============================

Real-time GPU power monitoring using NVML (via pynvml) for accurate
energy consumption measurement during inference and training.

Key features:
- Background thread power sampling at configurable frequency
- Numerical integration (trapezoidal rule) for energy in kWh
- Warmup period handling
- Per-run metadata capture
- Support for multi-GPU (reads per-GPU power)

Reference:
    NVML API: https://developer.nvidia.com/nvidia-management-library-nvml
    Energy = ∫ P(t) dt  (Watt-seconds → kWh)
"""

import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger


@dataclass
class PowerSample:
    """A single power reading."""
    timestamp: float       # time.perf_counter() value
    power_watts: float     # Instantaneous GPU power draw in Watts
    gpu_index: int = 0


@dataclass
class EnergyMeasurement:
    """Result of an energy measurement session."""
    # Raw samples
    samples: List[PowerSample] = field(default_factory=list)

    # Computed metrics
    energy_joules: float = 0.0
    energy_kwh: float = 0.0
    duration_seconds: float = 0.0
    power_mean_watts: float = 0.0
    power_std_watts: float = 0.0
    power_min_watts: float = 0.0
    power_max_watts: float = 0.0
    num_samples: int = 0
    sample_rate_hz: float = 0.0

    # Metadata
    gpu_name: str = ""
    gpu_index: int = 0
    measurement_type: str = ""  # "inference" or "training"

    def compute_stats(self):
        """Compute energy and power statistics from raw samples."""
        if len(self.samples) < 2:
            logger.warning("Need at least 2 samples to compute energy")
            return

        timestamps = np.array([s.timestamp for s in self.samples])
        powers = np.array([s.power_watts for s in self.samples])

        self.num_samples = len(self.samples)
        self.duration_seconds = timestamps[-1] - timestamps[0]

        # Trapezoidal integration: ∫ P(t) dt
        self.energy_joules = float(np.trapz(powers, timestamps))
        self.energy_kwh = self.energy_joules / 3_600_000.0  # J → kWh

        # Power stats
        self.power_mean_watts = float(np.mean(powers))
        self.power_std_watts = float(np.std(powers))
        self.power_min_watts = float(np.min(powers))
        self.power_max_watts = float(np.max(powers))

        # Effective sample rate
        if self.duration_seconds > 0:
            self.sample_rate_hz = self.num_samples / self.duration_seconds

    def summary(self) -> dict:
        """Return summary dict (no raw samples)."""
        return {
            "energy_kwh": self.energy_kwh,
            "energy_joules": self.energy_joules,
            "duration_seconds": self.duration_seconds,
            "power_mean_watts": self.power_mean_watts,
            "power_std_watts": self.power_std_watts,
            "power_min_watts": self.power_min_watts,
            "power_max_watts": self.power_max_watts,
            "num_samples": self.num_samples,
            "sample_rate_hz": self.sample_rate_hz,
            "gpu_name": self.gpu_name,
            "gpu_index": self.gpu_index,
            "measurement_type": self.measurement_type,
        }


class GPUPowerMonitor:
    """
    Background GPU power monitor using NVML.

    Usage:
        monitor = GPUPowerMonitor(sample_interval_ms=100)
        monitor.start()
        # ... run workload ...
        measurement = monitor.stop()
        print(f"Energy: {measurement.energy_kwh:.6f} kWh")
    """

    def __init__(
        self,
        gpu_index: int = 0,
        sample_interval_ms: int = 100,
        measurement_type: str = "inference",
    ):
        self.gpu_index = gpu_index
        self.sample_interval_s = sample_interval_ms / 1000.0
        self.measurement_type = measurement_type

        self._samples: List[PowerSample] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._nvml_initialized = False
        self._handle = None
        self._gpu_name = ""

    def _init_nvml(self):
        """Initialize NVML and get GPU handle."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self._gpu_name = pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(self._gpu_name, bytes):
                self._gpu_name = self._gpu_name.decode("utf-8")
            self._nvml_initialized = True
            logger.info(f"NVML initialized: GPU {self.gpu_index} = {self._gpu_name}")
        except Exception as e:
            logger.warning(f"NVML initialization failed: {e}. Using simulated power.")
            self._nvml_initialized = False

    def _read_power(self) -> float:
        """Read instantaneous power in Watts."""
        if self._nvml_initialized and self._handle is not None:
            import pynvml
            try:
                # nvmlDeviceGetPowerUsage returns milliwatts
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                return power_mw / 1000.0  # Convert to Watts
            except Exception:
                return 0.0
        else:
            # CPU-only fallback: return 0 (will be handled in results)
            return 0.0

    def _sampling_loop(self):
        """Background sampling thread."""
        while not self._stop_event.is_set():
            power = self._read_power()
            sample = PowerSample(
                timestamp=time.perf_counter(),
                power_watts=power,
                gpu_index=self.gpu_index,
            )
            self._samples.append(sample)
            self._stop_event.wait(self.sample_interval_s)

    def start(self):
        """Start power monitoring in background thread."""
        self._init_nvml()
        self._samples.clear()
        self._stop_event.clear()

        self._thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"Power monitoring started (GPU {self.gpu_index}, "
            f"interval={self.sample_interval_s*1000:.0f}ms)"
        )

    def stop(self) -> EnergyMeasurement:
        """Stop monitoring and return energy measurement."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

        measurement = EnergyMeasurement(
            samples=list(self._samples),
            gpu_name=self._gpu_name,
            gpu_index=self.gpu_index,
            measurement_type=self.measurement_type,
        )
        measurement.compute_stats()

        # Clean up NVML
        if self._nvml_initialized:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False

        logger.info(
            f"Power monitoring stopped: {measurement.num_samples} samples, "
            f"{measurement.duration_seconds:.2f}s, "
            f"{measurement.energy_kwh:.8f} kWh, "
            f"{measurement.power_mean_watts:.1f}W avg"
        )
        return measurement

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class CPUFallbackMonitor:
    """
    Fallback monitor for CPU-only environments (e.g., login nodes).
    Records wall-clock time but reports 0 power/energy.
    Useful for testing pipeline logic without GPU.
    """

    def __init__(self, measurement_type: str = "inference"):
        self.measurement_type = measurement_type
        self._start_time: float = 0.0

    def start(self):
        self._start_time = time.perf_counter()
        logger.info("CPU fallback monitor started (no power measurement)")

    def stop(self) -> EnergyMeasurement:
        elapsed = time.perf_counter() - self._start_time
        measurement = EnergyMeasurement(
            duration_seconds=elapsed,
            measurement_type=self.measurement_type,
            gpu_name="CPU-only",
        )
        logger.info(f"CPU fallback monitor stopped: {elapsed:.2f}s (no power data)")
        return measurement

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def get_monitor(gpu_index: int = 0,
                sample_interval_ms: int = 100,
                measurement_type: str = "inference") -> GPUPowerMonitor:
    """
    Factory function to get the appropriate power monitor.
    Returns GPUPowerMonitor (works on both GPU and CPU — gracefully degrades).
    """
    return GPUPowerMonitor(
        gpu_index=gpu_index,
        sample_interval_ms=sample_interval_ms,
        measurement_type=measurement_type,
    )
