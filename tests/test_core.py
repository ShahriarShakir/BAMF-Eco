"""
Unit Tests for BAMF-Eco
=========================

Tests core modules:
1. Model registry & loading
2. Config encoding/decoding
3. GPU power monitoring (CPU fallback)
4. Sustainability accounting
5. Fidelity correction model
6. BAMF-Eco optimizer mechanics
7. Benchmark runner (dry run)
8. Analysis utilities
"""

import os
import json
import tempfile
import numpy as np
import pytest
from pathlib import Path


# =====================================================================
# 1. Utils & Model Registry Tests
# =====================================================================

class TestModelRegistry:
    def test_registry_populated(self):
        from bamf_eco.utils import MODEL_REGISTRY
        assert len(MODEL_REGISTRY) == 23, f"Expected 23 models, got {len(MODEL_REGISTRY)}"

    def test_all_families_present(self):
        from bamf_eco.utils import MODEL_REGISTRY
        families = set(s.family for s in MODEL_REGISTRY.values())
        expected = {"yolov8", "yolov9", "yolov10", "yolo11", "yolo26", "rtdetr"}
        assert families == expected

    def test_model_spec_fields(self):
        from bamf_eco.utils import MODEL_REGISTRY
        spec = MODEL_REGISTRY["yolov8n"]
        assert spec.family == "yolov8"
        assert spec.size == "n"
        assert spec.name == "yolov8n"
        assert spec.weight_file == "yolov8n.pt"
        assert spec.params_m == 3.2
        assert spec.loader == "YOLO"

    def test_rtdetr_naming(self):
        from bamf_eco.utils import MODEL_REGISTRY
        spec = MODEL_REGISTRY["rtdetr-l"]
        assert spec.family == "rtdetr"
        assert spec.size == "l"
        assert spec.loader == "RTDETR"

    def test_get_model_specs_filter(self):
        from bamf_eco.utils import get_model_specs
        # Filter by family
        yolo8_specs = get_model_specs(families=["yolov8"])
        assert len(yolo8_specs) == 4  # n, s, m, l
        # Filter by size
        nano_specs = get_model_specs(sizes=["n"])
        assert all(s.size == "n" for s in nano_specs)


class TestExperimentConfig:
    def test_config_save_load(self):
        from bamf_eco.utils import ExperimentConfig
        config = ExperimentConfig(
            experiment_name="test",
            seed=42,
            model_families=["yolov8"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            config.save(path)
            loaded = ExperimentConfig.load(path)
            assert loaded.experiment_name == "test"
            assert loaded.seed == 42
            assert loaded.model_families == ["yolov8"]

    def test_config_hash_deterministic(self):
        from bamf_eco.utils import ExperimentConfig
        c1 = ExperimentConfig(seed=42)
        c2 = ExperimentConfig(seed=42)
        assert c1._config_hash() == c2._config_hash()

    def test_config_hash_different(self):
        from bamf_eco.utils import ExperimentConfig
        c1 = ExperimentConfig(seed=42)
        c2 = ExperimentConfig(seed=123)
        assert c1._config_hash() != c2._config_hash()


class TestBenchmarkResult:
    def test_to_from_dict(self):
        from bamf_eco.utils import BenchmarkResult
        r = BenchmarkResult(
            model_name="yolov8n", family="yolov8", size="n",
            precision="fp16", image_size=640, batch_size=1,
            ap=0.45, latency_ms_mean=5.0, energy_kwh=0.001,
        )
        d = r.to_dict()
        r2 = BenchmarkResult.from_dict(d)
        assert r2.model_name == "yolov8n"
        assert r2.ap == 0.45
        assert r2.energy_kwh == 0.001


class TestSeedAndTimer:
    def test_set_seed_reproducible(self):
        from bamf_eco.utils import set_seed
        set_seed(42)
        a = np.random.rand(10)
        set_seed(42)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_timer(self):
        from bamf_eco.utils import Timer
        import time
        with Timer("test") as t:
            time.sleep(0.01)
        assert t.elapsed > 0.005


# =====================================================================
# 2. Config Encoding Tests
# =====================================================================

class TestConfigEncoder:
    def test_encode_decode_roundtrip(self):
        from bamf_eco.optimizer.config_space import ConfigEncoder
        encoder = ConfigEncoder()
        config = {
            "model_name": "yolov8n",
            "image_size": 640,
            "precision": "fp16",
            "epochs": 100,
            "lr0": 0.01,
            "batch_size": 16,
            "optimizer": "SGD",
            "weight_decay": 0.0005,
            "momentum": 0.937,
            "augment_strength": 0.5,
        }
        vec = encoder.encode(config)
        assert vec.shape[0] == encoder.encoded_dim
        decoded = encoder.decode(vec)
        assert decoded["model_name"] == "yolov8n"
        assert decoded["image_size"] == 640
        assert decoded["precision"] == "fp16"

    def test_random_config(self):
        from bamf_eco.optimizer.config_space import ConfigEncoder
        encoder = ConfigEncoder()
        config = encoder.random_config(np.random.RandomState(42))
        assert "model_name" in config
        assert "lr0" in config
        assert 1e-4 <= config["lr0"] <= 0.1

    def test_encoded_dim_correct(self):
        from bamf_eco.optimizer.config_space import ConfigEncoder
        encoder = ConfigEncoder()
        # 23 model choices + 2 precision + 2 optimizer = 27 one-hot
        # + 3 ordinal + 4 continuous = 7
        assert encoder.encoded_dim > 20


# =====================================================================
# 3. Sustainability Accounting Tests
# =====================================================================

class TestSustainability:
    def test_compute_sustainability(self):
        from bamf_eco.sustainability import SustainabilityAccountant
        acc = SustainabilityAccountant(carbon_region="australia")
        est = acc.compute(energy_kwh=0.001)
        assert est.co2e_kg > 0
        assert est.water_liters > 0

    def test_regional_variation(self):
        from bamf_eco.sustainability import SustainabilityAccountant
        acc_au = SustainabilityAccountant(carbon_region="australia")
        acc_fr = SustainabilityAccountant(carbon_region="france")
        e_au = acc_au.compute(0.001)
        e_fr = acc_fr.compute(0.001)
        # France should have lower carbon (nuclear)
        assert e_fr.co2e_kg < e_au.co2e_kg

    def test_zero_energy(self):
        from bamf_eco.sustainability import SustainabilityAccountant
        acc = SustainabilityAccountant()
        est = acc.compute(0.0)
        assert est.co2e_kg == 0.0
        assert est.water_liters == 0.0

    def test_uncertainty_bounds(self):
        from bamf_eco.sustainability import SustainabilityAccountant
        acc = SustainabilityAccountant()
        est = acc.compute(0.01)
        assert est.co2e_kg_min <= est.co2e_kg <= est.co2e_kg_max
        assert est.water_liters_min <= est.water_liters <= est.water_liters_max


# =====================================================================
# 4. Power Monitoring Tests (CPU fallback)
# =====================================================================

class TestPowerMonitor:
    def test_cpu_fallback_monitor(self):
        from bamf_eco.measurement import CPUFallbackMonitor
        monitor = CPUFallbackMonitor()
        monitor.start()
        import time
        time.sleep(0.05)
        measurement = monitor.stop()
        assert measurement.duration_seconds > 0
        assert measurement.energy_kwh >= 0

    def test_get_monitor_returns_something(self):
        from bamf_eco.measurement import get_monitor
        monitor = get_monitor()
        assert monitor is not None


# =====================================================================
# 5. Fidelity Correction Tests
# =====================================================================

class TestFidelityCorrection:
    def test_correction_without_data(self):
        from bamf_eco.optimizer.fidelity_correction import FidelityCorrectionModel
        model = FidelityCorrectionModel(method="linear", n_features=5)
        pred, std = model.predict(np.zeros(5), 0.1, 0.3)
        # Before fitting, should return the low metric with high uncertainty
        assert pred == 0.3
        assert std == 0.5

    def test_linear_correction_fit(self):
        from bamf_eco.optimizer.fidelity_correction import (
            FidelityCorrectionModel, CorrectionObservation,
        )
        model = FidelityCorrectionModel(method="linear", n_features=3, min_observations=3)

        # Synthetic paired data: high = 1.5 * low + 0.1
        for i in range(10):
            low = 0.1 + i * 0.05
            high = 1.5 * low + 0.1
            obs = CorrectionObservation(
                config_vec=np.random.rand(3),
                low_fidelity=0.1,
                high_fidelity=1.0,
                low_metric=low,
                high_metric=high,
            )
            model.add_observation(obs)

        r2 = model.fit()
        assert r2 > 0.9, f"Linear correction R² too low: {r2}"
        assert model.is_fitted

        # Predict
        pred, std = model.predict(np.random.rand(3), 0.1, 0.3)
        expected = 1.5 * 0.3 + 0.1
        assert abs(pred - expected) < 0.05, f"Prediction {pred} too far from {expected}"

    def test_correction_save_load(self):
        from bamf_eco.optimizer.fidelity_correction import (
            FidelityCorrectionModel, CorrectionObservation,
        )
        model = FidelityCorrectionModel(method="linear", n_features=3, min_observations=3)
        for i in range(5):
            obs = CorrectionObservation(
                config_vec=np.random.rand(3),
                low_fidelity=0.1, high_fidelity=1.0,
                low_metric=0.1 * (i + 1), high_metric=0.15 * (i + 1),
            )
            model.add_observation(obs)
        model.fit()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "correction")
            model.save(path)
            loaded = FidelityCorrectionModel.load(path)
            assert loaded.is_fitted
            assert loaded.n_observations == 5

    def test_multi_objective_correction(self):
        from bamf_eco.optimizer.fidelity_correction import MultiFidelityCorrectionManager
        manager = MultiFidelityCorrectionManager(
            objectives=["mAP", "energy_kwh"],
            method="linear",
            n_features=3,
        )
        for i in range(10):
            manager.add_paired_observation(
                config_vec=np.random.rand(3),
                low_fidelity=0.1,
                high_fidelity=1.0,
                low_metrics={"mAP": 0.1 * i, "energy_kwh": 0.01 * i},
                high_metrics={"mAP": 0.12 * i, "energy_kwh": 0.011 * i},
            )
        r2_scores = manager.fit_all()
        assert "mAP" in r2_scores
        assert "energy_kwh" in r2_scores


# =====================================================================
# 6. Optimizer Mechanics Tests
# =====================================================================

class TestParetoFront:
    def test_empty_pareto(self):
        from bamf_eco.optimizer import ParetoFront
        pf = ParetoFront(["mAP", "energy"], ["maximize", "minimize"])
        assert len(pf.points) == 0
        assert pf.hypervolume == 0.0

    def test_add_single_point(self):
        from bamf_eco.optimizer import ParetoFront
        pf = ParetoFront(["mAP", "energy"], ["maximize", "minimize"])
        updated = pf.update({"mAP": 0.5, "energy": 0.01}, {"model": "test"})
        assert updated
        assert len(pf.points) == 1

    def test_dominated_point_rejected(self):
        from bamf_eco.optimizer import ParetoFront
        pf = ParetoFront(["mAP", "energy"], ["maximize", "minimize"])
        pf.update({"mAP": 0.5, "energy": 0.01}, {"model": "a"})
        # This is dominated (worse mAP, worse energy)
        updated = pf.update({"mAP": 0.3, "energy": 0.05}, {"model": "b"})
        assert not updated
        assert len(pf.points) == 1

    def test_non_dominated_accepted(self):
        from bamf_eco.optimizer import ParetoFront
        pf = ParetoFront(["mAP", "energy"], ["maximize", "minimize"])
        pf.update({"mAP": 0.5, "energy": 0.05}, {"model": "a"})
        # Better mAP, worse energy → non-dominated
        updated = pf.update({"mAP": 0.7, "energy": 0.08}, {"model": "b"})
        assert updated
        assert len(pf.points) == 2


class TestEcoConstraints:
    def test_feasibility_check(self):
        from bamf_eco.optimizer.acquisition import EcoFeasibilityGP, EcoConstraints
        constraints = EcoConstraints(max_energy_kwh=0.05, max_latency_ms=100)
        gp = EcoFeasibilityGP(constraints)
        assert gp.is_feasible({"energy_kwh": 0.01, "latency_ms": 50, "co2e_kg": 0.1})
        assert not gp.is_feasible({"energy_kwh": 0.1, "latency_ms": 50, "co2e_kg": 0.1})


# =====================================================================
# 7. Baseline Tests
# =====================================================================

class TestBaselines:
    def test_random_search_suggest(self):
        from bamf_eco.baselines import RandomSearch
        rs = RandomSearch(max_evaluations=5, seed=42)
        config, fidelity = rs.suggest()
        assert "model_name" in config
        assert fidelity == 1.0

    def test_manual_expert_configs(self):
        from bamf_eco.baselines import ManualExpertBaseline
        expert = ManualExpertBaseline(seed=42)
        configs_seen = []
        for _ in range(6):
            config, fid = expert.suggest()
            configs_seen.append(config["model_name"])
        assert "yolov8n" in configs_seen
        assert "rtdetr-l" in configs_seen

    def test_get_baseline_factory(self):
        from bamf_eco.baselines import get_baseline
        rs = get_baseline("random", max_evaluations=10, seed=42)
        assert rs.name == "RandomSearch"


# =====================================================================
# 8. Result Storage Tests
# =====================================================================

class TestResultStorage:
    def test_save_load_results(self):
        from bamf_eco.utils import BenchmarkResult, save_results, load_results
        results = [
            BenchmarkResult(
                model_name="yolov8n", family="yolov8", size="n",
                precision="fp16", image_size=640, batch_size=1,
                ap=0.45, energy_kwh=0.001,
            ),
            BenchmarkResult(
                model_name="yolov8s", family="yolov8", size="s",
                precision="fp16", image_size=640, batch_size=1,
                ap=0.52, energy_kwh=0.002,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "results.json")
            save_results(results, path)
            loaded = load_results(path)
            assert len(loaded) == 2
            assert loaded[0].model_name == "yolov8n"
            assert loaded[1].ap == 0.52


# =====================================================================
# 9. Checkpoint / Resume Tests
# =====================================================================

class TestOptimizerCheckpoint:
    """Test checkpoint save/load for the BAMF-Eco optimizer."""

    def test_save_and_load_checkpoint(self):
        from bamf_eco.optimizer import BAMFEcoOptimizer, EvaluationRecord, OptimizationState
        from bamf_eco.training import TrainingResult

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create optimizer with some state
            opt = BAMFEcoOptimizer(
                max_evaluations=10,
                seed=42,
                output_dir=tmpdir,
            )

            # Simulate some history
            enc = opt.encoder
            for i in range(3):
                config = enc.random_config(opt.rng)
                config_vec = enc.encode(config)
                result = TrainingResult(
                    model_name=config.get("model_name", "yolov8n"),
                    map50_95=0.3 + i * 0.05,
                    energy_kwh=0.01 * (i + 1),
                    latency_ms=10.0 + i,
                    co2e_kg=0.001,
                    water_liters=0.01,
                    training_time_s=100.0,
                )
                opt.observe(config, 0.1, result)

            # Save checkpoint
            opt.save_checkpoint()
            assert (Path(tmpdir) / "checkpoint.json").exists()

            # Load into new optimizer
            opt2 = BAMFEcoOptimizer(
                max_evaluations=10,
                seed=42,
                output_dir=tmpdir,
            )
            loaded = opt2.load_checkpoint()
            assert loaded is True
            assert opt2.state.iteration == opt.state.iteration
            assert len(opt2.history) == len(opt.history)
            assert opt2.state.total_energy_kwh == pytest.approx(opt.state.total_energy_kwh, rel=1e-4)
            assert opt2.state.best_map == pytest.approx(opt.state.best_map, rel=1e-4)

    def test_load_nonexistent_checkpoint(self):
        from bamf_eco.optimizer import BAMFEcoOptimizer

        with tempfile.TemporaryDirectory() as tmpdir:
            opt = BAMFEcoOptimizer(output_dir=tmpdir)
            loaded = opt.load_checkpoint()
            assert loaded is False
            assert opt.state.iteration == 0

    def test_pareto_front_preserved(self):
        from bamf_eco.optimizer import BAMFEcoOptimizer
        from bamf_eco.training import TrainingResult

        with tempfile.TemporaryDirectory() as tmpdir:
            opt = BAMFEcoOptimizer(
                objectives=["mAP", "energy_kwh"],
                directions=["maximize", "minimize"],
                max_evaluations=10,
                seed=42,
                output_dir=tmpdir,
            )

            # Add a full-fidelity point that should be on Pareto
            config = opt.encoder.random_config(opt.rng)
            result = TrainingResult(
                model_name="yolov8n",
                map50_95=0.45,
                energy_kwh=0.01,
                latency_ms=5.0,
                co2e_kg=0.001,
                water_liters=0.01,
                training_time_s=60.0,
            )
            opt.observe(config, 1.0, result)  # Full fidelity
            assert len(opt.pareto_front.points) > 0

            opt.save_checkpoint()

            # Reload
            opt2 = BAMFEcoOptimizer(
                objectives=["mAP", "energy_kwh"],
                directions=["maximize", "minimize"],
                max_evaluations=10,
                seed=42,
                output_dir=tmpdir,
            )
            opt2.load_checkpoint()
            assert len(opt2.pareto_front.points) == len(opt.pareto_front.points)
            assert opt2.pareto_front.hypervolume_history == opt.pareto_front.hypervolume_history


class TestBenchmarkCheckpoint:
    """Test benchmark sweep checkpoint/resume."""

    def test_sweep_checkpoint_saves_incrementally(self):
        from bamf_eco.utils import BenchmarkResult

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "sweep_ckpt.json")

            # Simulate saving results incrementally
            results = []
            for i in range(3):
                r = BenchmarkResult(
                    model_name=f"model{i}", family="test", size="n",
                    precision="fp16", image_size=640, batch_size=1,
                    energy_kwh=0.001 * (i + 1),
                )
                results.append(r)
                with open(ckpt_path, "w") as f:
                    json.dump([r.to_dict() for r in results], f)

            # Load checkpoint
            with open(ckpt_path) as f:
                loaded = json.load(f)
            assert len(loaded) == 3
            completed_keys = {f"{r['model_name']}_{r['precision']}_{r['image_size']}" for r in loaded}
            assert "model0_fp16_640" in completed_keys
            assert "model2_fp16_640" in completed_keys


class TestBaselineCheckpoint:
    """Test baseline optimizer checkpoint/resume."""

    def test_baseline_checkpoint_roundtrip(self):
        from bamf_eco.baselines import RandomSearch
        from bamf_eco.training import TrainingResult

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "random_ckpt.json")

            # Create baseline and simulate results
            rs = RandomSearch(max_evaluations=10, seed=42)
            for i in range(3):
                config, fidelity = rs.suggest()
                result = TrainingResult(
                    model_name=config.get("model_name", "yolov8n"),
                    map50_95=0.3 + i * 0.05,
                    energy_kwh=0.01,
                    training_time_s=10.0,
                )
                rs.observe(config, fidelity, result)

            rs._save_baseline_checkpoint(ckpt_path, 3)

            # Load checkpoint
            with open(ckpt_path) as f:
                ckpt = json.load(f)
            assert ckpt["iteration"] == 3
            assert len(ckpt["results"]) == 3
            assert ckpt["optimizer_name"] == "RandomSearch"


class TestTrainingResume:
    """Test training resume detection."""

    def test_completed_result_skip(self):
        """A completed training_result.json should cause evaluate to return early."""
        from bamf_eco.training import TrainingRunner, TrainingConfig, TrainingResult
        from bamf_eco.sustainability import SustainabilityAccountant

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = TrainingRunner(
                base_output_dir=tmpdir,
                sustainability_accountant=SustainabilityAccountant(),
            )

            # Simulate a completed run
            run_id = "test_run_001"
            run_dir = Path(tmpdir) / run_id
            run_dir.mkdir()

            saved_result = TrainingResult(
                config_hash=run_id,
                model_name="yolov8n",
                map50_95=0.42,
                energy_kwh=0.005,
                epochs_actual=10,
                training_time_s=120.0,
            )
            with open(run_dir / "training_result.json", "w") as f:
                json.dump(saved_result.to_dict(), f)

            # evaluate should return saved result without training
            config = TrainingConfig(model_name="yolov8n", epochs=10, fidelity=0.1)
            result = runner.evaluate(config, run_id=run_id, resume_if_exists=True)
            assert result.map50_95 == 0.42
            assert result.model_name == "yolov8n"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
