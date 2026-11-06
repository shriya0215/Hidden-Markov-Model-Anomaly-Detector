"""
Configuration parameters for HMM Anomaly Detection System.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SensorConfig:
    # Sampling & Stream settings
    sampling_rate_hz: float = 10.0  # 10 samples per second
    train_duration_sec: float = 600.0  # 10 minutes of normal training data (6,000 samples)
    val_duration_sec: float = 200.0  # ~3.3 minutes validation data (2,000 samples)
    test_duration_sec: float = 800.0  # ~13.3 minutes test stream (8,000 samples)
    random_seed: int = 42

    # Physical machine sensor baseline parameters
    # Normal operation alternates between 3 distinct operating regimes:
    # 0: Idle/Warmup (RPM ~ 1200, Temp ~ 45C, Vibration ~ 0.5 g)
    # 1: Steady Operation (RPM ~ 2400, Temp ~ 68C, Vibration ~ 1.2 g)
    # 2: High-Load Machining (RPM ~ 3600, Temp ~ 85C, Vibration ~ 2.4 g)
    base_rpm: float = 2400.0
    base_vibration: float = 1.2
    base_temperature: float = 68.0


@dataclass(frozen=True)
class ModelConfig:
    # Model Selection range
    n_states_range: tuple = (2, 3, 4, 5, 6, 7, 8)
    covariance_type: str = "full"  # "full", "diag", "spherical"
    n_iter: int = 150
    tol: float = 1e-4
    n_restarts: int = 5
    random_state: int = 42


@dataclass(frozen=True)
class DetectionConfig:
    # Sliding window settings
    window_size: int = 30  # 3 seconds window at 10Hz
    step_size: int = 1  # Continuous sliding window (step = 1 sample)
    
    # Thresholding policy:
    # "percentile" (e.g. 1st or 2nd percentile of normal window likelihoods)
    # "sigma" (mean - k * std of normal window likelihoods)
    # "iqr" (Q1 - 1.5 * IQR)
    threshold_method: str = "percentile"
    percentile_threshold: float = 1.0  # 1st percentile of training distribution (1% FDR target)
    sigma_multiplier: float = 3.0  # 3-sigma rule
    
    # Persistence filter: number of consecutive anomalous windows required for alarm
    persistence_count: int = 3


# Paths
WORKSPACE_ROOT = Path(__file__).resolve().parent
DATA_DIR = WORKSPACE_ROOT / "data"
OUTPUT_DIR = WORKSPACE_ROOT / "visualizations"
MODEL_DIR = WORKSPACE_ROOT / "models"
