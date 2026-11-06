"""
Unit and Integration Tests for HMM Anomaly Detector.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.sensor_simulator import MachineSensorSimulator
from src.model_selection import calculate_hmm_free_parameters, HMMModelSelector
from src.anomaly_detector import HMMAnomalyDetector, StreamingAlert
from src.evaluator import DetectorEvaluator


@pytest.fixture
def sample_data():
    simulator = MachineSensorSimulator(random_seed=42, sampling_rate_hz=10.0)
    train_df, _ = simulator.generate_normal_stream(n_samples=500)
    val_df, _ = simulator.generate_normal_stream(n_samples=200)
    test_df, events = simulator.generate_test_stream_with_anomalies(n_samples=800)
    
    features = ["vibration", "temperature", "rpm"]
    return {
        "train": train_df[features].values,
        "val": val_df[features].values,
        "test": test_df[features].values,
        "y_test": test_df["is_anomaly"].values,
        "events": events
    }


def test_sensor_simulator(sample_data):
    X_train = sample_data["train"]
    X_test = sample_data["test"]
    events = sample_data["events"]

    assert X_train.shape == (500, 3)
    assert X_test.shape == (800, 3)
    assert len(events) > 0
    for ev in events:
        assert 0 <= ev.start_idx < ev.end_idx <= 800


def test_free_parameters_calculation():
    # K=3, D=3, full covariance
    # (3-1) + 3*2 + 3*3 + 3*(3*4/2) = 2 + 6 + 9 + 18 = 35
    params_full = calculate_hmm_free_parameters(3, 3, "full")
    assert params_full == 35

    # diag covariance: 2 + 6 + 9 + 3*3 = 26
    params_diag = calculate_hmm_free_parameters(3, 3, "diag")
    assert params_diag == 26


def test_detector_training_and_scoring(sample_data):
    X_train = sample_data["train"]
    detector = HMMAnomalyDetector(n_components=3, n_iter=30, random_state=42)
    
    assert not detector.is_fitted
    detector.fit(X_train)
    assert detector.is_fitted

    seq_score = detector.score_sequence(X_train[:50])
    assert isinstance(seq_score, float)
    assert not np.isnan(seq_score)


def test_sliding_window_scoring(sample_data):
    X_train = sample_data["train"]
    detector = HMMAnomalyDetector(n_components=3, n_iter=30, random_state=42)
    detector.fit(X_train)

    w_size = 20
    indices, scores = detector.score_sliding_windows(X_train, window_size=w_size, step_size=1)
    
    expected_count = len(X_train) - w_size + 1
    assert len(indices) == expected_count
    assert len(scores) == expected_count
    assert indices[0] == w_size - 1
    assert indices[-1] == len(X_train) - 1


def test_threshold_calibration(sample_data):
    X_train = sample_data["train"]
    detector = HMMAnomalyDetector(n_components=3, n_iter=30, random_state=42)
    detector.fit(X_train)

    thresh_perc = detector.calibrate_threshold(X_train, window_size=20, method="percentile", percentile=2.0)
    assert isinstance(thresh_perc, float)
    assert detector.threshold_ == thresh_perc

    thresh_sigma = detector.calibrate_threshold(X_train, window_size=20, method="sigma", sigma_multiplier=3.0)
    assert isinstance(thresh_sigma, float)


def test_streaming_generator(sample_data):
    X_train = sample_data["train"]
    X_test = sample_data["test"]

    detector = HMMAnomalyDetector(n_components=3, n_iter=30, random_state=42)
    detector.fit(X_train)
    detector.calibrate_threshold(X_train, window_size=20, percentile=1.0)

    alerts = list(detector.process_realtime_stream(X_test[:100], window_size=20, persistence_count=2))
    # Once window of size 20 is full, alerts start at index 19 through 99 (81 alerts)
    assert len(alerts) == 81
    assert isinstance(alerts[0], StreamingAlert)
    assert alerts[0].sample_index == 19


def test_model_serialization(sample_data):
    X_train = sample_data["train"]
    detector = HMMAnomalyDetector(n_components=3, n_iter=30, random_state=42)
    detector.fit(X_train)
    detector.calibrate_threshold(X_train, window_size=20, percentile=1.0)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        detector.save(tmp_path)
        loaded = HMMAnomalyDetector.load(tmp_path)
        assert loaded.is_fitted
        assert loaded.threshold_ == detector.threshold_
        assert loaded.calibrated_window_size_ == detector.calibrated_window_size_

        score_orig = detector.score_sequence(X_train[:30])
        score_loaded = loaded.score_sequence(X_train[:30])
        np.testing.assert_almost_equal(score_orig, score_loaded, decimal=4)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
