"""
Dataset generation script for normal and anomalous machine sensor streams.
"""

import os
from pathlib import Path
from src.sensor_simulator import MachineSensorSimulator
import config


def generate_all_datasets():
    data_dir = config.DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    simulator = MachineSensorSimulator(
        random_seed=config.SensorConfig.random_seed,
        sampling_rate_hz=config.SensorConfig.sampling_rate_hz
    )

    fs = config.SensorConfig.sampling_rate_hz
    n_train = int(config.SensorConfig.train_duration_sec * fs)
    n_val = int(config.SensorConfig.val_duration_sec * fs)
    n_test = int(config.SensorConfig.test_duration_sec * fs)

    print(f"Generating training set ({n_train} samples, {config.SensorConfig.train_duration_sec}s)...")
    train_df, _ = simulator.generate_normal_stream(n_samples=n_train)
    train_path = data_dir / "normal_sensor_train.csv"
    train_df.to_csv(train_path, index=False)
    print(f"Saved: {train_path}")

    print(f"Generating validation set ({n_val} samples, {config.SensorConfig.val_duration_sec}s)...")
    val_df, _ = simulator.generate_normal_stream(n_samples=n_val)
    val_path = data_dir / "normal_sensor_val.csv"
    val_df.to_csv(val_path, index=False)
    print(f"Saved: {val_path}")

    print(f"Generating test stream with anomalies ({n_test} samples, {config.SensorConfig.test_duration_sec}s)...")
    test_df, anomalies = simulator.generate_test_stream_with_anomalies(n_samples=n_test)
    test_path = data_dir / "test_sensor_stream.csv"
    test_df.to_csv(test_path, index=False)
    print(f"Saved: {test_path} ({len(anomalies)} injected anomaly intervals)")

    return train_path, val_path, test_path


if __name__ == "__main__":
    generate_all_datasets()
