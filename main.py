"""
Main pipeline execution for Hidden Markov Model Anomaly Detection.

Workflow:
1. Ensure datasets exist (or generate them)
2. Perform rigorous model selection (BIC/AIC/Validation Log-Likelihood)
3. Train optimal GaussianHMM model on normal sensor data
4. Calibrate anomaly threshold on normal sliding window likelihood distribution
5. Stream continuous test sensor data and detect anomalies in real-time
6. Evaluate detection accuracy, false alarm rates, and detection latency
7. Generate and save publication-quality visualizations
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure workspace root is in python path
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import config
from data.generate_data import generate_all_datasets
from src.sensor_simulator import MachineSensorSimulator
from src.model_selection import HMMModelSelector
from src.anomaly_detector import HMMAnomalyDetector
from src.evaluator import DetectorEvaluator
from src.visualizer import AnomalyVisualizer


def run_pipeline():
    print("=" * 80)
    print(" HIDDEN MARKOV MODEL REAL-TIME ANOMALY DETECTION SYSTEM ")
    print("=" * 80)

    # 1. Dataset Generation / Loading
    train_path = config.DATA_DIR / "normal_sensor_train.csv"
    val_path = config.DATA_DIR / "normal_sensor_val.csv"
    test_path = config.DATA_DIR / "test_sensor_stream.csv"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        print("\n[Step 1/7] Datasets not found. Generating simulated sensor datasets...")
        generate_all_datasets()
    else:
        print("\n[Step 1/7] Loading existing sensor datasets...")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    feature_cols = ["vibration", "temperature", "rpm"]
    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_test = test_df["is_anomaly"].values
    t_test = test_df["timestamp_sec"].values

    print(f"  - Training Normal Data:   {X_train.shape[0]} samples ({X_train.shape[0]/config.SensorConfig.sampling_rate_hz:.1f}s)")
    print(f"  - Validation Normal Data: {X_val.shape[0]} samples ({X_val.shape[0]/config.SensorConfig.sampling_rate_hz:.1f}s)")
    print(f"  - Test Stream Data:       {X_test.shape[0]} samples ({X_test.shape[0]/config.SensorConfig.sampling_rate_hz:.1f}s)")
    print(f"  - Features: {feature_cols}")

    # 2. Model Selection (BIC / AIC / Holdout Validation)
    print("\n[Step 2/7] Running Model Selection (K in 2..8 states)...")
    selector = HMMModelSelector(
        candidate_states=config.ModelConfig.n_states_range,
        covariance_type=config.ModelConfig.covariance_type,
        n_iter=config.ModelConfig.n_iter,
        tol=config.ModelConfig.tol,
        n_restarts=config.ModelConfig.n_restarts,
        random_state=config.ModelConfig.random_state
    )

    selection_df = selector.fit_evaluate(X_train, X_val)
    print("\nModel Selection Summary Table:")
    print("-" * 90)
    print(f"{'K':>3} | {'Params':>6} | {'Train LL':>12} | {'Val LL':>12} | {'AIC':>12} | {'BIC':>12} | {'Val LL/step':>11}")
    print("-" * 90)
    for _, row in selection_df.iterrows():
        print(f"{int(row['n_components']):3d} | {int(row['n_parameters']):6d} | {row['train_log_likelihood']:12.2f} | "
              f"{row['val_log_likelihood']:12.2f} | {row['aic']:12.2f} | {row['bic']:12.2f} | {row['val_ll_per_sample']:11.4f}")
    print("-" * 90)

    summary = selector.get_selection_summary()
    best_k = summary["best_k"]
    print(f"\nOptimal Model Decision: K = {best_k} Hidden States")
    print(f"Rationale: {summary['rationale']}")

    # 3. Train Selected Optimal Model
    print(f"\n[Step 3/7] Training Final GaussianHMM (K={best_k}, Covariance='{config.ModelConfig.covariance_type}')...")
    detector = HMMAnomalyDetector(
        n_components=best_k,
        covariance_type=config.ModelConfig.covariance_type,
        n_iter=config.ModelConfig.n_iter,
        tol=config.ModelConfig.tol,
        random_state=config.ModelConfig.random_state,
        normalize_inputs=True
    )
    detector.fit(X_train)
    print(f"Model converged: {detector.model.monitor_.converged} in {detector.model.monitor_.iter} iterations.")

    # 4. Calibrate Anomaly Threshold on Normal Training Windows
    print("\n[Step 4/7] Calibrating Anomaly Detection Threshold on Normal Windows...")
    threshold = detector.calibrate_threshold(
        X_train,
        window_size=config.DetectionConfig.window_size,
        step_size=config.DetectionConfig.step_size,
        method=config.DetectionConfig.threshold_method,
        percentile=config.DetectionConfig.percentile_threshold,
        sigma_multiplier=config.DetectionConfig.sigma_multiplier,
        normalize_by_length=True
    )

    train_scores = detector.train_window_scores_
    print(f"  - Window Size: {config.DetectionConfig.window_size} samples ({config.DetectionConfig.window_size/config.SensorConfig.sampling_rate_hz:.1f}s)")
    print(f"  - Training Normal Windows Count: {len(train_scores)}")
    print(f"  - Normal Windows Score Mean:     {np.mean(train_scores):.4f}")
    print(f"  - Normal Windows Score Std:      {np.std(train_scores):.4f}")
    print(f"  - Calibrated Threshold ({config.DetectionConfig.threshold_method}={config.DetectionConfig.percentile_threshold}%): {threshold:.4f}")

    # 5. Real-Time Streaming Detection on Test Stream
    print("\n[Step 5/7] Simulating Real-Time Online Stream Processing...")
    window_indices, window_scores = detector.score_sliding_windows(
        X_test,
        window_size=config.DetectionConfig.window_size,
        step_size=config.DetectionConfig.step_size,
        normalize_by_length=True
    )
    detected_mask = window_scores < threshold

    # Simulate online tick-by-tick streaming generator
    streaming_alerts = list(detector.process_realtime_stream(
        X_test,
        timestamps_sec=t_test,
        window_size=config.DetectionConfig.window_size,
        persistence_count=config.DetectionConfig.persistence_count
    ))
    persisted_alarms = [a for a in streaming_alerts if a.is_persisted_alarm]
    print(f"  - Total Stream Ticks Processed: {len(streaming_alerts)}")
    print(f"  - Instantaneous Flagged Windows: {np.sum(detected_mask)} / {len(window_scores)} ({100*np.mean(detected_mask):.2f}%)")
    print(f"  - Persisted Alarms (Filtered):   {len(persisted_alarms)} alarm ticks")

    # 6. Quantitative Evaluation
    print("\n[Step 6/7] Evaluating Detection Performance against Ground Truth...")
    y_test_aligned = y_test[window_indices]
    neg_scores = -window_scores  # Higher score = higher anomaly likelihood

    metrics = DetectorEvaluator.evaluate_detections(
        y_true=y_test_aligned,
        y_pred=detected_mask.astype(int),
        anomaly_scores=neg_scores
    )

    print("\nEvaluation Metrics (Sample/Window Aligned):")
    print(f"  - Precision:    {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  - Recall:       {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  - F1-Score:     {metrics['f1_score']:.4f}")
    print(f"  - Specificity:  {metrics['specificity']:.4f}")
    print(f"  - ROC-AUC:      {metrics['roc_auc']:.4f}")
    print(f"  - PR-AUC:       {metrics['pr_auc']:.4f}")
    print(f"  - True Pos:     {metrics['true_positives']} | False Pos: {metrics['false_positives']}")
    print(f"  - True Neg:     {metrics['true_negatives']} | False Neg: {metrics['false_negatives']}")

    # Latency evaluation per injected anomaly event
    simulator = MachineSensorSimulator(
        random_seed=config.SensorConfig.random_seed,
        sampling_rate_hz=config.SensorConfig.sampling_rate_hz
    )
    _, anomalies = simulator.generate_test_stream_with_anomalies(n_samples=len(X_test))

    flagged_indices = window_indices[detected_mask]
    latency_df = DetectorEvaluator.evaluate_event_latency(
        anomalies,
        flagged_indices,
        sampling_rate_hz=config.SensorConfig.sampling_rate_hz
    )
    print("\nAnomaly Event Detection & Latency:")
    print("-" * 95)
    print(f"{'Anomaly Type':<27} | {'Start':>6} | {'End':>6} | {'Detected?':<9} | {'Latency (sec)':<14} | {'Description'}")
    print("-" * 95)
    for _, r in latency_df.iterrows():
        lat_str = f"{r['latency_sec']:.2f}s" if pd.notna(r['latency_sec']) else "N/A"
        det_str = "YES" if r['detected'] else "NO"
        print(f"{r['anomaly_type']:<27} | {r['start_idx']:6d} | {r['end_idx']:6d} | {det_str:<9} | {lat_str:<14} | {r['description'][:30]}...")
    print("-" * 95)

    # Threshold sensitivity sweep
    sweep_df = DetectorEvaluator.threshold_sensitivity_sweep(
        detector, X_test, y_test, window_size=config.DetectionConfig.window_size
    )
    print("\nThreshold Sensitivity Trade-off Sweep:")
    print("-" * 65)
    print(f"{'Percentile (%)':>14} | {'Threshold':>10} | {'Precision':>9} | {'Recall':>9} | {'F1-Score':>9} | {'FPR':>8}")
    print("-" * 65)
    for _, r in sweep_df.iterrows():
        print(f"{r['percentile']:14.1f} | {r['threshold']:10.3f} | {r['precision']:9.3f} | {r['recall']:9.3f} | {r['f1_score']:9.3f} | {r['false_positive_rate']:8.4f}")
    print("-" * 65)

    # 7. Visualization Generation
    print("\n[Step 7/7] Generating Visualizations...")
    visualizer = AnomalyVisualizer(output_dir=config.OUTPUT_DIR)

    # Decoded hidden states for context
    hidden_states = detector.decode_hidden_states(X_test)

    dash_path = visualizer.plot_detection_dashboard(
        test_df=test_df,
        window_indices=window_indices,
        window_scores=window_scores,
        threshold=threshold,
        detected_mask=detected_mask,
        hidden_states=hidden_states,
        anomalies=anomalies,
        filename="hmm_anomaly_detection_dashboard.png"
    )
    print(f"  - Saved Dashboard: {dash_path}")

    model_sel_path = visualizer.plot_model_selection(
        selection_df=selection_df,
        filename="hmm_model_selection_bic_aic.png"
    )
    print(f"  - Saved Model Selection Plot: {model_sel_path}")

    dist_path = visualizer.plot_likelihood_distribution_and_threshold(
        train_scores=train_scores,
        threshold=threshold,
        threshold_method=config.DetectionConfig.threshold_method,
        percentile_val=config.DetectionConfig.percentile_threshold,
        filename="hmm_likelihood_distribution_threshold.png"
    )
    print(f"  - Saved Likelihood Distribution Plot: {dist_path}")

    # Save model and summary metadata
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = config.MODEL_DIR / "hmm_detector.joblib"
    detector.save(str(model_save_path))
    print(f"\nTrained detector model saved to: {model_save_path}")

    results_export = {
        "best_k": best_k,
        "selection_summary": summary["rationale"],
        "threshold": threshold,
        "metrics": metrics,
        "latency": latency_df.to_dict(orient="records"),
        "threshold_sweep": sweep_df.to_dict(orient="records")
    }
    results_path = config.WORKSPACE_ROOT / "data" / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results_export, f, indent=2)
    print(f"Evaluation metrics exported to: {results_path}")

    print("\n" + "=" * 80)
    print(" HMM ANOMALY DETECTION PIPELINE COMPLETED SUCCESSFULLY ")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
