"""
Performance Evaluation Module for HMM Anomaly Detector.

Computes:
- Binary classification metrics: Precision, Recall, F1-Score, Specificity
- ROC-AUC and PR-AUC using negative log-likelihood
- Event-based detection metrics and latency (time-to-detect)
- Threshold sweep analysis across percentiles
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from src.sensor_simulator import AnomalyEvent


class DetectorEvaluator:
    """
    Evaluates detector performance at both sample level and event level.
    """

    @staticmethod
    def evaluate_detections(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        anomaly_scores: np.ndarray
    ) -> Dict[str, Any]:
        """
        Computes standard classification metrics.
        anomaly_scores: Higher values indicate higher anomaly likelihood (e.g., -normalized_ll).
        """
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # AUC metrics
        try:
            roc_auc = roc_auc_score(y_true, anomaly_scores)
            pr_auc = average_precision_score(y_true, anomaly_scores)
        except Exception:
            roc_auc = float("nan")
            pr_auc = float("nan")

        return {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "specificity": float(specificity),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc)
        }

    @staticmethod
    def evaluate_event_latency(
        events: List[AnomalyEvent],
        detection_indices: np.ndarray,
        sampling_rate_hz: float = 10.0
    ) -> pd.DataFrame:
        """
        Calculates time-to-detect (latency) for each seeded ground truth anomaly event.
        """
        records = []
        detection_set = set(detection_indices)

        for ev in events:
            # Check which detection indices fall within [start_idx, end_idx]
            event_detections = [idx for idx in range(ev.start_idx, ev.end_idx) if idx in detection_set]

            if len(event_detections) > 0:
                first_detect_idx = min(event_detections)
                latency_samples = first_detect_idx - ev.start_idx
                latency_sec = latency_samples / sampling_rate_hz
                detected = True
            else:
                latency_samples = None
                latency_sec = None
                detected = False

            records.append({
                "anomaly_type": ev.anomaly_type,
                "start_idx": ev.start_idx,
                "end_idx": ev.end_idx,
                "duration_sec": (ev.end_idx - ev.start_idx) / sampling_rate_hz,
                "detected": detected,
                "latency_samples": latency_samples,
                "latency_sec": latency_sec,
                "description": ev.description
            })

        return pd.DataFrame(records)

    @staticmethod
    def threshold_sensitivity_sweep(
        detector,
        X_test: np.ndarray,
        y_test: np.ndarray,
        window_size: int = 30,
        percentiles: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    ) -> pd.DataFrame:
        """
        Sweeps anomaly thresholds across multiple percentiles to analyze Precision-Recall trade-off.
        """
        end_indices, test_scores = detector.score_sliding_windows(
            X_test, window_size=window_size, step_size=1, normalize_by_length=True
        )
        y_test_aligned = y_test[end_indices]
        neg_scores = -test_scores

        train_scores = detector.train_window_scores_
        if train_scores is None:
            raise RuntimeError("Detector must have train_window_scores_ calibrated.")

        records = []
        for p in percentiles:
            thresh = np.percentile(train_scores, p)
            y_pred = (test_scores < thresh).astype(int)

            prec = precision_score(y_test_aligned, y_pred, zero_division=0)
            rec = recall_score(y_test_aligned, y_pred, zero_division=0)
            f1 = f1_score(y_test_aligned, y_pred, zero_division=0)
            fpr = np.sum((y_pred == 1) & (y_test_aligned == 0)) / max(1, np.sum(y_test_aligned == 0))

            records.append({
                "percentile": p,
                "threshold": thresh,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "false_positive_rate": fpr
            })

        return pd.DataFrame(records)
