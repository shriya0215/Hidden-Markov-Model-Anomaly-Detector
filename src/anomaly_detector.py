"""
Real-Time HMM Anomaly Detector using hmmlearn.

Features:
- GaussianHMM training on normal multi-sensor time series
- Anomaly threshold calibration from normal window log-likelihood distribution
- Vectorized & streaming sliding window likelihood evaluation
- Persistence filtering for false-alarm suppression
- Online streaming tick-by-tick processor with ring buffer
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Generator, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import joblib


@dataclass
class StreamingAlert:
    sample_index: int
    timestamp_sec: float
    window_log_likelihood: float
    normalized_log_likelihood: float
    threshold: float
    is_anomaly: bool
    is_persisted_alarm: bool
    active_regime: int


class HMMAnomalyDetector:
    """
    Hidden Markov Model-based Sequential Anomaly Detector.
    """

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        n_iter: int = 150,
        tol: float = 1e-4,
        random_state: int = 42,
        normalize_inputs: bool = True
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.normalize_inputs = normalize_inputs

        self.scaler = StandardScaler() if normalize_inputs else None
        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
            init_params="stmc"
        )

        self.is_fitted = False
        self.threshold_: Optional[float] = None
        self.train_window_scores_: Optional[np.ndarray] = None
        self.threshold_method_: Optional[str] = None
        self.calibrated_window_size_: Optional[int] = None

    def fit(self, X: np.ndarray, lengths: Optional[List[int]] = None) -> "HMMAnomalyDetector":
        """
        Fits the scaler (if enabled) and trains the GaussianHMM on normal sensor sequences.
        """
        X_proc = X.copy()
        if self.normalize_inputs:
            X_proc = self.scaler.fit_transform(X_proc)

        self.model.fit(X_proc, lengths=lengths)
        self.is_fitted = True
        return self

    def score_sequence(self, X: np.ndarray) -> float:
        """
        Computes the log-likelihood ln P(X | model) of an observation sequence.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        X_proc = self.scaler.transform(X) if self.normalize_inputs else X
        return self.model.score(X_proc)

    def score_sliding_windows(
        self,
        X: np.ndarray,
        window_size: int = 30,
        step_size: int = 1,
        normalize_by_length: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes log-likelihood for all sliding windows over time-series X.

        Args:
            X: Array of shape (T, D)
            window_size: Number of time steps in each window
            step_size: Step size to advance the window
            normalize_by_length: If True, returns (log_likelihood / window_size)

        Returns:
            window_indices: End indices of each window (sample index aligned to window completion)
            scores: Array of log-likelihood scores for each window
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        X_proc = self.scaler.transform(X) if self.normalize_inputs else X
        T, D = X_proc.shape

        if T < window_size:
            raise ValueError(f"Sequence length {T} is smaller than window size {window_size}.")

        end_indices = []
        scores = []

        for start in range(0, T - window_size + 1, step_size):
            end = start + window_size
            window = X_proc[start:end]
            ll = self.model.score(window)
            if normalize_by_length:
                ll = ll / float(window_size)
            end_indices.append(end - 1)  # timestamp corresponding to latest sample in window
            scores.append(ll)

        return np.array(end_indices), np.array(scores)

    def calibrate_threshold(
        self,
        X_normal: np.ndarray,
        window_size: int = 30,
        step_size: int = 1,
        method: str = "percentile",
        percentile: float = 1.0,
        sigma_multiplier: float = 3.0,
        normalize_by_length: bool = True
    ) -> float:
        """
        Calibrates the anomaly threshold based on the normal training data's likelihood distribution.

        Args:
            X_normal: Normal training or validation sensor data
            window_size: Sliding window size
            method: 'percentile' (e.g. 1st percentile), 'sigma' (mean - k*std), or 'iqr' (Q1 - 1.5*IQR)
            percentile: Target percentile (default 1.0 -> 1% false alarm rate)
            sigma_multiplier: Multiplier for standard deviation rule
            normalize_by_length: Whether to score normalized by window length

        Returns:
            calibrated_threshold
        """
        _, scores = self.score_sliding_windows(
            X_normal,
            window_size=window_size,
            step_size=step_size,
            normalize_by_length=normalize_by_length
        )

        self.train_window_scores_ = scores
        self.calibrated_window_size_ = window_size
        self.threshold_method_ = method

        if method == "percentile":
            self.threshold_ = float(np.percentile(scores, percentile))
        elif method == "sigma":
            mu = float(np.mean(scores))
            sigma = float(np.std(scores))
            self.threshold_ = mu - sigma_multiplier * sigma
        elif method == "iqr":
            q25 = float(np.percentile(scores, 25))
            q75 = float(np.percentile(scores, 75))
            iqr = q75 - q25
            self.threshold_ = q25 - 1.5 * iqr
        else:
            raise ValueError(f"Unknown threshold method: {method}")

        return self.threshold_

    def predict_window(
        self,
        window: np.ndarray,
        normalize_by_length: bool = True
    ) -> Tuple[bool, float]:
        """
        Evaluates a single window against the calibrated threshold.
        Returns (is_anomaly, score).
        """
        if self.threshold_ is None:
            raise RuntimeError("Threshold not calibrated yet. Call calibrate_threshold() first.")

        w_proc = self.scaler.transform(window) if self.normalize_inputs else window
        score = self.model.score(w_proc)
        if normalize_by_length:
            score = score / float(len(window))

        is_anomaly = score < self.threshold_
        return is_anomaly, score

    def decode_hidden_states(self, X: np.ndarray) -> np.ndarray:
        """
        Viterbi decoding of the most likely sequence of hidden states.
        """
        X_proc = self.scaler.transform(X) if self.normalize_inputs else X
        _, state_sequence = self.model.decode(X_proc, algorithm="viterbi")
        return state_sequence

    def process_realtime_stream(
        self,
        stream_data: np.ndarray,
        timestamps_sec: Optional[np.ndarray] = None,
        window_size: int = 30,
        persistence_count: int = 3
    ) -> Generator[StreamingAlert, None, None]:
        """
        Generator simulating real-time online sensor telemetry stream.
        Maintains an internal FIFO ring buffer and evaluates each incoming tick.

        Yields:
            StreamingAlert object at each time tick once the buffer is primed.
        """
        if self.threshold_ is None:
            raise RuntimeError("Threshold must be calibrated before streaming.")

        buffer: List[np.ndarray] = []
        consecutive_anomalies = 0
        n_samples = stream_data.shape[0]

        if timestamps_sec is None:
            timestamps_sec = np.arange(n_samples)

        for i in range(n_samples):
            current_reading = stream_data[i]
            buffer.append(current_reading)

            if len(buffer) > window_size:
                buffer.pop(0)

            if len(buffer) == window_size:
                # Window is full; score this window
                window_arr = np.array(buffer)
                w_proc = self.scaler.transform(window_arr) if self.normalize_inputs else window_arr

                raw_ll = self.model.score(w_proc)
                norm_ll = raw_ll / float(window_size)

                is_anom = norm_ll < self.threshold_
                if is_anom:
                    consecutive_anomalies += 1
                else:
                    consecutive_anomalies = 0

                is_persisted = consecutive_anomalies >= persistence_count

                # Active regime is estimated state of latest sample in window
                _, state_path = self.model.decode(w_proc)
                active_state = int(state_path[-1])

                yield StreamingAlert(
                    sample_index=i,
                    timestamp_sec=float(timestamps_sec[i]),
                    window_log_likelihood=float(raw_ll),
                    normalized_log_likelihood=float(norm_ll),
                    threshold=float(self.threshold_),
                    is_anomaly=bool(is_anom),
                    is_persisted_alarm=bool(is_persisted),
                    active_regime=active_state
                )

    def save(self, filepath: str):
        """Saves detector model and calibration parameters to disk."""
        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "threshold": self.threshold_,
            "threshold_method": self.threshold_method_,
            "calibrated_window_size": self.calibrated_window_size_,
            "train_window_scores": self.train_window_scores_
        }
        joblib.dump(payload, filepath)

    @classmethod
    def load(cls, filepath: str) -> "HMMAnomalyDetector":
        """Loads saved detector from disk."""
        payload = joblib.load(filepath)
        detector = cls(
            n_components=payload["model"].n_components,
            covariance_type=payload["model"].covariance_type
        )
        detector.model = payload["model"]
        detector.scaler = payload["scaler"]
        detector.threshold_ = payload["threshold"]
        detector.threshold_method_ = payload["threshold_method"]
        detector.calibrated_window_size_ = payload["calibrated_window_size"]
        detector.train_window_scores_ = payload["train_window_scores"]
        detector.is_fitted = True
        return detector
