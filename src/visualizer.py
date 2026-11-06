"""
Visualization Module for HMM Anomaly Detection.

Produces:
1. Multi-panel real-time detection dashboard overlaying raw sensor readings,
   sliding-window log-likelihood, calibrated threshold, and Viterbi state tracking.
2. Model selection diagnostics (BIC, AIC, validation log-likelihood vs K).
3. Training likelihood distribution and threshold calibration curves.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import List, Optional
from pathlib import Path
from src.sensor_simulator import AnomalyEvent


class AnomalyVisualizer:
    """
    Renders rich diagnostic and operational plots for HMM Anomaly Detection.
    """

    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Professional clean styling
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def plot_detection_dashboard(
        self,
        test_df: pd.DataFrame,
        window_indices: np.ndarray,
        window_scores: np.ndarray,
        threshold: float,
        detected_mask: np.ndarray,
        hidden_states: Optional[np.ndarray] = None,
        anomalies: Optional[List[AnomalyEvent]] = None,
        filename: str = "hmm_anomaly_detection_dashboard.png"
    ) -> str:
        """
        Creates a multi-panel telemetry plot overlaying raw sensor signals with
        detected anomaly regions and log-likelihood curves.
        """
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True, gridspec_kw={"height_ratios": [2.5, 2.0, 2.0, 1.2]})

        t = test_df["timestamp_sec"].values
        vibration = test_df["vibration"].values
        temperature = test_df["temperature"].values
        rpm = test_df["rpm"].values

        t_windows = t[window_indices]

        # -------------------------------------------------------------
        # Subplot 1: Primary Sensor (Vibration) with Ground Truth & Detections
        # -------------------------------------------------------------
        ax1 = axes[0]
        ax1.plot(t, vibration, color="#1f77b4", linewidth=1.1, label="Vibration (g) [Raw Sensor]", alpha=0.85)

        # Shade ground truth anomaly periods in light red
        if anomalies:
            for ev in anomalies:
                ax1.axvspan(
                    t[ev.start_idx], t[ev.end_idx - 1],
                    color="#ff9999", alpha=0.35,
                    label="Ground Truth Anomaly" if ev == anomalies[0] else ""
                )
                # Label anomaly type above
                mid_t = (t[ev.start_idx] + t[ev.end_idx - 1]) / 2.0
                ax1.text(
                    mid_t, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 5.0,
                    ev.anomaly_type.replace("_", " ").title(),
                    rotation=30, ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color="#800000"
                )

        # Highlight detected anomalous time steps with red markers
        det_t = t_windows[detected_mask]
        det_vib = vibration[window_indices[detected_mask]]
        ax1.scatter(det_t, det_vib, color="#d62728", s=18, zorder=5, label="Flagged Anomaly Windows")

        ax1.set_ylabel("Vibration (g)", fontsize=11, fontweight="bold")
        ax1.set_title("Real-Time Machine Telemetry with HMM Anomaly Detection", fontsize=14, fontweight="bold", pad=12)
        ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)
        ax1.grid(True, linestyle="--", alpha=0.6)

        # -------------------------------------------------------------
        # Subplot 2: Secondary Sensors (Temperature and RPM)
        # -------------------------------------------------------------
        ax2 = axes[1]
        color_temp = "#d95f02"
        color_rpm = "#7570b3"

        ax2.plot(t, temperature, color=color_temp, linewidth=1.2, label="Spindle Temp (°C)")
        ax2.set_ylabel("Temperature (°C)", color=color_temp, fontsize=11, fontweight="bold")
        ax2.tick_params(axis="y", labelcolor=color_temp)

        ax2_twin = ax2.twinx()
        ax2_twin.plot(t, rpm, color=color_rpm, linewidth=1.1, linestyle="--", label="Rotational Speed (RPM)", alpha=0.8)
        ax2_twin.set_ylabel("Speed (RPM)", color=color_rpm, fontsize=11, fontweight="bold")
        ax2_twin.tick_params(axis="y", labelcolor=color_rpm)

        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9, fontsize=9)
        ax2.grid(True, linestyle="--", alpha=0.6)

        # -------------------------------------------------------------
        # Subplot 3: Sliding Window Log-Likelihood & Threshold
        # -------------------------------------------------------------
        ax3 = axes[2]
        ax3.plot(t_windows, window_scores, color="#2ca02c", linewidth=1.2, label="Sliding Window Log-Likelihood (per step)")
        ax3.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.8, label=f"Anomaly Threshold ({threshold:.2f})")

        # Fill regions below threshold in soft red
        ax3.fill_between(
            t_windows, window_scores, threshold,
            where=(window_scores < threshold),
            color="#ff4d4d", alpha=0.35, label="Anomaly Region (LL < Threshold)"
        )

        ax3.set_ylabel("Avg Log-Likelihood", fontsize=11, fontweight="bold")
        ax3.legend(loc="lower right", framealpha=0.9, fontsize=9)
        ax3.grid(True, linestyle="--", alpha=0.6)

        # -------------------------------------------------------------
        # Subplot 4: Inferred Viterbi Hidden States (Regime Tracking)
        # -------------------------------------------------------------
        ax4 = axes[3]
        if hidden_states is not None:
            ax4.step(t, hidden_states, color="#4b0082", where="mid", linewidth=1.3, label="Decoded State")
            ax4.set_yticks([0, 1, 2])
            ax4.set_yticklabels(["Idle", "Steady", "High-Load"])
        ax4.set_ylabel("Hidden State", fontsize=11, fontweight="bold")
        ax4.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
        ax4.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return str(filepath)

    def plot_model_selection(
        self,
        selection_df: pd.DataFrame,
        filename: str = "hmm_model_selection_bic_aic.png"
    ) -> str:
        """
        Plots BIC, AIC, and holdout validation log-likelihood curves vs K.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        k_vals = selection_df["n_components"].values
        bic_vals = selection_df["bic"].values
        aic_vals = selection_df["aic"].values
        val_ll = selection_df["val_ll_per_sample"].values

        # Best K by BIC
        best_k_bic = selection_df.loc[selection_df["bic"].idxmin(), "n_components"]

        # Panel 1: Information Criteria (BIC & AIC)
        ax1.plot(k_vals, bic_vals, marker="o", linewidth=2.0, color="#1f77b4", label="BIC (Bayesian Information Criterion)")
        ax1.plot(k_vals, aic_vals, marker="s", linewidth=2.0, linestyle="--", color="#ff7f0e", label="AIC (Akaike Information Criterion)")
        ax1.axvline(best_k_bic, color="#2ca02c", linestyle=":", linewidth=2, label=f"Optimal K = {best_k_bic}")
        ax1.scatter([best_k_bic], [selection_df.loc[selection_df["bic"].idxmin(), "bic"]], color="#d62728", s=100, zorder=5)

        ax1.set_xlabel("Number of Hidden States (K)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Information Criterion Score (Lower is Better)", fontsize=11, fontweight="bold")
        ax1.set_title("HMM Model Selection: Information Criteria vs K", fontsize=12, fontweight="bold")
        ax1.set_xticks(k_vals)
        ax1.legend(loc="upper right", framealpha=0.9)
        ax1.grid(True, linestyle="--", alpha=0.6)

        # Panel 2: Validation Log-Likelihood per Sample
        ax2.plot(k_vals, val_ll, marker="^", linewidth=2.0, color="#2ca02c", label="Holdout Validation Log-Likelihood")
        ax2.axvline(best_k_bic, color="#2ca02c", linestyle=":", linewidth=2, label=f"Selected K = {best_k_bic}")
        ax2.set_xlabel("Number of Hidden States (K)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Validation Avg Log-Likelihood (Higher is Better)", fontsize=11, fontweight="bold")
        ax2.set_title("Generalization: Validation Log-Likelihood vs K", fontsize=12, fontweight="bold")
        ax2.set_xticks(k_vals)
        ax2.legend(loc="lower right", framealpha=0.9)
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return str(filepath)

    def plot_likelihood_distribution_and_threshold(
        self,
        train_scores: np.ndarray,
        threshold: float,
        threshold_method: str = "percentile",
        percentile_val: float = 1.0,
        filename: str = "hmm_likelihood_distribution_threshold.png"
    ) -> str:
        """
        Plots the histogram and empirical density of normal training window likelihoods
        and indicates the calibrated threshold boundary.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram of training scores
        counts, bins, patches = ax.hist(
            train_scores, bins=50, density=True,
            color="#aec7e8", edgecolor="#1f77b4", alpha=0.75,
            label="Normal Training Windows Distribution"
        )

        # Mark threshold line
        ax.axvline(
            threshold, color="#d62728", linestyle="--", linewidth=2.2,
            label=f"Decision Threshold ({threshold:.2f}, {threshold_method}={percentile_val}%)"
        )

        # Highlight anomaly reject region
        x_fill = np.linspace(min(bins[0], threshold - 0.5), threshold, 100)
        ax.fill_betweenx(
            [0, max(counts) * 1.1], min(bins[0], threshold - 0.5), threshold,
            color="#ff9999", alpha=0.35, label="Flagged Anomaly Region (Likelihood < Threshold)"
        )

        mu = np.mean(train_scores)
        std = np.std(train_scores)

        ax.set_title(
            f"Training Data Log-Likelihood Distribution & Calibrated Anomaly Threshold\n"
            f"(Mean = {mu:.2f}, Std = {std:.2f})",
            fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Normalized Log-Likelihood per Time Step", fontsize=11, fontweight="bold")
        ax.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", framealpha=0.9, fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return str(filepath)
