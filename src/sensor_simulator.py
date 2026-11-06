"""
Industrial Sensor Simulator for Hidden Markov Model Anomaly Detection.

Simulates a rotating industrial machine (e.g., CNC milling spindle, turbine, or pump)
with multi-modal operational regimes (Markov states) and realistic anomalies:
- Normal regimes: Idle/Standby, Normal Steady-State, Peak Machining.
- Anomaly types:
  1. Vibration resonance / harmonic chatter (subtle pattern anomaly)
  2. Thermal degradation / friction drift (wear-out)
  3. Impact transient / shock spikes (point anomalies)
  4. Stuck-at / flatline sensor failure
  5. Regime sequence violation / rapid state chatter
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AnomalyEvent:
    start_idx: int
    end_idx: int
    anomaly_type: str
    description: str


class MachineSensorSimulator:
    """
    Simulates sequential multivariate sensor readings from industrial machinery.
    Signals simulated:
      - vibration: Acceleration in g (primary dynamic metric)
      - temperature: Spindle bearing temperature in °C
      - rpm: Spindle rotational speed in RPM
    """

    def __init__(self, random_seed: int = 42, sampling_rate_hz: float = 10.0):
        self.seed = random_seed
        self.fs = sampling_rate_hz
        self.rng = np.random.RandomState(random_seed)

        # Operational regimes: 0: Idle, 1: Steady-State, 2: High-Load
        self.n_regimes = 3

        # State transition probability matrix for normal operation (sticky transitions)
        self.transition_matrix = np.array([
            [0.985, 0.015, 0.000],  # From Idle: stays idle or ramps to Steady
            [0.005, 0.985, 0.010],  # From Steady: occasional idle or ramps to High-Load
            [0.000, 0.020, 0.980],  # From High-Load: steps down to Steady
        ])

        # Emission distributions (Mean vectors [vibration, temperature, rpm])
        self.regime_means = np.array([
            [0.35, 38.0, 800.0],    # Idle: low vibration, ambient temp, idle RPM
            [1.20, 62.0, 2400.0],   # Steady: nominal vibration, running temp, rated RPM
            [2.40, 82.0, 3600.0],   # High-Load: higher vibration, elevated temp, high RPM
        ])

        # Emission standard deviations per regime
        self.regime_stds = np.array([
            [0.08, 1.2, 35.0],
            [0.15, 2.0, 50.0],
            [0.25, 2.5, 70.0],
        ])

    def generate_normal_stream(self, n_samples: int) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generates normal operational time series following Markov regime transitions.

        Returns:
            df: DataFrame with timestamps, sensor columns, and regime labels.
            hidden_states: 1D array of ground truth operational regime IDs.
        """
        states = np.zeros(n_samples, dtype=int)
        current_state = 0  # start at idle

        for t in range(n_samples):
            states[t] = current_state
            # Transition to next state
            current_state = self.rng.choice(
                self.n_regimes,
                p=self.transition_matrix[current_state]
            )

        # Generate emissions with temporal smoothing
        vibration = np.zeros(n_samples)
        temperature = np.zeros(n_samples)
        rpm = np.zeros(n_samples)

        # Base autoregressive noise for realistic temporal continuity
        phi = 0.65  # AR(1) coefficient for temporal autocorrelation

        prev_v_noise = 0.0
        prev_t_noise = 0.0
        prev_r_noise = 0.0

        for t in range(n_samples):
            s = states[t]
            mu = self.regime_means[s]
            sigma = self.regime_stds[s]

            # Innovation noise
            eps_v = self.rng.normal(0, sigma[0] * np.sqrt(1 - phi**2))
            eps_t = self.rng.normal(0, sigma[1] * np.sqrt(1 - phi**2))
            eps_r = self.rng.normal(0, sigma[2] * np.sqrt(1 - phi**2))

            prev_v_noise = phi * prev_v_noise + eps_v
            prev_t_noise = phi * prev_t_noise + eps_t
            prev_r_noise = phi * prev_r_noise + eps_r

            # Periodic micro-vibration from shaft rotation (sinusoidal component)
            shaft_freq = (mu[2] / 60.0)  # Hz
            t_sec = t / self.fs
            rotation_wave = 0.06 * np.sin(2 * np.pi * shaft_freq * t_sec)

            vibration[t] = mu[0] + prev_v_noise + rotation_wave
            temperature[t] = mu[1] + prev_t_noise
            rpm[t] = mu[2] + prev_r_noise

        time_sec = np.arange(n_samples) / self.fs

        df = pd.DataFrame({
            "timestamp_sec": time_sec,
            "vibration": vibration,
            "temperature": temperature,
            "rpm": rpm,
            "true_regime": states,
            "is_anomaly": np.zeros(n_samples, dtype=int),
            "anomaly_type": ["none"] * n_samples
        })

        return df, states

    def generate_test_stream_with_anomalies(
        self,
        n_samples: int = 8000
    ) -> Tuple[pd.DataFrame, List[AnomalyEvent]]:
        """
        Generates a continuous test stream seeded with diverse real-world anomalies.
        """
        df, states = self.generate_normal_stream(n_samples)
        anomalies: List[AnomalyEvent] = []

        vibration = df["vibration"].values.copy()
        temperature = df["temperature"].values.copy()
        rpm = df["rpm"].values.copy()
        is_anomaly = np.zeros(n_samples, dtype=int)
        anomaly_type_col = ["none"] * n_samples

        # Helper to mark ground truth
        def apply_event(start: int, end: int, atype: str, desc: str):
            is_anomaly[start:end] = 1
            for k in range(start, end):
                anomaly_type_col[k] = atype
            anomalies.append(AnomalyEvent(start, end, atype, desc))

        # -------------------------------------------------------------
        # Anomaly 1: High-Frequency Resonance / Bearing Chatter (Indices 1200 - 1500)
        # Severe harmonic chatter at 3x frequency while RPM and Temp look superficially normal
        # -------------------------------------------------------------
        a1_start, a1_end = int(0.15 * n_samples), int(0.19 * n_samples)
        t_span = np.arange(a1_end - a1_start) / self.fs
        chatter = 1.1 * np.sin(2 * np.pi * 18.5 * t_span) + self.rng.normal(0, 0.35, a1_end - a1_start)
        vibration[a1_start:a1_end] += chatter
        apply_event(
            a1_start, a1_end, "bearing_chatter",
            "High-frequency harmonic resonance indicating bearing cage failure"
        )

        # -------------------------------------------------------------
        # Anomaly 2: Thermal Friction Drift & Wear (Indices 2600 - 3100)
        # Slow creeping temperature rise + slight friction vibration increase
        # -------------------------------------------------------------
        a2_start, a2_end = int(0.325 * n_samples), int(0.39 * n_samples)
        span_len = a2_end - a2_start
        ramp = np.linspace(0.0, 28.0, span_len)  # Temp ramps up by 28°C
        vib_ramp = np.linspace(0.0, 0.9, span_len)
        temperature[a2_start:a2_end] += ramp
        vibration[a2_start:a2_end] += vib_ramp
        apply_event(
            a2_start, a2_end, "thermal_friction_drift",
            "Continuous thermal runaway due to severe lubrication breakdown"
        )

        # -------------------------------------------------------------
        # Anomaly 3: Mechanical Shock / Transient Point Spikes (Indices 4200 - 4350)
        # Repeated explosive transient impact shocks (tool chipping)
        # -------------------------------------------------------------
        a3_start, a3_end = int(0.525 * n_samples), int(0.545 * n_samples)
        for spike_idx in range(a3_start, a3_end, 15):
            spike_w = 4
            vibration[spike_idx:spike_idx + spike_w] += self.rng.uniform(3.5, 6.0)
            rpm[spike_idx:spike_idx + spike_w] -= self.rng.uniform(300.0, 600.0)
        apply_event(
            a3_start, a3_end, "transient_impact_spikes",
            "Periodic high-amplitude impact spikes from broken cutter inserts"
        )

        # -------------------------------------------------------------
        # Anomaly 4: Sensor Freezing / Flatline Failure (Indices 5500 - 5800)
        # Sensor freezes at constant dead value with zero natural variance
        # -------------------------------------------------------------
        a4_start, a4_end = int(0.687 * n_samples), int(0.725 * n_samples)
        vibration[a4_start:a4_end] = 0.05
        rpm[a4_start:a4_end] = 1800.0  # static frozen value
        # slight noise only
        temperature[a4_start:a4_end] = temperature[a4_start]
        apply_event(
            a4_start, a4_end, "sensor_freeze",
            "Sensor telemetry frozen due to bus communication failure"
        )

        # -------------------------------------------------------------
        # Anomaly 5: Regime Sequence Violation / Chatter (Indices 6800 - 7200)
        # Machine abnormally switches between Idle and Peak every 3 steps (impossible under normal physics)
        # -------------------------------------------------------------
        a5_start, a5_end = int(0.85 * n_samples), int(0.90 * n_samples)
        for k in range(a5_start, a5_end):
            flip = 0 if (k // 4) % 2 == 0 else 2
            vibration[k] = self.regime_means[flip, 0] + self.rng.normal(0, 0.1)
            rpm[k] = self.regime_means[flip, 2] + self.rng.normal(0, 40)
        apply_event(
            a5_start, a5_end, "regime_sequence_violation",
            "Rapid abnormal hunting/chatter between idle and peak load regimes"
        )

        df["vibration"] = vibration
        df["temperature"] = temperature
        df["rpm"] = rpm
        df["is_anomaly"] = is_anomaly
        df["anomaly_type"] = anomaly_type_col

        return df, anomalies
