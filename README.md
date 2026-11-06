# Real-Time Anomaly Detection with Hidden Markov Models (HMM)

An end-to-end probabilistic anomaly detection system for multivariate industrial machine sensor telemetry using Hidden Markov Models with Gaussian emissions (`hmmlearn`).

---

## 1. System Overview & Problem Formulation

In modern manufacturing and industrial IoT, unexpected machine degradation (e.g., spindle bearing wear, tool chatter, lubrication failure) causes catastrophic downtime. Traditional univariate thresholding (such as upper/lower tolerance limits on temperature or vibration) fails to detect **subtle temporal pattern anomalies**, **harmonic resonances**, or **abnormal regime transitions** where individual readings may remain within nominal static bounds.

Hidden Markov Models (HMMs) model both:
1. **Continuous Multivariate Emissions**: The probabilistic distribution of physical sensor signals ($\text{vibration } [g]$, $\text{temperature } [^\circ\text{C}]$, $\text{rotational speed } [\text{RPM}]$) within discrete operational regimes.
2. **Discrete Temporal Transitions**: The Markovian transition probabilities governing switches between machine states (e.g., Idle $\rightarrow$ Nominal Steady-State $\rightarrow$ Peak Machining).

```
          [ Hidden State s_t ] ------------> [ Hidden State s_{t+1} ]
                   |                                     |
                   v                                     v
       [ Emission x_t ~ N(μ, Σ) ]           [ Emission x_{t+1} ~ N(μ, Σ) ]
```

---

## 2. Theoretical Foundations

### 2.1 Gaussian Hidden Markov Model Definition
An HMM $\lambda = (A, B, \pi)$ is parameterized by:
- **Initial State Distribution**: $\pi = [\pi_i]$, where $\pi_i = P(s_1 = i)$ for $i \in \{1, \dots, K\}$.
- **Transition Probability Matrix**: $A = [a_{ij}]$, where $a_{ij} = P(s_{t+1} = j \mid s_t = i)$.
- **Gaussian Emission Probabilities**: $B = \{b_j(x_t)\}$, where each hidden state $j$ emits observations following a multivariate normal distribution:
  $$b_j(x_t) = \mathcal{N}(x_t \mid \mu_j, \Sigma_j) = \frac{1}{(2\pi)^{D/2} |\Sigma_j|^{1/2}} \exp\left( -\frac{1}{2} (x_t - \mu_j)^\top \Sigma_j^{-1} (x_t - \mu_j) \right)$$

### 2.2 Sequence Log-Likelihood via Forward Algorithm
The log-likelihood of a sequence $X = (x_1, \dots, x_T)$ is efficiently computed using the forward variable $\alpha_t(i) = P(x_1, \dots, x_t, s_t = i \mid \lambda)$:
$$\alpha_1(i) = \pi_i b_i(x_1)$$
$$\alpha_{t+1}(j) = \left[ \sum_{i=1}^K \alpha_t(i) a_{ij} \right] b_j(x_{t+1})$$
$$P(X \mid \lambda) = \sum_{i=1}^K \alpha_T(i), \quad \ln L(X) = \ln P(X \mid \lambda)$$

### 2.3 Normalized Sliding Window Score
To make likelihood scores comparable across arbitrary window lengths $W$, we compute the average log-likelihood per time-step:
$$\ell_{\text{norm}}(X_{t:t+W}) = \frac{1}{W} \ln P(X_{t:t+W} \mid \lambda)$$

---

## 3. Model Selection Architecture & Rationale

Selecting the optimal number of hidden states $K$ balances model expressiveness against overfitting. For an HMM with full covariance matrices, the total number of free parameters $p$ is:
$$p = \underbrace{(K - 1)}_{\pi} + \underbrace{K(K - 1)}_{A} + \underbrace{K \cdot D}_{\mu} + \underbrace{K \cdot \frac{D(D + 1)}{2}}_{\Sigma} = K^2 - 1 + K \cdot D + K \cdot \frac{D(D + 1)}{2}$$

We evaluate candidate models across $K \in \{2, 3, 4, 5, 6, 7, 8\}$ using:
- **Bayesian Information Criterion (BIC)**: $\text{BIC} = -2 \ln L + p \ln(N)$
- **Akaike Information Criterion (AIC)**: $\text{AIC} = -2 \ln L + 2p$
- **Holdout Validation Log-Likelihood**: Generalization performance on an unseen normal operating sequence.

### Empirical Model Selection Sweep Table ($N = 6,000$ training samples, $D = 3$ features)

| Hidden States ($K$) | Free Parameters ($p$) | Training Log-Likelihood | Validation Log-Likelihood | AIC | BIC | Val Log-Likelihood / step |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 21 | -54,695.88 | -17,662.56 | 109,433.77 | 109,574.46 | -8.8313 |
| 3 | 35 | -42,750.44 | -14,005.20 | 85,570.88 | 85,805.36 | -7.0026 |
| 4 | 51 | -42,417.13 | -13,924.41 | 84,936.26 | 85,277.93 | -6.9622 |
| 5 | 69 | -41,982.51 | -13,761.53 | 84,103.02 | 84,565.29 | -6.8808 |
| 6 | 89 | -41,872.45 | -13,778.86 | 83,922.89 | 84,519.15 | -6.8894 |
| 7 | 111 | -41,815.01 | -13,756.19 | 83,852.02 | 84,595.66 | -6.8781 |
| **8** | **135** | **-41,642.60** | **-13,697.21** | **83,555.19** | **84,459.63** | **-6.8486** |

**Selection Rationale**:
Models with $K \ge 3$ capture the primary physical operational regimes (Idle, Steady-State, Peak Machining), producing an immediate jump of over 12,000 nats in log-likelihood. While $K=3$ captures the coarse regimes, $K=8$ achieves the lowest BIC score ($84,459.63$) and highest validation log-likelihood ($-6.8486$ nats/sample) by resolving subtle intra-regime load variations and shaft micro-dynamics without overparameterizing.

---

## 4. Anomaly Threshold Calibration

The anomaly threshold $\tau$ is calibrated strictly on normal operating sequences:
1. All training windows of length $W = 30$ samples ($3.0$ seconds at $10\text{ Hz}$) are scored.
2. The empirical distribution of normal log-likelihoods has mean $\mu_{\ell} = -69.53$ and standard deviation $\sigma_{\ell} = 59.01$.
3. We set the threshold using the **1st percentile ($\alpha = 1.0\%$)** of the training distribution:
   $$\tau = \text{Quantile}_{0.01}(\{\ell_{\text{norm}, i}\}) = -159.37$$
4. An observation window is flagged as anomalous if:
   $$\ell_{\text{norm}}(X_{t:t+W}) < \tau$$

### Threshold Sensitivity Trade-off Sweep

| Target Percentile (%) | Calibrated Threshold $\tau$ | Precision | Recall | F1-Score | False Positive Rate |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.1% | -165.321 | **0.934** | 0.560 | 0.700 | **0.0107** |
| 0.5% | -160.623 | 0.889 | 0.597 | **0.715** | 0.0203 |
| **1.0% (Default)** | **-159.372** | **0.867** | **0.605** | **0.713** | **0.0252** |
| 2.0% | -156.792 | 0.799 | 0.622 | 0.700 | 0.0424 |
| 5.0% | -153.403 | 0.709 | 0.640 | 0.673 | 0.0715 |
| 10.0% | -150.365 | 0.609 | **0.656** | 0.632 | 0.1146 |

---

## 5. Real-Time Streaming Architecture

The real-time streaming pipeline operates tick-by-tick with a rolling FIFO buffer:

```
 Incoming Sensor Tick (x_t)
            │
            ▼
 ┌──────────────────────┐
 │   FIFO Ring Buffer   │  (Size W = 30)
 └──────────┬───────────┘
            │ Window Full
            ▼
 ┌──────────────────────┐
 │  Feature Scaler Z(X) │  (Standardized using training μ, σ)
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │ Forward Algorithm LL │  Score: l_norm = (1/W) * ln P(W | HMM)
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │ Threshold Comparator │  l_norm < τ ?
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │  Persistence Filter  │  Requires N_persist consecutive violations (e.g. 3)
 └──────────┬───────────┘
            │
            ▼
    EMIT ALARM EVENT
```

---

## 6. Experimental Results & Verification

### 6.1 Sample-Level Classification Metrics

- **Precision**: **86.71%** (1,031 true positive alarm windows vs 158 false positive windows)
- **Specificity**: **97.48%** (Low false alarm rate on normal operating periods)
- **ROC-AUC**: **0.8637**
- **PR-AUC**: **0.7629**
- **F1-Score**: **0.7128**

### 6.2 Event-Level Detection & Latency Benchmark

The test stream was injected with 5 realistic industrial machine fault classes. **The detector achieved 100% event-level detection rate (5 / 5 detected)**:

| Fault Classification | Anomaly Interval (Samples) | Duration (sec) | Detected? | Time-to-Detect Latency | Fault Description |
|:---|:---:|:---:|:---:|:---:|:---|
| **Bearing Chatter** | 1200 – 1520 | 32.0s | **YES** | **9.80s** | High-frequency harmonic vibration resonance |
| **Thermal Drift** | 2600 – 3120 | 52.0s | **YES** | **5.40s** | Thermal runaway due to lubrication failure |
| **Impact Spikes** | 4200 – 4360 | 16.0s | **YES** | **0.20s** | High-amplitude cutter insert impact fracture |
| **Sensor Freeze** | 5496 – 5800 | 30.4s | **YES** | **0.00s** | Telemetry bus failure / zero-variance flatline |
| **Regime Chatter** | 6800 – 7200 | 40.0s | **YES** | **0.10s** | Abnormal rapid hunting between idle and peak |

---

## 7. Visualizations

The pipeline generates three publication-quality diagnostic figures in `visualizations/`:
1. `hmm_anomaly_detection_dashboard.png`: 4-panel dashboard showing raw vibration, temperature & RPM telemetry, sliding window log-likelihood vs threshold, and inferred Viterbi hidden states.
2. `hmm_model_selection_bic_aic.png`: BIC, AIC, and holdout validation log-likelihood curves across $K \in [2, 8]$.
3. `hmm_likelihood_distribution_threshold.png`: Normal training likelihood histogram and density with calibrated decision cutoff.

---

## 8. Repository Structure

```
.
├── config.py                  # Global simulation, model, and detector configuration
├── main.py                    # End-to-end executable pipeline
├── requirements.txt           # Python package dependencies
├── README.md                  # Comprehensive system documentation
├── data/
│   ├── generate_data.py       # Data generator utility
│   ├── normal_sensor_train.csv # 6,000 normal training samples (10 Hz)
│   ├── normal_sensor_val.csv   # 2,000 normal validation samples (10 Hz)
│   ├── test_sensor_stream.csv  # 8,000 continuous test samples with injected anomalies
│   └── evaluation_results.json # Serialized evaluation metrics & latencies
├── models/
│   └── hmm_detector.joblib    # Serialized trained HMM detector & scaler
├── src/
│   ├── __init__.py
│   ├── sensor_simulator.py    # Industrial multi-sensor Markov machine simulator
│   ├── model_selection.py     # BIC/AIC and validation log-likelihood model selector
│   ├── anomaly_detector.py    # Core HMM detector with real-time streaming generator
│   ├── evaluator.py           # Evaluation metrics, latency calculator & threshold sweeper
│   └── visualizer.py          # Multi-panel visualization generator
├── tests/
│   ├── __init__.py
│   └── test_detector.py       # Comprehensive pytest unit & integration test suite
└── visualizations/
    ├── hmm_anomaly_detection_dashboard.png
    ├── hmm_likelihood_distribution_threshold.png
    └── hmm_model_selection_bic_aic.png
```

---

## 9. How to Run

### Step 1: Run Unit Tests
```bash
python -m pytest tests -v
```

### Step 2: Run End-to-End Pipeline
```bash
python main.py
```
This generates data (if missing), runs the model selection sweep, fits the selected HMM, calibrates the threshold, processes the streaming test data, prints evaluation tables, and exports visualization dashboards.
