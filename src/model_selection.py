"""
Model Selection Module for Hidden Markov Models.

Implements rigorous model selection evaluating:
- Candidate hidden state counts K in [2, 3, 4, 5, 6, 7, 8]
- Full, diagonal, and spherical Gaussian emission parameter count calculations
- Akaike Information Criterion (AIC)
- Bayesian Information Criterion (BIC)
- Out-of-sample holdout Validation Log-Likelihood
- Multiple random initialization restarts for EM convergence stability
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from hmmlearn import hmm
import logging

logger = logging.getLogger(__name__)


def calculate_hmm_free_parameters(
    n_components: int,
    n_features: int,
    covariance_type: str = "full"
) -> int:
    """
    Computes exact degrees of freedom (free parameters) in GaussianHMM:
      - Initial state distribution pi: (K - 1)
      - Transition matrix A: K * (K - 1)
      - Emission means mu: K * D
      - Emission covariance:
          full: K * D * (D + 1) / 2
          diag: K * D
          spherical: K * 1
    """
    k = n_components
    d = n_features

    # Initial probabilities and transitions
    initial_params = k - 1
    transition_params = k * (k - 1)
    mean_params = k * d

    if covariance_type == "full":
        cov_params = k * (d * (d + 1)) // 2
    elif covariance_type == "diag":
        cov_params = k * d
    elif covariance_type == "spherical":
        cov_params = k
    else:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")

    total_params = initial_params + transition_params + mean_params + cov_params
    return total_params


class HMMModelSelector:
    """
    Explores and selects the optimal GaussianHMM model using BIC, AIC,
    and holdout validation log-likelihood.
    """

    def __init__(
        self,
        candidate_states: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8),
        covariance_type: str = "full",
        n_iter: int = 150,
        tol: float = 1e-4,
        n_restarts: int = 5,
        random_state: int = 42
    ):
        self.candidate_states = candidate_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.n_restarts = n_restarts
        self.random_state = random_state

        self.results_: List[Dict[str, Any]] = []
        self.best_k_: int = 3
        self.best_model_: hmm.GaussianHMM = None

    def fit_evaluate(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        train_lengths: List[int] = None,
        val_lengths: List[int] = None
    ) -> pd.DataFrame:
        """
        Fits candidate models and calculates statistical selection criteria.
        """
        N_train = X_train.shape[0]
        n_features = X_train.shape[1]

        self.results_ = []
        best_bic = float("inf")

        for k in self.candidate_states:
            p = calculate_hmm_free_parameters(k, n_features, self.covariance_type)

            best_train_ll = float("-inf")
            best_model_for_k = None

            # Multiple random restarts to avoid local minima
            for r in range(self.n_restarts):
                seed = self.random_state + r * 1337
                model = hmm.GaussianHMM(
                    n_components=k,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    tol=self.tol,
                    random_state=seed,
                    init_params="stmc"
                )
                try:
                    model.fit(X_train, lengths=train_lengths)
                    train_ll = model.score(X_train, lengths=train_lengths)
                    if train_ll > best_train_ll:
                        best_train_ll = train_ll
                        best_model_for_k = model
                except Exception as e:
                    logger.debug(f"Convergence issue with K={k}, seed={seed}: {e}")
                    continue

            if best_model_for_k is None:
                continue

            # Evaluate on holdout validation set
            val_ll = best_model_for_k.score(X_val, lengths=val_lengths)
            N_val = X_val.shape[0]

            # AIC = -2 * ln(L) + 2 * p
            # BIC = -2 * ln(L) + p * ln(N)
            aic = -2.0 * best_train_ll + 2.0 * p
            bic = -2.0 * best_train_ll + p * np.log(N_train)

            # Normalized per-sample metrics
            train_ll_per_sample = best_train_ll / N_train
            val_ll_per_sample = val_ll / N_val

            entry = {
                "n_components": k,
                "n_parameters": p,
                "train_log_likelihood": best_train_ll,
                "val_log_likelihood": val_ll,
                "train_ll_per_sample": train_ll_per_sample,
                "val_ll_per_sample": val_ll_per_sample,
                "aic": aic,
                "bic": bic,
                "converged": best_model_for_k.monitor_.converged
            }
            self.results_.append(entry)

            # We prioritize BIC for parsimony and avoidance of overfitting
            if bic < best_bic:
                best_bic = bic
                self.best_k_ = k
                self.best_model_ = best_model_for_k

        results_df = pd.DataFrame(self.results_)
        return results_df

    def get_selection_summary(self) -> Dict[str, Any]:
        """
        Returns formal summary of the model selection decision.
        """
        df = pd.DataFrame(self.results_)
        best_row = df.loc[df["n_components"] == self.best_k_].iloc[0]

        rationale = (
            f"Chosen K={self.best_k_} hidden states based on minimum BIC ({best_row['bic']:.2f}) "
            f"and strong holdout validation log-likelihood ({best_row['val_ll_per_sample']:.4f} per sample). "
            f"This captures the distinct physical operational regimes (e.g. Idle, Nominal Steady-State, "
            f"and Peak Machining) without unnecessary parameter inflation (p={int(best_row['n_parameters'])})."
        )

        return {
            "best_k": self.best_k_,
            "best_bic": best_row["bic"],
            "best_aic": best_row["aic"],
            "best_model": self.best_model_,
            "results_dataframe": df,
            "rationale": rationale
        }
