"""
Model definitions: Bayesian logistic regression (Laplace approximation),
fairness metrics, and the Fair Active Learning engine.

Sensitive attribute: Age (Young < 35 -> Z=0 | Old >= 35 -> Z=1).
Methods: naive, alpha_aggregate, fal_nested, fal_nested_append.
See README.md for the full write-up.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, brier_score_loss,
)
from scipy.special import expit as sigmoid
from collections import defaultdict

SEED = 143
np.random.seed(SEED)

LABEL_MAP = {
    "naive":             "Naive (Uncertainty Only)",
    "alpha_aggregate":   "FAL α-Aggregate (Bayes)",
    "fal_nested":        "FAL-Nested (Bayes)",
    "fal_nested_append": "FAL-Nested-Append (Bayes)",
}

# ══════════════════════════════════════════════════════════════════════════════
# 1 ─ DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_credit_data(path: str = "cs-training.csv", n: int = 5000):
    """
    Load & preprocess Give Me Some Credit.
    Returns X (standardised), y (target), z (sensitive attr), feature_names.
    """
    df = pd.read_csv(path).dropna().head(n)
    drop_cols = [c for c in ["SeriousDlqin2yrs", "Unnamed: 0"] if c in df.columns]
    y = df["SeriousDlqin2yrs"].values.astype(int)
    z = (df["age"] >= 35).astype(int).values
    feat_df = df.drop(columns=drop_cols)
    feature_names = list(feat_df.columns)
    X = StandardScaler().fit_transform(feat_df.values.astype(float))
    return X, y, z, feature_names


# ══════════════════════════════════════════════════════════════════════════════
# 2 ─ BAYESIAN LOGISTIC REGRESSION  (Laplace Approximation)
# ══════════════════════════════════════════════════════════════════════════════

class BayesianLogisticRegression:
    """
    Bayesian Logistic Regression fitted via Laplace Approximation.

    Prior   : θ ~ N(0, (1/λ)·I)        (isotropic Gaussian)
    Posterior: p(θ|D) ≈ N(θ_MAP, Σ)
               Σ = (H_MAP + λ·I)^{-1}
               H_MAP = X^T diag(p(1-p)) X   (observed Fisher information)

    Posterior predictive (probit approximation, Mackay 1992):
        p(y=1 | x, D) = σ( κ(σ²_x) · θ_MAP · x )
        σ²_x = x^T Σ x
        κ(v)  = (1 + π·v/8)^{-½}

    Parameters
    ----------
    lam         : prior precision  (ridge strength, λ = 1/prior_variance)
    n_iter      : gradient-ascent MAP iterations
    lr          : MAP learning rate
    n_mc_samples: number of Monte-Carlo samples drawn from posterior for
                  fairness-gain estimation
    bgd_weight  : sample-weight multiplier for the minority group in MAP fit
    """

    def __init__(self, lam: float = 1.0, n_iter: int = 200,
                 lr: float = 0.05, n_mc_samples: int = 30,
                 bgd_weight: float = 5.0):
        self.lam          = lam
        self.n_iter       = n_iter
        self.lr           = lr
        self.n_mc_samples = n_mc_samples
        self.bgd_weight   = bgd_weight

        self.theta_map_: np.ndarray | None = None   # (d,)
        self.Sigma_: np.ndarray | None = None        # (d, d)
        self._fitted = False

    # ── MAP estimation (gradient ascent on log-posterior) ─────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray,
            z: np.ndarray | None = None) -> "BayesianLogisticRegression":
        """
        Fit MAP estimate and compute Laplace posterior covariance.

        Balance Gradient Descent is applied: minority samples (z==0) receive
        weight `bgd_weight`; majority receive weight 1.
        """
        n, d = X.shape

        # Sample weights (Balance GD)
        if z is not None:
            w = np.where(z == 0, self.bgd_weight, 1.0).astype(float)
        else:
            w = np.ones(n, dtype=float)
        w = w / w.mean()   # normalise so total weight ≈ n

        # Initialise θ
        theta = np.zeros(d)

        for _ in range(self.n_iter):
            logits = X @ theta                           # (n,)
            p      = sigmoid(logits)                     # (n,)
            # Weighted gradient of log-likelihood
            grad_ll = X.T @ (w * (y - p))               # (d,)
            # Gradient of log-prior  (Gaussian: -λ·θ)
            grad_prior = -self.lam * theta               # (d,)
            theta = theta + self.lr * (grad_ll + grad_prior)

        self.theta_map_ = theta

        # ── Laplace covariance  Σ = (H + λI)^{-1} ───────────────────────────
        p      = sigmoid(X @ theta)
        # Weighted Fisher: X^T diag(w·p·(1-p)) X
        R      = w * p * (1.0 - p)                      # (n,)
        H      = (X * R[:, None]).T @ X                 # (d, d)
        A      = H + self.lam * np.eye(d)               # posterior precision
        try:
            self.Sigma_ = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            self.Sigma_ = np.linalg.pinv(A)

        self._fitted = True
        return self

    # ── Posterior predictive mean  p(y=1|x,D) ────────────────────────────────

    def predict_proba_pp(self, X: np.ndarray) -> np.ndarray:
        """
        Posterior predictive probability via probit approximation.
        p(y=1|x,D) = σ( κ(σ²_x) · θ_MAP · x )
        Returns shape (n,).
        """
        if not self._fitted:
            return np.full(len(X), 0.5)
        mu     = X @ self.theta_map_                    # (n,)  MAP logit
        var_x  = np.einsum("nd,de,ne->n", X, self.Sigma_, X)  # x^T Σ x
        kappa  = 1.0 / np.sqrt(1.0 + np.pi * var_x / 8.0)
        return sigmoid(kappa * mu)                      # (n,)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns (n, 2) array compatible with sklearn convention."""
        p1 = self.predict_proba_pp(X)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba_pp(X) >= 0.5).astype(int)

    # ── Epistemic uncertainty  U(x) = H[p(y|x,D)] ───────────────────────────

    def epistemic_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Epistemic uncertainty as Shannon entropy of the posterior predictive.
        U(x) = -p·log(p) - (1-p)·log(1-p)
        where p = p(y=1|x,D)   (posterior predictive, not point estimate).
        Shape: (n,).
        """
        p  = self.predict_proba_pp(X)
        p  = np.clip(p, 1e-12, 1 - 1e-12)
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))

    # ── Posterior predictive SAMPLING (for fairness-gain MC) ─────────────────

    def sample_y_posterior_predictive(self, x: np.ndarray) -> np.ndarray:
        """
        Draw `n_mc_samples` label samples from p(y|x, D) by:
          1. Drawing θ ~ N(θ_MAP, Σ)   (posterior samples)
          2. Computing  p_k = σ(θ_k · x)
          3. Sampling   y_k ~ Bernoulli(p_k)

        This is exact Monte-Carlo integration over the posterior, avoiding
        the probit approximation for the fairness-gain calculation.
        Returns shape (n_mc_samples,) of 0/1 labels.
        """
        if not self._fitted:
            return (np.random.rand(self.n_mc_samples) > 0.5).astype(int)

        theta_samples = np.random.multivariate_normal(
            self.theta_map_, self.Sigma_, size=self.n_mc_samples
        )                                               # (S, d)
        logits = theta_samples @ x                      # (S,)
        probs  = sigmoid(logits)                        # (S,)
        return (np.random.rand(self.n_mc_samples) < probs).astype(int)
# ══════════════════════════════════════════════════════════════════════════════
# 3 ─ FAIRNESS METRICS
# ══════════════════════════════════════════════════════════════════════════════

def demographic_parity_gap(y_pred: np.ndarray, z: np.ndarray) -> float:
    """P(ŷ=1 | Z=0) – P(ŷ=1 | Z=1)"""
    rate0 = float(y_pred[z == 0].mean()) if (z == 0).any() else 0.0
    rate1 = float(y_pred[z == 1].mean()) if (z == 1).any() else 0.0
    return rate0 - rate1


def equalized_odds_gap(y_true: np.ndarray, y_pred: np.ndarray,
                       z: np.ndarray) -> dict:
    def tpr_fpr(mask):
        yt, yp = y_true[mask], y_pred[mask]
        if len(yt) == 0 or yt.sum() == 0:
            return 0.0, 0.0
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return tpr, fpr

    tpr0, fpr0 = tpr_fpr(z == 0)
    tpr1, fpr1 = tpr_fpr(z == 1)
    return {
        "tpr_gap": abs(tpr0 - tpr1),  "fpr_gap": abs(fpr0 - fpr1),
        "tpr_young": tpr0, "tpr_old": tpr1,
        "fpr_young": fpr0, "fpr_old": fpr1,
    }


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                                n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


# ══════════════════════════════════════════════════════════════════════════════
# 4 ─ BAYESIAN FAIRNESS GAIN
# ══════════════════════════════════════════════════════════════════════════════

def bayesian_fairness_gain(
        model: BayesianLogisticRegression,
        x_cand: np.ndarray,          # (d,) single candidate feature vector
        z_cand: int,                 # sensitive attribute of candidate
        X_labeled: np.ndarray,       # (m, d)
        y_labeled: np.ndarray,       # (m,)
        z_labeled: np.ndarray,       # (m,)
) -> float:
    """
    Bayesian Expected Fairness Gain of labeling candidate x_cand.

    ΔF(x*) = E_{y ~ p(y|x*,D)}[ |DP_gap(D ∪ {x*,y})| ] − |DP_gap(D)|

    A NEGATIVE value means labeling x* is expected to REDUCE the absolute
    demographic parity gap — i.e. it is a fairness-improving acquisition.

    Implementation
    ──────────────
    1. Compute current |DP_gap| on the existing labeled set.
    2. Draw S label samples y_s ~ p(y|x*, D)  via posterior predictive MC.
    3. For each sample:
         a. Temporarily augment the labeled set with (x*, y_s, z*).
         b. Predict on the augmented set with the current MAP weights.
         c. Compute |DP_gap| on the augmented predictions.
    4. Return mean( |DP_gap_aug_s| ) − |DP_gap_current|.
    """
    if not model._fitted:
        return 0.0

    # Current DP gap
    y_pred_now = model.predict(X_labeled)
    dp_now     = abs(demographic_parity_gap(y_pred_now, z_labeled))

    # Posterior predictive label samples for the candidate
    y_samples = model.sample_y_posterior_predictive(x_cand)   # (S,)

    X_aug = np.vstack([X_labeled, x_cand[None, :]])            # (m+1, d)
    z_aug = np.append(z_labeled, z_cand)                       # (m+1,)

    dp_aug_list = []
    for y_s in y_samples:
        y_aug      = np.append(y_labeled, y_s)
        y_pred_aug = model.predict(X_aug)
        dp_aug_list.append(abs(demographic_parity_gap(y_pred_aug, z_aug)))

    return float(np.mean(dp_aug_list)) - dp_now
# ══════════════════════════════════════════════════════════════════════════════
# 5 ─ ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class FairBayesianEngine:
    """
    Fair Active-Learning engine with a true Bayesian backbone.

    Parameters
    ----------
    mode         : 'naive' | 'alpha_aggregate' | 'fal_nested' | 'fal_nested_append'
    alpha        : trade-off weight ∈ [0,1]
                   score = (1-α)·U(x) − α·ΔF(x)
                   α=0 → pure uncertainty; α=1 → pure Bayesian fairness gain
    bgd_weight   : minority over-weighting in Balance GD during MAP fit
    top_k_frac   : fraction kept by Stage-1 uncertainty filter (nested modes)
    lam          : Bayesian prior precision λ
    n_mc_samples : MC draws per fairness-gain estimate
    fg_subsample : max pool size for fairness-gain computation (speed control)
    """

    def __init__(self, mode: str = "alpha_aggregate",
                 alpha: float = 0.5,
                 bgd_weight: float = 5.0,
                 top_k_frac: float = 0.10,
                 lam: float = 1.0,
                 n_mc_samples: int = 20,
                 fg_subsample: int = 200):
        self.mode        = mode
        self.alpha       = alpha
        self.top_k_frac  = top_k_frac
        self.fg_subsample = fg_subsample

        self.model = BayesianLogisticRegression(
            lam=lam, n_iter=200, lr=0.05,
            n_mc_samples=n_mc_samples, bgd_weight=bgd_weight,
        )
        self.labeled_idx: list[int] = []
        self._fitted = False

    # ── Refit ─────────────────────────────────────────────────────────────────

    def update(self, X_labeled: np.ndarray, y_labeled: np.ndarray,
               z_labeled: np.ndarray) -> None:
        """Refit the full Laplace-approximate posterior on the labeled set."""
        self.model.fit(X_labeled, y_labeled, z_labeled)
        self._fitted = True

    # ── Uncertainty (epistemic, from posterior predictive) ────────────────────

    def _uncertainty(self, X_pool: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.random.rand(len(X_pool))
        return self.model.epistemic_uncertainty(X_pool)

    # ── Bayesian fairness gain for a candidate subset ─────────────────────────

    def _fairness_gains(self, X_pool: np.ndarray, Z_pool: np.ndarray,
                        available: np.ndarray,
                        X_labeled: np.ndarray, y_labeled: np.ndarray,
                        z_labeled: np.ndarray) -> np.ndarray:
        """
        Compute ΔF(x_i) for each candidate i in `available`.
        To keep runtime tractable, evaluates at most `fg_subsample` candidates;
        the rest are assigned ΔF = 0 (neutral).
        Returns array of shape (len(available),).
        """
        gains = np.zeros(len(available), dtype=float)
        if not self._fitted:
            return gains

        # Sub-sample for speed if the pool is large
        if len(available) > self.fg_subsample:
            chosen = np.random.choice(len(available), self.fg_subsample,
                                      replace=False)
        else:
            chosen = np.arange(len(available))

        for local_idx in chosen:
            global_idx = available[local_idx]
            gains[local_idx] = bayesian_fairness_gain(
                model    = self.model,
                x_cand   = X_pool[global_idx],
                z_cand   = int(Z_pool[global_idx]),
                X_labeled= X_labeled,
                y_labeled= y_labeled,
                z_labeled= z_labeled,
            )
        return gains

    # ── Selection strategies ──────────────────────────────────────────────────

    def select_next(self, X_pool: np.ndarray, Z_pool: np.ndarray,
                    available: np.ndarray,
                    X_labeled: np.ndarray, y_labeled: np.ndarray,
                    z_labeled: np.ndarray) -> int:
        """
        Choose the next sample to label.

        All non-naive modes use the Bayesian fairness gain ΔF(x).
        """
        U      = self._uncertainty(X_pool[available])           # (|avail|,)
        U_norm = (U - U.min()) / (np.ptp(U) + 1e-12)

        if self.mode == "naive":
            return int(available[np.argmax(U_norm)])

        # Compute Bayesian fairness gains for all modes that need them
        FG = self._fairness_gains(X_pool, Z_pool, available,
                                  X_labeled, y_labeled, z_labeled)
        # Normalise so both terms are on [0,1]
        # ΔF is typically small negatives (good) to small positives (bad).
        # We negate and normalise so that higher = more fairness-improving.
        FG_neg  = -FG
        FG_norm = (FG_neg - FG_neg.min()) / (np.ptp(FG_neg) + 1e-12)

        if self.mode == "alpha_aggregate":
            # score = (1-α)·U + α·FG_norm
            # Higher → more uncertain AND more fairness-improving
            scores = (1.0 - self.alpha) * U_norm + self.alpha * FG_norm
            return int(available[np.argmax(scores)])

        elif self.mode == "fal_nested":
            return self._nested(U_norm, FG_norm, available, Z_pool,
                                append_fallback=False)

        elif self.mode == "fal_nested_append":
            return self._nested(U_norm, FG_norm, available, Z_pool,
                                append_fallback=True)

        else:
            raise ValueError(f"Unknown mode: {self.mode!r}")

    def _nested(self, U_norm: np.ndarray, FG_norm: np.ndarray,
                available: np.ndarray, Z_pool: np.ndarray,
                append_fallback: bool) -> int:
        """
        FAL-Nested (and FAL-Nested-Append) selection logic.

        Stage 1 — filter to top-K% most uncertain candidates.
        Stage 2 — within that pool, select by maximum Bayesian fairness gain
                  (most fairness-improving among the uncertain candidates).

        FAL-Nested-Append extension:
          If no minority (Z=0) appears in the top-K pool, bypass Stage-1 and
          select the most fairness-improving minority from the full pool.
          This prevents minority starvation when uncertainty concentrates on
          the majority group.
        """
        top_k    = max(1, int(len(available) * self.top_k_frac))
        top_local = np.argsort(U_norm)[-top_k:]          # most uncertain

        minority_in_top = (Z_pool[available[top_local]] == 0)

        if minority_in_top.any() or not append_fallback:
            # Stage-2: best Bayesian fairness gain within top-K
            # For fal_nested when no minority: still pick best FG in top-K
            best_in_top = top_local[np.argmax(FG_norm[top_local])]
            return int(available[best_in_top])

        else:
            # Append fallback: best fairness gain among ALL minority in full pool
            minority_mask = (Z_pool[available] == 0)
            if minority_mask.any():
                minority_local = np.where(minority_mask)[0]
                best_local = minority_local[np.argmax(FG_norm[minority_local])]
            else:
                best_local = np.argmax(FG_norm)
            return int(available[best_local])

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.zeros(len(X), dtype=int)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.full((len(X), 2), 0.5)
        return self.model.predict_proba(X)
# ══════════════════════════════════════════════════════════════════════════════
# 6 ─ EXPERIMENT LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(X, y, z, mode: str, steps: int = 150,
                   alpha: float = 0.5, bgd_weight: float = 5.0,
                   top_k_frac: float = 0.10, lam: float = 1.0,
                   n_mc_samples: int = 20, fg_subsample: int = 200,
                   eval_every: int = 5) -> dict:
    engine = FairBayesianEngine(
        mode=mode, alpha=alpha, bgd_weight=bgd_weight,
        top_k_frac=top_k_frac, lam=lam,
        n_mc_samples=n_mc_samples, fg_subsample=fg_subsample,
    )
    available = np.arange(len(X))
    np.random.shuffle(available)
    available = list(available)

    history   = defaultdict(list)
    step_axis = []

    # ── Warm-start: 5 samples per class (balanced) ───────────────────────────
    warm = []
    for cls in [0, 1]:
        cands = [i for i in available if y[i] == cls][:5]
        warm.extend(cands)
    for i in warm:
        engine.labeled_idx.append(i)
        available.remove(i)
    engine.update(X[engine.labeled_idx], y[engine.labeled_idx],
                  z[engine.labeled_idx])

    # ── Active-learning loop ─────────────────────────────────────────────────
    for step in range(steps):
        lbl_arr = np.array(engine.labeled_idx)
        idx = engine.select_next(
            X_pool   = X,
            Z_pool   = z,
            available= np.array(available),
            X_labeled= X[lbl_arr],
            y_labeled= y[lbl_arr],
            z_labeled= z[lbl_arr],
        )
        minority_flag = int(z[idx] == 0)
        engine.labeled_idx.append(idx)
        available.remove(idx)
        engine.update(X[engine.labeled_idx], y[engine.labeled_idx],
                      z[engine.labeled_idx])

        if (step + 1) % eval_every == 0:
            lbl           = engine.labeled_idx
            y_lbl, z_lbl  = y[lbl], z[lbl]
            y_pred         = engine.predict(X[lbl])
            y_prob         = engine.predict_proba(X[lbl])[:, 1]

            step_axis.append(step + 1)
            history["minority_share"].append(float((z_lbl == 0).mean()))
            history["majority_share"].append(float((z_lbl == 1).mean()))
            history["minority_selected_this_step"].append(minority_flag)
            history["accuracy"].append(accuracy_score(y_lbl, y_pred))
            try:
                history["auc_roc"].append(roc_auc_score(y_lbl, y_prob))
            except Exception:
                history["auc_roc"].append(0.5)
            history["f1"].append(f1_score(y_lbl, y_pred, zero_division=0))
            history["brier"].append(brier_score_loss(y_lbl, y_prob))
            history["dp_gap"].append(demographic_parity_gap(y_pred, z_lbl))
            eo = equalized_odds_gap(y_lbl, y_pred, z_lbl)
            for k, v in eo.items():
                history[k].append(v)
            history["ece"].append(expected_calibration_error(y_lbl, y_prob))
            u = engine._uncertainty(X[lbl])
            history["avg_uncertainty"].append(float(u.mean()))

    return {
        "history":     dict(history),
        "steps":       step_axis,
        "engine":      engine,
        "labeled_idx": engine.labeled_idx,
    }
