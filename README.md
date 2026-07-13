# Fair Bayesian Engine

A Fair Active Learning (FAL) system for credit risk modelling, built on a
Bayesian logistic regression backbone (Laplace approximation). It compares
four label-selection strategies on the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)
dataset and measures whether fairness-aware sampling can close a
demographic representation gap without sacrificing predictive quality.

**Sensitive attribute:** age — Young (`< 35`) is the minority group (`Z=0`),
Old (`>= 35`) is the majority (`Z=1`).

## The problem

A standard uncertainty-based active learner gravitates toward data-rich
regions dominated by the majority group. Over 150 labelling steps it ends
up sampling ~90% Old borrowers and only ~10% Young — the model barely
learns from the minority group and produces biased predictions for them.

## Strategies compared

| Strategy | Mechanic | Goal |
|---|---|---|
| **Naive** | Uncertainty only (Shannon entropy of the posterior predictive) | Maximise accuracy; ignores fairness |
| **FAL α-Aggregate** | `score = (1-α)·uncertainty + α·fairness_gain` | Balance predictive quality and minority representation |
| **FAL-Nested** | Stage 1: top-10% most uncertain. Stage 2: best fairness gain within that pool | Uncertainty-first, fairness as tiebreaker |
| **FAL-Nested-Append** | FAL-Nested + fallback: if no minority appears in the top-K, pick the best minority fairness-gain from the full pool | Prevents minority starvation |

Every model refit also applies **Balance Gradient Descent**: minority
(Young) examples get a 5× sample weight during MAP estimation, so each
gradient step corrects harder for errors on the under-represented group.

## Bayesian backbone

Logistic regression fit via Laplace approximation:
- MAP estimate `θ_MAP = argmax log p(y|X,θ) + log p(θ)`
- Posterior covariance `Σ = (H + λI)⁻¹`
- Posterior predictive `p(y=1|x,D) = σ(κ(σ²)·θ_MAP·x)` (MacKay 1992 probit approximation)
- Epistemic uncertainty = Shannon entropy of the posterior predictive
- Fairness gain `ΔF(x*,z*)` = expected change in the demographic parity
  gap from labelling `x*`, estimated by Monte-Carlo sampling from the
  posterior predictive

## Results (from the reference run, 150 steps)

- **Representation:** minority share rose from ~9% (Naive) to ~50%
  (α-Aggregate) and beyond (Nested).
- **Predictive quality:** accuracy, AUC-ROC, and F1 stayed within a few
  points of Naive across all strategies.
- **Fairness:** demographic parity gap and TPR/FPR gaps between age
  groups shrank substantially under both fair strategies.
- **Calibration:** ECE and Brier score were comparable across strategies;
  all converged toward the calibration diagonal.
- **Uncertainty:** Naive resolved uncertainty fastest by staying in
  familiar territory; fair strategies kept uncertainty elevated longer,
  reflecting genuine exploration of the minority region.

**Bottom line:** minority representation moved from ~10% to ~50% in a
live active-learning loop without a collapse in accuracy, by combining
FAL selection with Balance Gradient Descent.

## Files

```
model.py       # data loading, Bayesian logistic regression, fairness
               # metrics, the FAL engine, and the active-learning loop
visualize.py   # 18 diagnostic plots + console summary table
test.py        # runs all 4 strategies end-to-end and produces the outputs
```

## Usage

Download `cs-training.csv` from the Give Me Some Credit dataset and place
it in the project root, then:

```bash
pip install numpy pandas scikit-learn scipy matplotlib
python test.py
```

This prints a metrics summary table and saves 18 plots to `./plots`,
covering representation, predictive performance, fairness gaps,
per-group TPR/FPR, calibration, and an overall algorithm comparison
heatmap.
