"""
Runs all four strategies (naive, alpha_aggregate, fal_nested,
fal_nested_append) on the Give Me Some Credit dataset, prints a summary
table, and saves 18 diagnostic plots to ./plots.

Usage:
    python test.py
"""

from src.model import load_credit_data, run_experiment, LABEL_MAP
from src.visualize import save_all_plots, print_summary


def main():
    print("Loading data …")
    X, y, z, feats = load_credit_data()
    print(f"  {len(X)} samples  |  {X.shape[1]} features  |  "
          f"Young(Z=0)={(z==0).sum()}  Old(Z=1)={(z==1).sum()}  "
          f"Delinquent={y.sum()}")

    # ── Hyperparameters ───────────────────────────────────────────────────
    STEPS        = 150   # active-learning iterations per strategy
    EVAL_EVERY   = 5     # evaluate every N steps
    ALPHA        = 0.5   # fairness/uncertainty trade-off
    BGD_WEIGHT   = 5.0   # minority overweighting in Laplace MAP fit
    TOP_K_FRAC   = 0.10  # Stage-1 uncertainty filter fraction
    LAM          = 1.0   # Bayesian prior precision (λ = 1/σ²_prior)
    N_MC_SAMPLES = 20    # posterior predictive MC draws per fairness-gain eval
    FG_SUBSAMPLE = 150   # max candidates evaluated for fairness gain per step

    results = {}
    for mode in ["naive", "alpha_aggregate", "fal_nested", "fal_nested_append"]:
        print(f"\nRunning  [{LABEL_MAP[mode]}]  …")
        results[mode] = run_experiment(
            X, y, z,
            mode        = mode,
            steps       = STEPS,
            alpha       = ALPHA,
            bgd_weight  = BGD_WEIGHT,
            top_k_frac  = TOP_K_FRAC,
            lam         = LAM,
            n_mc_samples= N_MC_SAMPLES,
            fg_subsample= FG_SUBSAMPLE,
            eval_every  = EVAL_EVERY,
        )
        print(f"  Done. Labeled {len(results[mode]['labeled_idx'])} points.")

    print_summary(results)
    save_all_plots(results, X, y, out_dir="plots")


if __name__ == "__main__":
    main()
