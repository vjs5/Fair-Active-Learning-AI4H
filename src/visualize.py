"""
Fair Bayesian Engine — plots & summary table.

18 diagnostic plots comparing the naive / alpha_aggregate / fal_nested /
fal_nested_append strategies, plus a console summary table.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from src.model import LABEL_MAP

PALETTE = {
    "naive":             "#E63946",
    "alpha_aggregate":   "#2A9D8F",
    "fal_nested":        "#457B9D",
    "fal_nested_append": "#8338EC",
    "target":            "#aaaaaa",
    "young":             "#F4A261",
    "old":               "#264653",
}
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8F9FA",
    "axes.edgecolor":   "#CCCCCC",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.color":       "#DDDDDD",
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
})

# ══════════════════════════════════════════════════════════════════════════════
# 7 ─ VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def smooth(arr, w=3):
    arr = np.array(arr, dtype=float)
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="same")


def _ax_line(ax, results, key, smooth_w=1, **kwargs):
    for mode, res in results.items():
        vals = smooth(res["history"][key], smooth_w)
        ax.plot(res["steps"], vals,
                label=LABEL_MAP[mode], color=PALETTE[mode], **kwargs)


def _new_fig(title):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    return fig, ax


def _save(fig, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {filename}")


# ── 18 plot functions ─────────────────────────────────────────────────────────

def plot_minority_share_over_time(results, out_dir):
    fig, ax = _new_fig("Minority (Young) Representation Share over Steps")
    _ax_line(ax, results, "minority_share", smooth_w=1, lw=2)
    ax.axhline(0.5, color=PALETTE["target"], ls="--", lw=1.2, label="Target 50 %")
    ax.set_xlabel("Active-Learning Step"); ax.set_ylabel("Fraction of labeled set")
    ax.set_ylim(0, 1); ax.legend(fontsize=9)
    _save(fig, out_dir, "01_minority_share_over_time.png")


def plot_final_minority_share_bar(results, out_dir):
    fig, ax = _new_fig("Final Minority Share by Strategy")
    modes_list = list(results.keys())
    y_vals = [results[m]["history"]["minority_share"][-1] for m in modes_list]
    bars = ax.bar([LABEL_MAP[m] for m in modes_list], y_vals,
                  color=[PALETTE[m] for m in modes_list], edgecolor="white", width=0.5)
    ax.axhline(0.5, color=PALETTE["target"], ls="--", lw=1.2, label="Target 50 %")
    for bar, v in zip(bars, y_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.1%}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 0.8); ax.set_ylabel("Minority fraction")
    ax.tick_params(axis="x", labelsize=8); ax.legend(fontsize=9)
    _save(fig, out_dir, "02_final_minority_share_bar.png")


def plot_group_composition_stacked(results, out_dir):
    fig, ax = _new_fig("Group Composition of Labeled Set (Final Step)")
    modes_list = list(results.keys())
    x = np.arange(len(modes_list))
    min_vals = [results[m]["history"]["minority_share"][-1] for m in modes_list]
    maj_vals = [1 - v for v in min_vals]
    ax.bar(x, min_vals, label="Young (Z=0)", color=PALETTE["young"], width=0.5)
    ax.bar(x, maj_vals, bottom=min_vals, label="Old (Z=1)", color=PALETTE["old"], width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_MAP[m] for m in modes_list], fontsize=8)
    ax.set_ylabel("Share"); ax.legend(fontsize=9)
    _save(fig, out_dir, "03_group_composition_stacked.png")


def plot_accuracy(results, out_dir):
    fig, ax = _new_fig("Accuracy over Active-Learning Steps")
    _ax_line(ax, results, "accuracy", smooth_w=3, lw=2)
    ax.set_xlabel("Step"); ax.set_ylabel("Accuracy"); ax.legend(fontsize=9)
    _save(fig, out_dir, "04_accuracy.png")


def plot_auc_roc(results, out_dir):
    fig, ax = _new_fig("AUC-ROC over Active-Learning Steps")
    _ax_line(ax, results, "auc_roc", smooth_w=3, lw=2)
    ax.set_xlabel("Step"); ax.set_ylabel("AUC-ROC"); ax.legend(fontsize=9)
    _save(fig, out_dir, "05_auc_roc.png")


def plot_f1(results, out_dir):
    fig, ax = _new_fig("F1 Score over Active-Learning Steps")
    _ax_line(ax, results, "f1", smooth_w=3, lw=2)
    ax.set_xlabel("Step"); ax.set_ylabel("F1 Score"); ax.legend(fontsize=9)
    _save(fig, out_dir, "06_f1_score.png")


def plot_demographic_parity_gap(results, out_dir):
    fig, ax = _new_fig("Demographic Parity Gap  |P(ŷ=1|Young) – P(ŷ=1|Old)|")
    _ax_line(ax, results, "dp_gap", smooth_w=3, lw=2)
    ax.axhline(0, color=PALETTE["target"], ls="--", lw=1.2, label="Ideal (0)")
    ax.set_xlabel("Step"); ax.set_ylabel("Gap  (lower → fairer)"); ax.legend(fontsize=9)
    _save(fig, out_dir, "07_demographic_parity_gap.png")


def plot_tpr_gap(results, out_dir):
    fig, ax = _new_fig("Equalized Odds — TPR Gap  |TPR_Young – TPR_Old|")
    _ax_line(ax, results, "tpr_gap", smooth_w=3, lw=2)
    ax.axhline(0, color=PALETTE["target"], ls="--", lw=1.2, label="Ideal (0)")
    ax.set_xlabel("Step"); ax.set_ylabel("TPR Gap"); ax.legend(fontsize=9)
    _save(fig, out_dir, "08_equalized_odds_tpr_gap.png")


def plot_fpr_gap(results, out_dir):
    fig, ax = _new_fig("Equalized Odds — FPR Gap  |FPR_Young – FPR_Old|")
    _ax_line(ax, results, "fpr_gap", smooth_w=3, lw=2)
    ax.axhline(0, color=PALETTE["target"], ls="--", lw=1.2, label="Ideal (0)")
    ax.set_xlabel("Step"); ax.set_ylabel("FPR Gap"); ax.legend(fontsize=9)
    _save(fig, out_dir, "09_equalized_odds_fpr_gap.png")


def plot_final_tpr_per_group(results, out_dir):
    fig, ax = _new_fig("Final True Positive Rate per Group")
    modes_list = list(results.keys())
    x = np.arange(len(modes_list)); w = 0.3
    tpr_young = [results[m]["history"]["tpr_young"][-1] for m in modes_list]
    tpr_old   = [results[m]["history"]["tpr_old"][-1]   for m in modes_list]
    ax.bar(x - w/2, tpr_young, width=w, label="Young (Z=0)", color=PALETTE["young"])
    ax.bar(x + w/2, tpr_old,   width=w, label="Old (Z=1)",   color=PALETTE["old"])
    ax.set_xticks(x); ax.set_xticklabels([LABEL_MAP[m] for m in modes_list], fontsize=8)
    ax.set_ylabel("TPR"); ax.set_ylim(0, 1); ax.legend(fontsize=9)
    _save(fig, out_dir, "10_final_tpr_per_group.png")


def plot_final_fpr_per_group(results, out_dir):
    fig, ax = _new_fig("Final False Positive Rate per Group")
    modes_list = list(results.keys())
    x = np.arange(len(modes_list)); w = 0.3
    fpr_young = [results[m]["history"]["fpr_young"][-1] for m in modes_list]
    fpr_old   = [results[m]["history"]["fpr_old"][-1]   for m in modes_list]
    ax.bar(x - w/2, fpr_young, width=w, label="Young (Z=0)", color=PALETTE["young"])
    ax.bar(x + w/2, fpr_old,   width=w, label="Old (Z=1)",   color=PALETTE["old"])
    ax.set_xticks(x); ax.set_xticklabels([LABEL_MAP[m] for m in modes_list], fontsize=8)
    ax.set_ylabel("FPR"); ax.set_ylim(0, 1); ax.legend(fontsize=9)
    _save(fig, out_dir, "11_final_fpr_per_group.png")


def plot_ece(results, out_dir):
    fig, ax = _new_fig("Expected Calibration Error (ECE) over Steps")
    _ax_line(ax, results, "ece", smooth_w=3, lw=2)
    ax.axhline(0, color=PALETTE["target"], ls="--", lw=1.2, label="Ideal (0)")
    ax.set_xlabel("Step"); ax.set_ylabel("ECE  (lower → better)"); ax.legend(fontsize=9)
    _save(fig, out_dir, "12_ece.png")


def plot_epistemic_uncertainty(results, out_dir):
    fig, ax = _new_fig("Average Posterior Predictive Entropy over Steps")
    _ax_line(ax, results, "avg_uncertainty", smooth_w=3, lw=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("H[p(y|x,D)]  — posterior predictive entropy")
    ax.legend(fontsize=9)
    _save(fig, out_dir, "13_epistemic_uncertainty.png")


def plot_brier_score(results, out_dir):
    fig, ax = _new_fig("Brier Score over Active-Learning Steps")
    _ax_line(ax, results, "brier", smooth_w=3, lw=2)
    ax.set_xlabel("Step"); ax.set_ylabel("Brier Score  (lower → better)")
    ax.legend(fontsize=9)
    _save(fig, out_dir, "14_brier_score.png")


def plot_calibration_curves(results, X, y, out_dir):
    fig, ax = _new_fig("Reliability Diagram — Calibration Curves (Final Model)")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for mode, res in results.items():
        eng  = res["engine"]
        lbl  = res["labeled_idx"]
        prob = eng.predict_proba(X[lbl])[:, 1]
        frac_pos, mean_pred = calibration_curve(
            y[lbl], prob, n_bins=8, strategy="uniform")
        ax.plot(mean_pred, frac_pos, "o-",
                label=LABEL_MAP[mode], color=PALETTE[mode], lw=1.8, ms=5)
    ax.set_xlabel("Mean Predicted Probability (posterior predictive)")
    ax.set_ylabel("Fraction of Positives")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=9)
    _save(fig, out_dir, "15_calibration_curves.png")


def plot_minority_acquisition_rate(results, out_dir):
    fig, ax = _new_fig("Minority Acquisition Rate (Rolling Avg) per Strategy")
    for mode, res in results.items():
        vals = smooth(res["history"]["minority_selected_this_step"], w=5)
        ax.plot(res["steps"], vals,
                label=LABEL_MAP[mode], color=PALETTE[mode], lw=2)
    ax.axhline(0.5, color=PALETTE["target"], ls="--", lw=1.2,
               label="Equal selection (50 %)")
    ax.set_xlabel("Active-Learning Step")
    ax.set_ylabel("P(minority selected) — rolling avg")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=9)
    _save(fig, out_dir, "16_minority_acquisition_rate.png")


def plot_fairness_radar(results, out_dir):
    metrics_labels = ["Accuracy", "AUC-ROC", "F1", "1−DP Gap", "1−TPR Gap",
                      "1−FPR Gap", "1−ECE", "Minority\nShare"]

    def extract(mode):
        h = results[mode]["history"]
        return [
            h["accuracy"][-1],
            h["auc_roc"][-1],
            h["f1"][-1],
            max(0, 1 - abs(h["dp_gap"][-1])),
            max(0, 1 - h["tpr_gap"][-1]),
            max(0, 1 - h["fpr_gap"][-1]),
            max(0, 1 - h["ece"][-1]),
            h["minority_share"][-1],
        ]

    n = len(metrics_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.suptitle("Fairness & Performance Radar (Final Step)",
                 fontsize=13, fontweight="bold", y=1.02)
    ax.set_facecolor("#F8F9FA")
    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_labels, fontsize=9)
    ax.yaxis.set_tick_params(labelsize=7)
    for mode in results:
        vals = extract(mode) + [extract(mode)[0]]
        ax.plot(angles, vals, color=PALETTE[mode], lw=2, label=LABEL_MAP[mode])
        ax.fill(angles, vals, color=PALETTE[mode], alpha=0.10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    _save(fig, out_dir, "17_fairness_radar.png")


def plot_algorithm_comparison_heatmap(results, out_dir):
    modes_list = list(results.keys())
    col_keys   = ["accuracy", "auc_roc", "f1", "dp_gap",
                  "tpr_gap", "fpr_gap", "ece", "brier", "minority_share"]
    col_labels = ["Accuracy", "AUC-ROC", "F1", "DP Gap",
                  "TPR Gap", "FPR Gap", "ECE", "Brier", "Min. Share"]
    lower_better = {"dp_gap", "tpr_gap", "fpr_gap", "ece", "brier"}

    data = np.array([
        [results[m]["history"][k][-1] for k in col_keys]
        for m in modes_list
    ])
    norm_data = np.zeros_like(data)
    for j, k in enumerate(col_keys):
        col = data[:, j]; mn, mx = col.min(), col.max()
        if mx == mn:
            norm_data[:, j] = 0.5
        else:
            scaled = (col - mn) / (mx - mn)
            norm_data[:, j] = (1 - scaled) if k in lower_better else scaled

    fig, ax = plt.subplots(figsize=(13, max(3, len(modes_list) * 1.4)))
    fig.suptitle("Algorithm Comparison Heatmap (Final Checkpoint)",
                 fontsize=13, fontweight="bold", y=1.02)
    im = ax.imshow(norm_data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(modes_list)))
    ax.set_yticklabels([LABEL_MAP[m] for m in modes_list], fontsize=10)
    for i in range(len(modes_list)):
        for j in range(len(col_keys)):
            ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center",
                    fontsize=9, color="black", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04,
                 label="Relative quality (green = better)")
    _save(fig, out_dir, "18_algorithm_comparison_heatmap.png")


def save_all_plots(results: dict, X, y, out_dir: str = "plots") -> None:
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nSaving plots → {os.path.abspath(out_dir)}/")
    plot_minority_share_over_time(results, out_dir)
    plot_final_minority_share_bar(results, out_dir)
    plot_group_composition_stacked(results, out_dir)
    plot_accuracy(results, out_dir)
    plot_auc_roc(results, out_dir)
    plot_f1(results, out_dir)
    plot_demographic_parity_gap(results, out_dir)
    plot_tpr_gap(results, out_dir)
    plot_fpr_gap(results, out_dir)
    plot_final_tpr_per_group(results, out_dir)
    plot_final_fpr_per_group(results, out_dir)
    plot_ece(results, out_dir)
    plot_epistemic_uncertainty(results, out_dir)
    plot_brier_score(results, out_dir)
    plot_calibration_curves(results, X, y, out_dir)
    plot_minority_acquisition_rate(results, out_dir)
    plot_fairness_radar(results, out_dir)
    plot_algorithm_comparison_heatmap(results, out_dir)
    print(f"Done — {len(os.listdir(out_dir))} files in '{out_dir}/'.\n")


# ══════════════════════════════════════════════════════════════════════════════
# 8 ─ SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: dict):
    print("\n" + "═" * 100)
    print("  FINAL METRICS SUMMARY  (last evaluation checkpoint)")
    print("═" * 100)
    print(f"{'Mode':<30}  {'MinShare':>8}  {'Accuracy':>8}  {'AUC-ROC':>8}"
          f"  {'F1':>6}  {'DP Gap':>7}  {'TPR Gap':>7}  {'FPR Gap':>7}"
          f"  {'ECE':>6}  {'Brier':>6}")
    print("─" * 100)
    for mode, res in results.items():
        h = res["history"]
        print(f"{LABEL_MAP[mode]:<30}  "
              f"{h['minority_share'][-1]:>8.3f}  "
              f"{h['accuracy'][-1]:>8.3f}  "
              f"{h['auc_roc'][-1]:>8.3f}  "
              f"{h['f1'][-1]:>6.3f}  "
              f"{h['dp_gap'][-1]:>+7.3f}  "
              f"{h['tpr_gap'][-1]:>7.3f}  "
              f"{h['fpr_gap'][-1]:>7.3f}  "
              f"{h['ece'][-1]:>6.3f}  "
              f"{h['brier'][-1]:>6.3f}")
    print("═" * 100 + "\n")
