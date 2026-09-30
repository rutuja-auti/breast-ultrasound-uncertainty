"""
Experiment 1: compare four uncertainty methods on the held-out TEST set.

    python src/evaluate.py

Needs checkpoints/seed0.pt (for single / MC dropout / TTA). The ensemble is
included if 2 or more seeds are trained (we use 5: see run_experiments.ps1).

Research questions this script answers:
  RQ1  Accuracy:     does any uncertainty method also improve Dice / IoU?
  RQ2  Calibration:  whose probabilities are most honest? (ECE, reliability diagram)
  RQ3  Usefulness:   does uncertainty flag the images the model gets wrong?
                     (Spearman correlation, and the "refer to a radiologist" experiment)
  RQ4  Subgroups:    is the model worse / less sure on malignant tumours?

Statistics: every mean Dice gets a 95% bootstrap confidence interval, and each
method is compared with the single-model baseline using a paired Wilcoxon
signed-rank test (paired = the same test images are used by both methods).

Outputs in results/evaluation/:
  results_table.md          main table, ready to paste into your dissertation
  metrics.json              all numbers
  reliability_diagram.png   RQ2
  referral_curve.png        RQ3
  uncertainty_vs_dice.png   RQ3 (for the best method)
  examples.png              best and worst test cases with uncertainty maps
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon

from data import CLASSES, get_datasets, to_tensor
from metrics import dice_iou, expected_calibration_error
from uncertainty import METHODS, entropy, image_uncertainty
from unet import load_models

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "evaluation"
REFERRAL_FRACTIONS = np.linspace(0, 0.5, 11)  # refer 0%, 5%, ..., 50% of images


def bootstrap_ci(values, n_boot=2000, seed=0):
    """95% confidence interval of the mean, by resampling test images with replacement.
    It tells the reader how much the result could change with a different test set."""
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, len(values)).mean() for _ in range(n_boot)]
    return np.percentile(means, [2.5, 97.5])


def referral_curve(uncertainty, dice):
    """Mean Dice on the images the model KEEPS after referring the most uncertain
    fraction to a radiologist. Also the 'oracle' curve (refer the truly worst
    images first), the best any uncertainty measure could possibly do."""
    by_unc = np.argsort(-uncertainty)
    by_err = np.argsort(dice)
    kept, oracle = [], []
    for f in REFERRAL_FRACTIONS:
        k = int(round(f * len(dice)))
        kept.append(dice[by_unc[k:]].mean())
        oracle.append(dice[by_err[k:]].mean())
    return np.array(kept), np.array(oracle)


def evaluate_method(prob, masks, labels):
    """All metrics for one method's predictions."""
    dice, iou = dice_iou(prob > 0.5, masks)
    ece, bin_conf, bin_acc, _ = expected_calibration_error(prob, masks)
    unc = image_uncertainty(prob)
    rho, p_rho = spearmanr(unc, dice)
    kept, oracle = referral_curve(unc, dice)
    by_class = {}
    for label, cls in enumerate(CLASSES):
        sel = labels == label
        by_class[cls] = {"n": int(sel.sum()), "dice": float(dice[sel].mean()),
                         "uncertainty": float(unc[sel].mean())}
    i20 = int(np.argmin(np.abs(REFERRAL_FRACTIONS - 0.2)))
    return {
        "dice_per_image": dice, "unc_per_image": unc, "prob": prob,
        "bin_conf": bin_conf, "bin_acc": bin_acc, "kept": kept, "oracle": oracle,
        "summary": {
            "dice": float(dice.mean()),
            "dice_95ci": [float(v) for v in bootstrap_ci(dice)],
            "iou": float(iou.mean()),
            # Complete failures: tumour missed or found in the wrong place.
            "failures_dice_below_0.1": int((dice < 0.1).sum()),
            "ece": float(ece),
            "spearman_rho": float(rho), "spearman_p": float(p_rho),
            "dice_after_20pct_referral": float(kept[i20]),
            "referral_gain_20pct": float(kept[i20] - dice.mean()),
            "by_class": by_class,
        },
    }


def plot_reliability(results, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0.5, 1], [0.5, 1], "k--", label="perfectly calibrated")
    for name, r in results.items():
        ax.plot(r["bin_conf"], r["bin_acc"], "o-", ms=3,
                label=f"{name} (ECE={r['summary']['ece']:.4f})")
    ax.set(xlabel="model confidence", ylabel="actual accuracy",
           title="Reliability diagram (all test pixels)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_referral(results, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, r in results.items():
        ax.plot(REFERRAL_FRACTIONS * 100, r["kept"], "o-", ms=3, label=name)
    best = max(results, key=lambda n: results[n]["summary"]["dice"])
    ax.plot(REFERRAL_FRACTIONS * 100, results[best]["oracle"], "k--", lw=1,
            label=f"oracle ({best})")
    ax.set(xlabel="% of test images referred to a radiologist (most uncertain first)",
           ylabel="mean Dice on images the AI keeps",
           title="Uncertainty-based referral (flat line = uncertainty is useless)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_uncertainty_vs_dice(r, labels, name, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for label, cls in enumerate(CLASSES):
        sel = labels == label
        ax.scatter(r["unc_per_image"][sel], r["dice_per_image"][sel], alpha=0.7, label=cls)
    s = r["summary"]
    ax.set(xlabel="image uncertainty (mean entropy)", ylabel="Dice",
           title=f"{name}: Spearman ρ = {s['spearman_rho']:.2f} (p = {s['spearman_p']:.1e})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_examples(images, masks, r, name, path, n=3):
    """The n best and n worst test images for one method."""
    dice, prob = r["dice_per_image"], r["prob"]
    order = np.argsort(dice)
    picks = list(order[-n:][::-1]) + list(order[:n])
    titles = ["ultrasound", "ground truth", "prediction", "errors", f"uncertainty ({name})"]
    fig, axes = plt.subplots(len(picks), 5, figsize=(13, 2.7 * len(picks)))
    for row, i in enumerate(picks):
        pred = prob[i] > 0.5
        # Errors: red = false alarm (predicted tumour, isn't), blue = missed tumour.
        err = np.zeros((*pred.shape, 3))
        err[pred & (masks[i] == 0)] = [1, 0, 0]
        err[~pred & (masks[i] == 1)] = [0, 0.4, 1]
        panels = [images[i], masks[i], pred, err, entropy(prob[i])]
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            if col == 4:
                ax.imshow(panel, cmap="magma", vmin=0, vmax=1)
            else:
                ax.imshow(panel, cmap="gray" if panel.ndim == 2 else None)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col], fontsize=10)
        label = "BEST" if row < n else "WORST"
        axes[row, 0].set_ylabel(f"{label}\nDice={dice[i]:.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_table(results, pvalues, path):
    lines = [
        "| Method | Dice (95% CI) | IoU | Failures (Dice < 0.1) | ECE (lower = better) "
        "| Spearman rho (more negative = better) | Dice gain @ 20% referral "
        "| Images better / worse than single | p vs single |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        s = r["summary"]
        lo, hi = s["dice_95ci"]
        p = pvalues.get(name)
        # A small p-value only says the methods DIFFER, not which is better,
        # so we also show how many images improved vs got worse.
        vs = "-" if name == "single" else f"{s['images_better']} / {s['images_worse']}"
        lines.append(
            f"| {name} | {s['dice']:.3f} ({lo:.3f}-{hi:.3f}) | {s['iou']:.3f} "
            f"| {s['failures_dice_below_0.1']} | {s['ece']:.4f} "
            f"| {s['spearman_rho']:.2f} | {s['referral_gain_20pct']:+.3f} | {vs} "
            f"| {'-' if p is None else f'{p:.3g}'} |")
    lines += ["", "| Method | " + " | ".join(f"{c} Dice | {c} uncertainty" for c in CLASSES) + " |",
              "|---|" + "---|---|" * len(CLASSES)]
    for name, r in results.items():
        bc = r["summary"]["by_class"]
        lines.append(f"| {name} | " + " | ".join(
            f"{bc[c]['dice']:.3f} | {bc[c]['uncertainty']:.3f}" for c in CLASSES) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_models(ROOT / "checkpoints", device)

    _, _, test_ds, _ = get_datasets()
    images = test_ds.images[test_ds.indices]
    masks = test_ds.masks[test_ds.indices]
    labels = test_ds.labels[test_ds.indices]
    x = to_tensor(images)
    print(f"Test set: {len(x)} images\n")

    torch.manual_seed(0)  # makes MC dropout reproducible
    results = {}
    for name, predict in METHODS.items():
        if name == "ensemble" and len(models) < 2:
            print("Skipping ensemble: train more seeds first (see run_experiments.ps1)")
            continue
        print(f"Running {name} ...")
        results[name] = evaluate_method(predict(models, x, device), masks, labels)

    # Paired test: is each method's per-image Dice different from the baseline's?
    pvalues = {}
    base = results["single"]["dice_per_image"]
    for name, r in results.items():
        if name != "single":
            diff = r["dice_per_image"] - base
            pvalues[name] = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
            r["summary"]["images_better"] = int((diff > 0.001).sum())
            r["summary"]["images_worse"] = int((diff < -0.001).sum())

    OUT.mkdir(parents=True, exist_ok=True)
    main_method = "ensemble" if "ensemble" in results else "mc_dropout"
    plot_reliability(results, OUT / "reliability_diagram.png")
    plot_referral(results, OUT / "referral_curve.png")
    plot_uncertainty_vs_dice(results[main_method], labels, main_method, OUT / "uncertainty_vs_dice.png")
    plot_examples(images, masks, results[main_method], main_method, OUT / "examples.png")

    summary = {name: r["summary"] for name, r in results.items()}
    for name, p in pvalues.items():
        summary[name]["wilcoxon_p_vs_single"] = p
    summary["test_images"] = len(x)
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))

    print("\n" + write_table(results, pvalues, OUT / "results_table.md"))
    print("\nHow to read this:")
    print("  ECE:        lower = probabilities are more honest")
    print("  Spearman rho: more NEGATIVE = uncertainty better at flagging bad segmentations")
    print("  Referral:   bigger gain = referring uncertain cases helps more")
    print("  p < 0.05:   per-image Dice differs significantly from the single model;")
    print("              check 'better / worse' to see in which direction")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
