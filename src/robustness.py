"""
Experiment 2: what happens when image quality gets worse?

    python src/robustness.py

Real-world ultrasound is often worse than a curated dataset: cheaper or older
machines, low-resource clinics, rushed scans. We simulate three kinds of
degradation at increasing severity and ask:

  RQ5  How fast does segmentation accuracy (Dice) drop?
  RQ6  Does the model's UNCERTAINTY rise as its accuracy falls? If yes, the AI
       can warn "this scan is poor quality, don't trust me" instead of silently
       failing. Which uncertainty method reacts best?

Corruptions (severity 0 = original image):
  speckle noise    extra grainy ultrasound noise
  blur             loss of sharpness (poor focus / motion)
  low resolution   image downsampled then upsampled (cheap, low-res devices)

Outputs in results/robustness/:
  robustness.png            Dice and uncertainty vs severity, per method
  corruption_examples.png   what each corruption looks like
  robustness.json           all numbers
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

from data import get_datasets, to_tensor
from metrics import dice_iou
from uncertainty import METHODS, image_uncertainty
from unet import load_models

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "robustness"


def speckle(x, s):
    g = torch.Generator().manual_seed(0)  # same noise for every method = fair comparison
    return (x * (1 + s * torch.randn(x.shape, generator=g))).clamp(0, 1)


def blur(x, sigma):
    if sigma == 0:
        return x
    k = 2 * math.ceil(3 * sigma) + 1  # kernel wide enough for this sigma
    return gaussian_blur(x, [k, k], [sigma, sigma])


def low_resolution(x, factor):
    if factor == 1:
        return x
    size = x.shape[-1]
    small = F.interpolate(x, size=size // factor, mode="bilinear", antialias=True)
    return F.interpolate(small, size=size, mode="bilinear")


CORRUPTIONS = {
    "speckle noise":  (speckle,        [0, 0.1, 0.2, 0.3, 0.5]),
    "blur":           (blur,           [0, 1, 2, 3, 5]),
    "low resolution": (low_resolution, [1, 2, 4, 6, 8]),
}


def plot_examples(x, path):
    fig, axes = plt.subplots(len(CORRUPTIONS), 5, figsize=(12, 7.5))
    for row, (name, (fn, levels)) in enumerate(CORRUPTIONS.items()):
        for col, level in enumerate(levels):
            ax = axes[row, col]
            ax.imshow(fn(x[:1], level)[0, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"severity {col}  ({level})", fontsize=9)
        axes[row, 0].set_ylabel(name)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_results(results, path):
    fig, axes = plt.subplots(2, len(CORRUPTIONS), figsize=(14, 7), sharex=True)
    for col, name in enumerate(CORRUPTIONS):
        for method, curves in results[name].items():
            sev = range(len(curves["dice"]))
            axes[0, col].plot(sev, curves["dice"], "o-", ms=4, label=method)
            axes[1, col].plot(sev, curves["uncertainty"], "o-", ms=4, label=method)
        axes[0, col].set_title(name)
        axes[1, col].set_xlabel("severity (0 = original)")
    axes[0, 0].set_ylabel("mean Dice  (higher = better)")
    axes[1, 0].set_ylabel("mean uncertainty\n(should RISE as Dice falls)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_models(ROOT / "checkpoints", device)
    methods = {n: f for n, f in METHODS.items() if n != "ensemble" or len(models) >= 2}

    _, _, test_ds, _ = get_datasets()
    x = to_tensor(test_ds.images[test_ds.indices])
    masks = test_ds.masks[test_ds.indices]

    OUT.mkdir(parents=True, exist_ok=True)
    plot_examples(x, OUT / "corruption_examples.png")

    torch.manual_seed(0)
    results = {}
    for name, (fn, levels) in CORRUPTIONS.items():
        results[name] = {m: {"levels": levels, "dice": [], "uncertainty": []} for m in methods}
        for level in levels:
            xc = fn(x, level)
            line = f"{name:15s} {level:>4}:"
            for method, predict in methods.items():
                prob = predict(models, xc, device)
                dice, _ = dice_iou(prob > 0.5, masks)
                unc = image_uncertainty(prob)
                results[name][method]["dice"].append(float(dice.mean()))
                results[name][method]["uncertainty"].append(float(unc.mean()))
                line += f"  {method} Dice {dice.mean():.3f} unc {unc.mean():.3f} |"
            print(line)

    plot_results(results, OUT / "robustness.png")
    (OUT / "robustness.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
