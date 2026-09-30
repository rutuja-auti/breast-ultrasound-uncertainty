"""
Four ways to get a prediction AND an uncertainty estimate from our U-Net(s).

Every method returns the averaged tumour probability for each pixel. The
uncertainty map is then the ENTROPY of that probability (see entropy() below),
so all four methods are compared in exactly the same way.

  1. single      One U-Net, one forward pass. Uncertainty comes only from how
                 close its sigmoid output is to 0.5. This is the BASELINE that
                 the other methods must beat.

  2. mc_dropout  One U-Net, dropout kept ON at test time, 20 forward passes.
                 (Gal & Ghahramani, ICML 2016)

  3. tta         Test-Time Augmentation: one U-Net, but the INPUT is changed
                 (flipped, brightened/darkened) and the answers are averaged.
                 Captures uncertainty from the image itself. (Wang et al.,
                 Neurocomputing 2019)

  4. ensemble    5 U-Nets trained with different random seeds, averaged.
                 Often the strongest method, but 5x the training cost.
                 (Lakshminarayanan et al., NeurIPS 2017)

Trade-offs to discuss in your dissertation:
  - Training cost:  ensemble 5x, the others 1x.
  - Test-time cost: single 1 pass, TTA 6, ensemble 5, MC dropout 20.
"""

import numpy as np
import torch

from data import normalize
from unet import enable_mc_dropout

MC_SAMPLES = 20
# TTA variants: (horizontal flip?, gamma). Gamma < 1 brightens, > 1 darkens.
TTA_VARIANTS = [(flip, gamma) for flip in (False, True) for gamma in (0.8, 1.0, 1.25)]


def entropy(prob):
    """Binary entropy, scaled to [0, 1]: 0 = certain, 1 = completely unsure (p = 0.5)."""
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2)


@torch.no_grad()
def forward(model, x, device, batch_size=16):
    """Run model on images x (N, 1, H, W) in [0, 1]; return probabilities (N, H, W)."""
    out = []
    for i in range(0, len(x), batch_size):
        xb = normalize(x[i:i + batch_size]).to(device)
        out.append(torch.sigmoid(model(xb))[:, 0].cpu())
    return torch.cat(out).numpy()


def predict_single(models, x, device):
    model = models[0]
    model.eval()
    return forward(model, x, device)


def predict_mc_dropout(models, x, device, samples=MC_SAMPLES):
    model = models[0]
    enable_mc_dropout(model)
    total = sum(forward(model, x, device) for _ in range(samples))
    model.eval()
    return total / samples


def predict_tta(models, x, device):
    model = models[0]
    model.eval()
    total = 0
    for flip, gamma in TTA_VARIANTS:
        xi = x ** gamma
        if flip:
            xi = xi.flip(-1)
        p = forward(model, xi, device)
        if flip:
            p = p[..., ::-1]  # flip the prediction back so it lines up with the original
        total = total + p
    return total / len(TTA_VARIANTS)


def predict_ensemble(models, x, device):
    for m in models:
        m.eval()
    return sum(forward(m, x, device) for m in models) / len(models)


METHODS = {
    "single": predict_single,
    "mc_dropout": predict_mc_dropout,
    "tta": predict_tta,
    "ensemble": predict_ensemble,
}


def image_uncertainty(prob, threshold=0.05):
    """One uncertainty number per image: mean entropy over the region where the
    model thinks there MIGHT be a tumour (prob > threshold). Averaging over the
    whole image would be dominated by the easy, certain background."""
    ent = entropy(prob)
    scores = []
    for h, p in zip(ent, prob):
        region = p > threshold
        scores.append(h[region].mean() if region.any() else h.mean())
    return np.array(scores)
