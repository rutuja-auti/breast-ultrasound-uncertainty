"""
Loss function and evaluation metrics.

Dice score (the standard metric for medical segmentation):

        2 * |prediction ∩ truth|
 Dice = ------------------------      1.0 = perfect overlap, 0.0 = no overlap
        |prediction| + |truth|

IoU (Intersection over Union, a.k.a. Jaccard) is similar but stricter:
 IoU = |prediction ∩ truth| / |prediction ∪ truth|
"""

import numpy as np
import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    """Binary cross-entropy + soft Dice loss.

    BCE judges every pixel on its own. Tumours often cover only a small part of
    the image, so a model could get low BCE by predicting "background"
    everywhere. The Dice term measures overlap with the tumour directly, so it
    penalises that. Using both is a common and robust choice.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target, eps=1.0):
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(1, 2, 3))
        total = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2 * inter + eps) / (total + eps)
        return self.bce(logits, target) + (1 - dice.mean())


def dice_iou(pred, target, eps=1e-7):
    """Per-image Dice and IoU for binary masks. Inputs: (N, H, W) arrays of 0/1."""
    pred = pred.reshape(len(pred), -1).astype(bool)
    target = target.reshape(len(target), -1).astype(bool)
    inter = (pred & target).sum(1)
    union = (pred | target).sum(1)
    total = pred.sum(1) + target.sum(1)
    dice = (2 * inter + eps) / (total + eps)
    iou = (inter + eps) / (union + eps)
    return dice, iou


def expected_calibration_error(prob, target, n_bins=15):
    """Pixel-level Expected Calibration Error (ECE).

    A model is CALIBRATED if, among all pixels where it says "70% tumour",
    about 70% really are tumour. We group pixels into confidence bins and
    measure the average gap between confidence and actual accuracy.
    Lower ECE = more honest probabilities. Doctors need this: a confident
    wrong answer is far more dangerous than an uncertain one.

    Returns (ece, bin_confidence, bin_accuracy, bin_count) for plotting a
    reliability diagram.
    """
    prob = prob.ravel()
    target = target.ravel().astype(bool)
    pred = prob >= 0.5
    # Confidence = probability of the class the model actually chose.
    conf = np.where(pred, prob, 1 - prob)
    correct = pred == target

    edges = np.linspace(0.5, 1.0, n_bins + 1)
    bin_id = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    count = np.bincount(bin_id, minlength=n_bins)
    conf_sum = np.bincount(bin_id, weights=conf, minlength=n_bins)
    acc_sum = np.bincount(bin_id, weights=correct, minlength=n_bins)

    nonzero = count > 0
    bin_conf = np.where(nonzero, conf_sum / np.maximum(count, 1), np.nan)
    bin_acc = np.where(nonzero, acc_sum / np.maximum(count, 1), np.nan)
    ece = np.nansum(count / count.sum() * np.abs(bin_acc - bin_conf))
    return ece, bin_conf, bin_acc, count
