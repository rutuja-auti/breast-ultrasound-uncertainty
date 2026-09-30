"""
Train the U-Net on the BUSI breast ultrasound dataset.

    python src/train.py                 # one model, seed 0 (~5-10 min on an RTX 3060)
    python src/train.py --seed 3        # another model with a different random start
    python src/train.py --epochs 2      # quick test that everything works

Training 5 models with seeds 0-4 gives a DEEP ENSEMBLE (Lakshminarayanan et al.,
2017): same data, same architecture, but different random initialisation and
data order, so they make slightly different mistakes. Their disagreement is
another way to measure uncertainty. run_experiments.ps1 does this for you.

Saves:
    checkpoints/seed{N}.pt                 weights with the best validation Dice
    results/training/log_seed{N}.csv       loss and Dice for every epoch
    results/training/curves_seed{N}.png    plot of the above
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save plots to files without opening a window
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import get_datasets
from metrics import DiceBCELoss, dice_iou
from unet import UNet

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    dices = []
    for x, y in loader:
        logits = model(x.to(device))
        pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()[:, 0]
        d, _ = dice_iou(pred, y.numpy()[:, 0])
        dices.extend(d)
    return float(np.mean(dices))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds, val_ds, _, _ = get_datasets()
    # num_workers=0 keeps things simple on Windows; data is already in memory.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = UNet(dropout=args.dropout).to(device)
    loss_fn = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Cosine schedule: learning rate starts high and smoothly decays to ~0.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # Mixed precision: faster and uses less GPU memory, no accuracy loss here.
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    ckpt_path = ROOT / "checkpoints" / f"seed{args.seed}.pt"
    log_dir = ROOT / "results" / "training"
    ckpt_path.parent.mkdir(exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log = []
    best_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, enabled=use_amp):
                loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()

        train_loss = float(np.mean(losses))
        val_dice = validate(model, val_loader, device)
        log.append((epoch, train_loss, val_dice))

        marker = ""
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch},
                       ckpt_path)
            marker = "  <- best so far, saved"
        print(f"epoch {epoch:3d}/{args.epochs}  loss {train_loss:.4f}  "
              f"val Dice {val_dice:.4f}  ({time.time() - t0:.1f}s){marker}")

    with open(log_dir / f"log_seed{args.seed}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_dice"])
        w.writerows(log)

    epochs, tl, vd = zip(*log)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(epochs, tl)
    ax[0].set(title="Training loss", xlabel="epoch", ylabel="Dice + BCE loss")
    ax[1].plot(epochs, vd, color="tab:green")
    ax[1].set(title=f"Validation Dice (seed {args.seed})", xlabel="epoch", ylabel="Dice", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(log_dir / f"curves_seed{args.seed}.png", dpi=120)
    print(f"\nDone. Best validation Dice: {best_dice:.4f}. Saved to {ckpt_path.name}")


if __name__ == "__main__":
    main()
