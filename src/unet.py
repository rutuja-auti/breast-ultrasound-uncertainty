"""
U-Net, written from scratch.

Paper: Ronneberger, Fischer & Brox, "U-Net: Convolutional Networks for
Biomedical Image Segmentation", MICCAI 2015 - one of the most cited papers in
medical imaging.

Shape of the network (the "U"):

    input 1x256x256 (grayscale ultrasound)
      enc1  32 ──────────────────────────────► dec1  32 ─► 1x256x256 (tumour logit)
        enc2  64 ──────────────────────► dec2  64
          enc3  128 ─────────────► dec3  128
            enc4  256 ──────► dec4  256
                   bottleneck 512

  - The ENCODER (left) shrinks the image and learns "what" is there.
  - The DECODER (right) grows it back and learns "where" exactly it is.
  - SKIP CONNECTIONS (arrows) copy fine detail from encoder to decoder so the
    predicted outline is sharp.

The one addition to the original U-Net is DROPOUT in the decoder. During
training it regularises the model. At test time we deliberately keep it ON and
run the model many times (Monte Carlo dropout, Gal & Ghahramani, ICML 2016):
where the predictions disagree, the model is uncertain. See evaluate.py.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(3x3 conv -> BatchNorm -> ReLU) twice, optionally followed by dropout."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            # Dropout2d drops whole feature maps, which works better for conv nets.
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32, dropout=0.2):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]  # 32, 64, 128, 256, 512

        self.enc1 = DoubleConv(in_ch, c[0])
        self.enc2 = DoubleConv(c[0], c[1])
        self.enc3 = DoubleConv(c[1], c[2])
        self.enc4 = DoubleConv(c[2], c[3])
        self.bottleneck = DoubleConv(c[3], c[4], dropout)
        self.pool = nn.MaxPool2d(2)

        # Each decoder step: upsample x2, concatenate the skip connection, convolve.
        self.up4 = nn.ConvTranspose2d(c[4], c[3], 2, stride=2)
        self.dec4 = DoubleConv(c[3] * 2, c[3], dropout)
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.dec3 = DoubleConv(c[2] * 2, c[2], dropout)
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.dec2 = DoubleConv(c[1] * 2, c[1], dropout)
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec1 = DoubleConv(c[0] * 2, c[0])

        # 1x1 conv turns 32 feature maps into 1 map of logits (before sigmoid).
        self.head = nn.Conv2d(c[0], out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)                     # 32  x 256 x 256
        e2 = self.enc2(self.pool(e1))         # 64  x 128 x 128
        e3 = self.enc3(self.pool(e2))         # 128 x 64 x 64
        e4 = self.enc4(self.pool(e3))         # 256 x 32 x 32
        b = self.bottleneck(self.pool(e4))    # 512 x 16 x 16

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)                  # 1 x 256 x 256 logits


def enable_mc_dropout(model):
    """Put the model in eval mode, but switch dropout layers back ON.

    BatchNorm must stay in eval mode (use its saved statistics); only dropout
    should stay random so that each forward pass gives a different sample.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout2d):
            m.train()


def load_models(checkpoint_dir, device):
    """Load every checkpoints/seed*.pt, sorted by seed (seed 0 first)."""
    paths = sorted(checkpoint_dir.glob("seed*.pt"), key=lambda p: int(p.stem[4:]))
    if not paths:
        raise SystemExit(f"No checkpoints found in {checkpoint_dir}. Run: python src/train.py")
    models = []
    for path in paths:
        ckpt = torch.load(path, map_location=device)
        model = UNet(dropout=ckpt["args"]["dropout"]).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models.append(model)
    print(f"Loaded {len(models)} model(s): {', '.join(p.stem for p in paths)}")
    return models


if __name__ == "__main__":
    net = UNet()
    n_params = sum(p.numel() for p in net.parameters())
    out = net(torch.randn(2, 1, 256, 256))
    print(f"U-Net with {n_params / 1e6:.1f}M parameters, output shape {tuple(out.shape)}")
