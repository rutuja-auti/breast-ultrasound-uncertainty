"""
Data pipeline for the BUSI breast ultrasound dataset.

BUSI (Breast Ultrasound Images): ultrasound scans of women aged 25-75, each
labelled benign / malignant / normal, with a hand-drawn mask outlining the
tumour. Paper: Al-Dhabyani et al., "Dataset of breast ultrasound images",
Data in Brief, 2020.

We use the 647 benign + malignant images (the 133 "normal" images have no
tumour to outline; including them is a possible extension, see README).

What this file does:
  1. Finds the dataset you downloaded from Kaggle (zip or unzipped folder).
  2. Converts every image to grayscale, resizes to IMG_SIZE x IMG_SIZE and
     caches everything in one .npz file so later runs start in seconds.
  3. Splits into train / validation / test with a fixed seed, STRATIFIED by
     class (so each split has the same benign:malignant ratio).
  4. Provides a PyTorch Dataset that applies random augmentations to training
     images.
"""

import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FOLDER_NAME = "Dataset_BUSI_with_GT"
CLASSES = ["benign", "malignant"]
IMG_SIZE = 256
SEED = 42

KAGGLE_ID = "aryashah2k/breast-ultrasound-images-dataset"

DOWNLOAD_HELP = f"""
Could not download BUSI from Kaggle: {{error}}

Option A - automatic (recommended), one-time login:
  1. Make a free account at https://www.kaggle.com if you don't have one.
  2. In this folder run:   .venv\\Scripts\\kaggle auth login
     (a browser window opens - approve it)
  3. Run this script again.
  (Alternative: generate a token at https://www.kaggle.com/settings/api and
   save it as {Path.home() / ".kaggle" / "access_token"})

Option B - manual:
  1. Open https://www.kaggle.com/datasets/{KAGGLE_ID}
  2. Click "Download" (you get archive.zip) and put the zip inside: {DATA_DIR}
  3. Run this script again (it will unzip it for you).
"""


def _search():
    found = [p for p in DATA_DIR.rglob(FOLDER_NAME) if p.is_dir()]
    return found[0] if found else None


def find_raw_dir():
    """Locate the Dataset_BUSI_with_GT folder. If it's missing, unzip a manually
    downloaded zip, or else download the dataset with the Kaggle API."""
    DATA_DIR.mkdir(exist_ok=True)
    if raw := _search():
        return raw
    for zip_path in DATA_DIR.glob("*.zip"):
        print(f"Unzipping {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA_DIR)
        if raw := _search():
            return raw
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()  # reads ~/.kaggle/kaggle.json
        print(f"Downloading {KAGGLE_ID} from Kaggle (~200 MB) ...")
        api.dataset_download_files(KAGGLE_ID, path=str(DATA_DIR), unzip=True, quiet=False)
    except (Exception, SystemExit) as e:  # the Kaggle library exits itself if not logged in
        reason = e if isinstance(e, Exception) else "not logged in to Kaggle"
        raise SystemExit(DOWNLOAD_HELP.format(error=reason))
    if raw := _search():
        return raw
    raise SystemExit(f"Downloaded, but no '{FOLDER_NAME}' folder found inside {DATA_DIR}")


def load_arrays():
    """Return (images, masks, labels, names) as numpy arrays, building the cache if needed.

    images: uint8, shape (N, IMG_SIZE, IMG_SIZE), grayscale
    masks:  uint8, shape (N, IMG_SIZE, IMG_SIZE), 0 (background) or 1 (tumour)
    labels: int, 0 = benign, 1 = malignant
    """
    cache = DATA_DIR / f"busi_{IMG_SIZE}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["images"], d["masks"], d["labels"], d["names"]

    raw_dir = find_raw_dir()
    images, masks, labels, names = [], [], [], []
    for label, cls in enumerate(CLASSES):
        # Image files look like "benign (12).png"; masks like "benign (12)_mask.png".
        img_paths = sorted(p for p in (raw_dir / cls).glob("*.png") if "_mask" not in p.name)
        print(f"Loading {len(img_paths)} {cls} images (one-time) ...")
        for img_path in img_paths:
            img = Image.open(img_path).convert("L")
            # A few images have 2+ tumours, stored as extra files "_mask_1.png" etc.
            # We merge all of them into one mask.
            mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            for mask_path in raw_dir.joinpath(cls).glob(f"{img_path.stem}_mask*.png"):
                m = Image.open(mask_path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
                mask |= (np.asarray(m) > 127).astype(np.uint8)
            images.append(np.asarray(img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)))
            masks.append(mask)
            labels.append(label)
            names.append(f"{cls}/{img_path.name}")

    images, masks = np.stack(images), np.stack(masks)
    labels, names = np.array(labels), np.array(names)
    np.savez_compressed(cache, images=images, masks=masks, labels=labels, names=names)
    return images, masks, labels, names


def get_splits(labels, val_frac=0.15, test_frac=0.15):
    """Fixed, reproducible, class-stratified split (saved to data/splits.json)."""
    split_file = DATA_DIR / "splits.json"
    if split_file.exists():
        return json.loads(split_file.read_text())
    rng = np.random.default_rng(SEED)
    splits = {"train": [], "val": [], "test": []}
    for label in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == label)).tolist()
        n_val, n_test = round(len(idx) * val_frac), round(len(idx) * test_frac)
        splits["test"] += idx[:n_test]
        splits["val"] += idx[n_test:n_test + n_val]
        splits["train"] += idx[n_test + n_val:]
    split_file.write_text(json.dumps(splits))
    return splits


def normalize(x):
    """Per-image standardisation to zero mean / unit variance.
    Ultrasound brightness depends heavily on the machine's gain setting, so each
    image is standardised on its own. Works on one image (1, H, W) or a batch
    (N, 1, H, W)."""
    dims = tuple(range(x.dim() - 3, x.dim()))
    mean = x.mean(dim=dims, keepdim=True)
    std = x.std(dim=dims, keepdim=True)
    return (x - mean) / (std + 1e-6)


def to_tensor(images):
    """uint8 array (N, H, W) -> float tensor (N, 1, H, W) in [0, 1], NOT normalised."""
    return torch.from_numpy(images).unsqueeze(1).float() / 255.0


class UltrasoundDataset(Dataset):
    """Returns (image, mask) tensors.

    image: float32 (1, H, W), normalised to zero mean / unit variance per image
    mask:  float32 (1, H, W), 0 or 1
    """

    def __init__(self, images, masks, labels, indices, augment=False):
        self.images = images
        self.masks = masks
        self.labels = labels
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        img = torch.from_numpy(self.images[j]).unsqueeze(0).float() / 255.0
        msk = torch.from_numpy(self.masks[j]).unsqueeze(0).float()

        if self.augment:
            img, msk = self._augment(img, msk)
        return normalize(img), msk

    @staticmethod
    def _augment(img, msk):
        """Random augmentations that are realistic for ultrasound.
        We only flip left-right: the probe is always at the TOP of the image, so
        flipping upside-down or rotating would create images that never occur."""
        if torch.rand(1) < 0.5:
            img, msk = img.flip(-1), msk.flip(-1)
        # Random gamma: brightens or darkens mid-tones, like different gain
        # settings. (Plain brightness scaling would be undone by normalize().)
        gamma = float(torch.empty(1).uniform_(0.7, 1.4))
        img = img.clamp(0, 1) ** gamma
        # Multiplicative "speckle" noise: the grainy texture typical of ultrasound.
        if torch.rand(1) < 0.5:
            img = img * (1 + 0.1 * torch.randn_like(img))
        return img.clamp(0, 1), msk


def get_datasets():
    images, masks, labels, names = load_arrays()
    splits = get_splits(labels)
    train = UltrasoundDataset(images, masks, labels, splits["train"], augment=True)
    val = UltrasoundDataset(images, masks, labels, splits["val"])
    test = UltrasoundDataset(images, masks, labels, splits["test"])
    return train, val, test, names


if __name__ == "__main__":
    # Run `python src/data.py` to prepare the data and check it looks right.
    train, val, test, _ = get_datasets()
    for name, ds in [("train", train), ("val", val), ("test", test)]:
        n_mal = int(ds.labels[ds.indices].sum())
        print(f"{name:5s}: {len(ds):3d} images ({len(ds) - n_mal} benign, {n_mal} malignant)")
    x, y = train[0]
    print("image", tuple(x.shape), "mask", tuple(y.shape),
          f"tumour covers {y.mean():.1%} of this image")
