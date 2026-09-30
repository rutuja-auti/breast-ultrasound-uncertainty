# Trustworthy Breast Tumour Segmentation in Ultrasound
### Comparing uncertainty estimation methods for safer AI-assisted diagnosis

This is a final-year-project-level study. It trains a U-Net to outline breast tumours in ultrasound images. It then compares **four ways of making the AI say how sure it is**, and tests whether that uncertainty is useful:
- Does it flag the model's mistakes?
- Could it decide which scans a radiologist should double-check?
- Does it warn us when image quality is poor?

> ⚠️ Research and learning project only. **Not** a medical device and not for real diagnosis.

---

## Contents
1. [The problem, in plain English](#1-the-problem-in-plain-english)
2. [Research questions](#2-research-questions)
3. [Key ideas, explained simply](#3-key-ideas-explained-simply)
4. [The dataset](#4-the-dataset-busi)
5. [Project structure](#5-project-structure)
6. [How to run it](#6-how-to-run-it-windows-step-by-step)
7. [Understanding your results](#7-understanding-your-results)
8. [Limitations](#8-limitations-be-honest-about-these)
9. [Extensions](#9-extensions-make-it-your-own)
10. [References](#10-references)

---

## 1. The problem, in plain English

- **Breast cancer** is the most commonly diagnosed cancer in women worldwide. Finding it early saves lives.
- **Ultrasound** is cheap, portable, has no radiation, and works well on dense breast tissue. It's widely used, especially in places where mammography machines are rare.
- But ultrasound images are **hard to read**. They're grainy (called *speckle noise*), tumour edges are blurry, and quality depends heavily on the machine and the operator.
- AI can help by outlining suspicious areas automatically. But an AI that is **confidently wrong** could make a doctor miss a cancer.
- A safer AI **knows when it doesn't know**. It can say *"I'm unsure about this scan, please check it carefully."*

**This project asks: which way of measuring AI uncertainty works best for breast ultrasound, and is it actually useful?**

```
                         ┌─► tumour outline (segmentation)
 ultrasound image ─► U-Net(s) ─┼─► uncertainty map ("I'm unsure here")
                         └─► refer to radiologist?  yes / no
```

---

## 2. Research questions

| # | Question | How we test it | Script |
|---|---|---|---|
| RQ1 | Do uncertainty methods also change **accuracy**? | Dice / IoU with 95% confidence intervals + Wilcoxon test | `evaluate.py` |
| RQ2 | Which method gives the most **honest probabilities**? | Expected Calibration Error, reliability diagram | `evaluate.py` |
| RQ3 | Does uncertainty **flag the mistakes**? | Spearman correlation (uncertainty vs Dice) + referral experiment | `evaluate.py` |
| RQ4 | Is the model worse or less sure on **malignant** tumours? | Dice and uncertainty per class | `evaluate.py` |
| RQ5 | How much does accuracy drop on **poor-quality** images? | Dice under noise / blur / low resolution | `robustness.py` |
| RQ6 | Does uncertainty **rise when quality drops**? | Uncertainty under the same corruptions | `robustness.py` |

**Hypotheses** (write these *before* seeing results; then report honestly whether they held):
- H1: All three uncertainty methods are better calibrated than the single model. The ensemble is the best.
- H2: Uncertainty correlates negatively with Dice, so referring uncertain cases raises the Dice of the cases the AI keeps.
- H3: Malignant tumours, which have irregular edges, get lower Dice and higher uncertainty than benign ones.
- H4: As image quality degrades, Dice falls and uncertainty rises.

---

## 3. Key ideas, explained simply

### Image segmentation
- **Classification** answers "is there a tumour?" (yes/no).
- **Segmentation** answers "which exact pixels are tumour?". The output is a **mask**, a black-and-white image where tumour = white.
- Doctors need the outline to measure size and shape. Irregular, spiky shapes are a warning sign of cancer.

### U-Net (the model), in [src/unet.py](src/unet.py)
U-Net is the most famous network in medical imaging (Ronneberger et al., 2015):
- **Left side (encoder):** shrinks the image step by step and learns *what* is there.
- **Right side (decoder):** grows it back and learns *where* exactly the tumour is.
- **Skip connections:** pass fine details across, so the outline is sharp.

We add **dropout** layers, which randomly switch off some neurons. This is needed for MC dropout (below).

### Training, in [src/train.py](src/train.py)
1. The model guesses a mask for an image.
2. The **loss** measures how wrong the guess is. We use Dice loss + binary cross-entropy.
3. The **optimizer** (AdamW) nudges the ~7.8 million weights to reduce the loss.
4. Repeat over all training images: that's one **epoch**. We run 60 epochs.

**Data augmentation** creates realistic variety: left-right flips, brightness (gamma) changes and extra speckle noise. We do *not* flip upside-down, because the ultrasound probe is always at the top of the image.

### Train / validation / test split, in [src/data.py](src/data.py)
647 images, split once with a fixed seed and **stratified** (same benign:malignant ratio in every split):

| Split | ~Images | Used for |
|---|---|---|
| Train | 453 | Learning the weights |
| Validation | 97 | Choosing the best epoch |
| Test | 97 | Final results only. **Never** used for any decision |

### Dice score: accuracy, in [src/metrics.py](src/metrics.py)
```
Dice = 2 × overlap / (predicted area + true area)       1.0 = perfect, 0.0 = no overlap
```
Why not simple pixel accuracy? If the tumour covers 5% of the image, predicting "no tumour" everywhere scores 95% accuracy but is useless. Dice doesn't have that problem.

### The four uncertainty methods, in [src/uncertainty.py](src/uncertainty.py)

| Method | Idea | Training cost | Test cost |
|---|---|---|---|
| **Single** (baseline) | One model. Uncertainty = how close its output is to 50%. | 1× | 1 pass |
| **MC dropout** | Keep dropout ON at test time and run 20 times. Disagreement = uncertainty. | 1× | 20 passes |
| **Test-time augmentation (TTA)** | Show the model slightly changed versions of the image (flipped, brighter, darker) and compare answers. | 1× | 6 passes |
| **Deep ensemble** | Train 5 models from different random starts and compare their answers. | 5× | 5 passes |

For every method, the uncertainty map is the **entropy** of the averaged prediction: 0 = certain, 1 = totally unsure.

### Calibration: are probabilities honest?
If the model says "80% sure" for many pixels, about 80% of them should really be tumour.
- The **reliability diagram** plots confidence against real accuracy. A perfect model lies on the diagonal.
- **ECE (Expected Calibration Error)** is the average distance from the diagonal. **Lower = better.**

### The referral experiment
Imagine a hospital workflow: the AI handles the scans it's sure about and **refers the most uncertain ones to a radiologist**. If the uncertainty is meaningful, the scans the AI keeps should have a **higher** average Dice. We plot this for 0–50% referral.

### Robustness, in [src/robustness.py](src/robustness.py)
We make the test images worse on purpose, at 5 severity levels, with **speckle noise**, **blur** and **low resolution**. A trustworthy model's uncertainty should **go up** as its accuracy goes down.

### Statistics (why they matter)
- **95% bootstrap confidence interval:** we resample the test images 2000 times to see how much the mean Dice could vary. With only ~97 test images, a difference of 0.01 Dice may just be luck.
- **Wilcoxon signed-rank test:** compares two methods *on the same images* (a paired test). p < 0.05 means the difference is unlikely to be chance.

---

## 4. The dataset: BUSI

**Breast Ultrasound Images Dataset** (Al-Dhabyani et al., *Data in Brief*, 2020)
- 780 images from 600 women aged 25–75, collected in 2018 at Baheya Hospital, Cairo, Egypt.
- Three classes: **normal (133), benign (437), malignant (210)**. Each image comes with a radiologist-drawn tumour mask.
- **We use the 647 benign + malignant images.** Normal images have no tumour to outline (see Extensions).
- Some images contain 2+ tumours (extra `_mask_1.png` files). The code merges them into one mask.
- Downloaded from Kaggle: <https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset>

Later researchers have reported that BUSI contains some **duplicate images and questionable annotations**. Duplicates that land in both the training and test sets would inflate the results (this is called *data leakage*). Checking for this is an excellent extension and discussion point (see Extensions).

---

## 5. Project structure

```
MS/
├── README.md                ← you are here
├── requirements.txt
├── run_experiments.ps1      ← runs EVERYTHING (train 5 models, evaluate, robustness)
└── src/
    ├── data.py              ← download (Kaggle), resize, split, augment
    ├── unet.py              ← the U-Net model; loading saved models
    ├── metrics.py           ← loss, Dice/IoU, calibration (ECE)
    ├── uncertainty.py       ← the 4 uncertainty methods
    ├── train.py             ← trains one model (one seed)
    ├── evaluate.py          ← Experiment 1: RQ1–RQ4
    └── robustness.py        ← Experiment 2: RQ5–RQ6

Created when you run it:
├── data/                    ← the dataset + cached arrays + splits.json
├── checkpoints/seed0..4.pt  ← trained models
└── results/
    ├── training/            ← training curves per seed
    ├── evaluation/          ← results table, plots, metrics.json
    └── robustness/          ← robustness plots and numbers
```

**Read the code in this order:** `data.py` → `unet.py` → `metrics.py` → `train.py` → `uncertainty.py` → `evaluate.py` → `robustness.py`. Every file has comments explaining each part.

---

## 6. How to run it (Windows, step by step)

Open PowerShell (or the VS Code terminal) in `D:\MS`.

**Step 1: Activate the virtual environment** (an isolated box for this project's libraries):
```powershell
.venv\Scripts\Activate.ps1
```
You should see `(.venv)` at the start of the line. Setting up on a new computer?
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

**Step 2: Log in to Kaggle** (one-time; needs a free account at kaggle.com):
```powershell
kaggle auth login
```
A browser window opens; approve it.

**Step 3: Download and prepare the data:**
```powershell
python src/data.py
```
Expected: about `train: 453 | val: 97 | test: 97`, with the benign/malignant counts for each.

**Step 4: Run all experiments** (~30–60 min on the RTX 3060):
```powershell
.\run_experiments.ps1
```
Or step by step, to understand each part:
```powershell
python src/train.py --seed 0        # train one model (repeat for seeds 1, 2, 3, 4)
python src/evaluate.py              # Experiment 1
python src/robustness.py            # Experiment 2
```
Tip: run `python src/train.py --epochs 2` first as a quick test.

---

## 7. Understanding your results

### `results/evaluation/`
| File | Shows | Good sign |
|---|---|---|
| `results_table.md` | The main table, ready for your dissertation | – |
| `reliability_diagram.png` | Calibration of all 4 methods | Lines near the diagonal, low ECE |
| `referral_curve.png` | Dice on kept scans vs % referred | Lines rise; closer to the dashed "oracle" is better |
| `uncertainty_vs_dice.png` | One dot per test image (benign/malignant coloured) | Downward trend (negative ρ) |
| `examples.png` | Best and worst cases: image, truth, prediction, errors, uncertainty | Bright uncertainty where the errors are (red = false alarm, blue = missed tumour) |

### `results/robustness/`
| File | Shows | Good sign |
|---|---|---|
| `corruption_examples.png` | What each corruption looks like | Use in your Methods chapter |
| `robustness.png` | Top: Dice vs severity. Bottom: uncertainty vs severity | Uncertainty rises as Dice falls |

### My results
Test set: 98 images (66 benign, 32 malignant). 5 U-Nets trained with seeds 0–4. Best validation Dice per seed: 0.728–0.747.

| Method | Dice (95% CI) | IoU | Failures (Dice < 0.1) | ECE ↓ | Spearman ρ | Dice after referring 20% | Images better / worse vs single | p vs single |
|---|---|---|---|---|---|---|---|---|
| Single | 0.716 (0.666–0.764) | 0.609 | 5 | 0.0364 | −0.86 | 0.812 | – | – |
| MC dropout | 0.713 (0.663–0.762) | 0.606 | 5 | 0.0340 | −0.87 | 0.811 | 29 / 47 | 0.012 |
| TTA | 0.705 (0.648–0.758) | 0.605 | 7 | 0.0313 | −0.87 | 0.810 | 48 / 39 | 0.60 |
| Ensemble | 0.716 (0.654–0.772) | **0.628** | 11 | **0.0245** | −0.87 | 0.809 | **66 / 25** | 0.002 |

| Method | Benign Dice | Benign uncertainty | Malignant Dice | Malignant uncertainty |
|---|---|---|---|---|
| Single | 0.716 | 0.341 | 0.715 | 0.371 |
| MC dropout | 0.714 | 0.377 | 0.711 | 0.413 |
| TTA | 0.708 | 0.390 | 0.697 | 0.428 |
| Ensemble | 0.707 | 0.421 | 0.734 | 0.477 |

**Robustness:** mean uncertainty at the worst severity vs the original images

| Method | Speckle noise | Blur | Low resolution |
|---|---|---|---|
| Single | 0.351 → 0.417 (+19%) | 0.351 → 0.418 (+19%) | 0.351 → 0.414 (+18%) |
| MC dropout | 0.388 → 0.464 (+20%) | 0.388 → 0.456 (+18%) | 0.389 → 0.450 (+16%) |
| TTA | 0.403 → 0.521 (+29%) | 0.403 → 0.470 (+17%) | 0.403 → 0.469 (+16%) |
| Ensemble | 0.439 → 0.658 (**+50%**) | 0.439 → 0.566 (**+29%**) | 0.439 → 0.555 (**+26%**) |

### Key findings
1. **Accuracy (RQ1):** all four methods reach about the same mean Dice (0.705–0.716), with heavily overlapping confidence intervals. The ensemble improves **most** images (66 of 98) but fails completely on more (11 vs 5). Averaging 5 models that disagree about *where* the tumour is can leave no pixel above 50%. MC dropout's "significant" p-value reflects a tiny, consistent *decrease*. A p-value alone doesn't tell you the direction.
2. **Calibration (RQ2):** H1 supported. All methods beat the single model, and the ensemble cuts ECE by about a third (0.036 → 0.025). But the reliability diagram shows **every method is over-confident**: at 80% confidence, only about 60% of pixels are correct. ECE looks small because easy background pixels dominate it.
3. **Flagging mistakes (RQ3):** H2 supported. Uncertainty is strongly linked to error for all methods (ρ ≈ −0.87). Referring the 20% most uncertain scans raises Dice from ~0.71 to ~0.81. Surprisingly, the **single model is already as good as the others at 20%**. The ensemble only pulls ahead above ~25% referral (e.g. 0.91 vs 0.87 at 50%).
4. **Benign vs malignant (RQ4):** H3 partly supported. Uncertainty is higher for malignant tumours with *every* method, but Dice is **not** lower (only 32 malignant test images, so treat this with caution).
5. **Robustness (RQ5–6):** H4 supported. Dice drops as quality worsens (0.72 → ~0.60 at the worst level). The **ensemble is the most robust** (best Dice at moderate damage) **and the most alert**: its uncertainty rises up to 50%, compared with about 19% for the single model. Under mild speckle noise, the single model's uncertainty doesn't rise at all while its accuracy falls. It fails silently.

**Overall:** on clean images, the extra uncertainty methods add surprisingly little over a single model's own confidence. Their value, especially the ensemble's, shows up when **image quality gets worse**, which is exactly when a warning matters most.

For context, published U-Net results on BUSI are often reported somewhere around 0.70–0.80 Dice, but setups differ a lot between papers. **Matching state-of-the-art is not the goal.** The comparison and analysis are what make this a strong project.

---

## 8. Limitations (be honest about these)

- **One dataset, one hospital, one scanner type.** Results may not transfer to other hospitals.
- **Small test set (~97 images)**, which is why we report confidence intervals.
- **2D still images**, while real ultrasound is a live video.
- **Possible duplicate images** in BUSI (see Section 4).
- **Simulated corruptions** only approximate real poor-quality scans.
- **The referral experiment is a simulation.** It is not a clinical study with real radiologists.

Discussing limitations clearly is a sign of research maturity, and examiners look for it.

---

## 9. Extensions (make it your own)

Pick one or two. They turn a good project into an excellent one.

1. **Duplicate / leakage check:** find near-identical images (e.g. by comparing downscaled images) and re-run with duplicates removed. Did the scores drop?
2. **Include normal images:** can the uncertainty tell you when there's *no* tumour and the model shows a false alarm?
3. **External validation:** test on a second breast ultrasound dataset from another hospital (e.g. the UDIAT "Dataset B"). Does uncertainty detect the domain shift?
4. **Temperature scaling:** a simple post-hoc calibration fix (Guo et al., 2017). How does it compare with the uncertainty methods?
5. **Stronger backbone:** swap the encoder for a pretrained ResNet and repeat the comparison.
6. **Benign vs malignant classification** from the predicted segmentation, e.g. using shape features.

---

## 10. References

- Al-Dhabyani, W. et al. (2020). Dataset of breast ultrasound images. *Data in Brief*, 28.
- Ronneberger, O., Fischer, P. & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*.
- Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian approximation. *ICML*.
- Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*.
- Wang, G. et al. (2019). Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation with convolutional neural networks. *Neurocomputing*.
- Guo, C. et al. (2017). On calibration of modern neural networks. *ICML*.
- Mehrtash, A. et al. (2020). Confidence calibration and predictive uncertainty estimation for deep medical image segmentation. *IEEE Transactions on Medical Imaging*.
- Jungo, A. & Reyes, M. (2019). Assessing reliability and challenges of uncertainty estimations for medical image segmentation. *MICCAI*.
