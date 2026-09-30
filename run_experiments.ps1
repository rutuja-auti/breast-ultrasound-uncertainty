# Runs the whole project from start to finish.
#   Usage (from the D:\MS folder):   .\run_experiments.ps1
# Takes roughly 30-60 minutes on an RTX 3060 (5 models x 60 epochs, then evaluation).

$ErrorActionPreference = "Stop"
$python = ".venv\Scripts\python.exe"

& $python src/data.py
if (-not $?) { exit 1 }

# Train 5 U-Nets with different random seeds: seed 0 is the "single" model,
# all 5 together form the deep ensemble.
foreach ($seed in 0..4) {
    Write-Host "`n===== Training model with seed $seed =====" -ForegroundColor Cyan
    & $python src/train.py --seed $seed
    if (-not $?) { exit 1 }
}

Write-Host "`n===== Experiment 1: comparing uncertainty methods =====" -ForegroundColor Cyan
& $python src/evaluate.py
if (-not $?) { exit 1 }

Write-Host "`n===== Experiment 2: robustness to poor image quality =====" -ForegroundColor Cyan
& $python src/robustness.py
if (-not $?) { exit 1 }

Write-Host "`nAll done. Results are in the results\ folder." -ForegroundColor Green
