| Method | Dice (95% CI) | IoU | Failures (Dice < 0.1) | ECE (lower = better) | Spearman rho (more negative = better) | Dice gain @ 20% referral | Images better / worse than single | p vs single |
|---|---|---|---|---|---|---|---|---|
| single | 0.716 (0.666-0.764) | 0.609 | 5 | 0.0364 | -0.86 | +0.096 | - | - |
| mc_dropout | 0.713 (0.663-0.762) | 0.606 | 5 | 0.0340 | -0.87 | +0.098 | 29 / 47 | 0.0118 |
| tta | 0.705 (0.648-0.758) | 0.605 | 7 | 0.0313 | -0.87 | +0.105 | 48 / 39 | 0.596 |
| ensemble | 0.716 (0.654-0.772) | 0.628 | 11 | 0.0245 | -0.87 | +0.093 | 66 / 25 | 0.00204 |

| Method | benign Dice | benign uncertainty | malignant Dice | malignant uncertainty |
|---|---|---|---|---|
| single | 0.716 | 0.341 | 0.715 | 0.371 |
| mc_dropout | 0.714 | 0.377 | 0.711 | 0.413 |
| tta | 0.708 | 0.390 | 0.697 | 0.428 |
| ensemble | 0.707 | 0.421 | 0.734 | 0.477 |
