# Wan2.2 14B attention ablation — VBench results

All methods use the same unquantized TurboWan2.2-I2V-A14B 720P high/low-noise checkpoints, 8 prompts and conditioning images, seed 0, 4 sampling steps, 1280×720 resolution, and 81 frames at 16 FPS. SLA uses `topk=0.1`. Q 4:8 pairwise divides every eight Q features into four adjacent pairs, scores each pair by the sum of its two absolute values, and retains the two highest-scoring pairs.

| Method | Subject consistency | Background consistency | Motion smoothness | Dynamic degree | Aesthetic quality | Imaging quality | Six-dimension mean* |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 0.954199 | 0.932174 | 0.988122 | 0.625000 | 0.619796 | 0.657766 | 0.796176 |
| SLA | 0.966483 | 0.944287 | 0.992271 | 0.375000 | 0.627141 | 0.676846 | 0.763671 |
| SLA + Q 2:4 | 0.967604 | 0.948794 | 0.993319 | 0.250000 | 0.627044 | 0.660583 | 0.741224 |
| SLA + Q 4:8 pairwise | 0.968825 | 0.947749 | 0.992036 | 0.375000 | 0.611857 | 0.675154 | 0.761770 |
| SLA + K 2:4 | 0.964187 | 0.947150 | 0.993234 | 0.250000 | 0.631279 | 0.657058 | 0.740485 |
| SLA + K 4:8 pairwise | 0.966901 | 0.944116 | 0.992439 | 0.250000 | 0.616120 | 0.675899 | 0.740913 |
| SLA + Q 2:4 share-index=2 | 0.964859 | 0.949801 | 0.992964 | 0.250000 | 0.624470 | 0.653317 | 0.739235 |
| SLA + K 2:4 share-index=2 | 0.964040 | 0.948154 | 0.992919 | 0.250000 | 0.627131 | 0.651485 | 0.738955 |

\* The final column is an unweighted arithmetic mean of the six reported custom-input dimensions. It is not the official VBench Total score, which requires the complete benchmark protocol and all dimensions.

## Change relative to Original

| Method | Subject | Background | Smoothness | Dynamic | Aesthetic | Imaging | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| SLA | +0.012285 | +0.012113 | +0.004148 | -0.250000 | +0.007345 | +0.019080 | -0.032505 |
| SLA + Q 2:4 | +0.013405 | +0.016620 | +0.005196 | -0.375000 | +0.007248 | +0.002817 | -0.054952 |
| SLA + Q 4:8 pairwise | +0.014627 | +0.015575 | +0.003914 | -0.250000 | -0.007939 | +0.017388 | -0.034406 |
| SLA + K 2:4 | +0.009989 | +0.014977 | +0.005111 | -0.375000 | +0.011483 | -0.000708 | -0.055691 |
| SLA + K 4:8 pairwise | +0.012702 | +0.011942 | +0.004317 | -0.375000 | -0.003675 | +0.018133 | -0.055264 |
| SLA + Q 2:4 share-index=2 | +0.010660 | +0.017627 | +0.004841 | -0.375000 | +0.004675 | -0.004449 | -0.056941 |
| SLA + K 2:4 share-index=2 | +0.009841 | +0.015980 | +0.004796 | -0.375000 | +0.007335 | -0.006282 | -0.057221 |

All 64 generated files were validated as 1280×720, 81 frames, 16 FPS, with a decodable final frame.
