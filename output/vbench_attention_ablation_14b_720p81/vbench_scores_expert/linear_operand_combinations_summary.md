# SLA linear-attention operand sparsity: VBench summary

All methods use the same 8 prompts, conditioning images, unquantized 14B high/low checkpoints, 1280x720 resolution, 81 frames, four sampling steps, and seed 0. Higher is better. Mean is the unweighted mean of the six requested dimensions, not the official VBench total.

| Method | SC | BC | AQ | IQ | OC | MC | Mean | Delta vs SLA | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SLA | 0.966483 | 0.944287 | 0.627141 | 0.676846 | 0.243953 | 0.992271 | 0.741830 | 0 | — |
| K + Q | 0.966481 | **0.947109** | 0.628657 | **0.676186** | 0.242511 | 0.992331 | **0.742212** | +0.000382 | [-0.000841, +0.001673] |
| K + KV | 0.966234 | 0.940174 | 0.629399 | 0.675549 | **0.244321** | 0.992354 | 0.741338 | -0.000492 | [-0.001561, +0.000283] |
| V + Q | **0.966338** | 0.942101 | **0.630959** | 0.674244 | 0.243662 | 0.992374 | 0.741613 | -0.000217 | [-0.001524, +0.000988] |
| V + KV | 0.966018 | 0.941903 | 0.627543 | 0.675626 | 0.243802 | **0.992390** | 0.741214 | -0.000616 | [-0.001927, +0.000586] |

The bold values among sparse methods identify per-column point-estimate winners. Every confidence interval crosses zero, so the 8-video experiment does not establish a statistically reliable quality difference from SLA. K+Q is the best point estimate and the most defensible candidate for a larger multi-seed experiment.
