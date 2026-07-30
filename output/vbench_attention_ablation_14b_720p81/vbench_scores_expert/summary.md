# Wan2.2 14B attention and activation-sparsity experiments

## Evaluation protocol

All 12 methods use the same `vbench_prompt.json` test set (8 prompts and 8
conditioning images), unquantized TurboWan2.2-I2V-A14B high-noise/low-noise
checkpoints, 1280×720 output, 81 frames, 16 fps, four sampling steps, and seed
0. Higher is better for every metric.

`Mean` is the unweighted arithmetic mean of the six requested VBench
dimensions; it is not the official VBench total score. `Delta vs SLA` is the
difference between this mean and plain SLA. Paired bootstrap confidence
intervals were computed only for the four linear-operand combination
experiments; `—` means that no interval was computed, not that the difference
is statistically significant.

Metric abbreviations:

- SC: subject consistency
- BC: background consistency
- AQ: aesthetic quality
- IQ: imaging quality
- OC: overall consistency (ViCLIP)
- MC: motion smoothness

## Complete results

Bold values are the global point-estimate winners across all 12 methods.

| Category | Method | SC | BC | AQ | IQ | OC | MC | Mean | Delta vs SLA | Paired bootstrap 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | Original | 0.954199 | 0.932174 | 0.619796 | 0.657766 | **0.254648** | 0.988122 | 0.734451 | -0.007379 | — |
| Baseline | SLA | 0.966483 | 0.944287 | 0.627141 | **0.676846** | 0.243953 | 0.992271 | 0.741830 | 0 | — |
| SLA input Q/K | SLA + Q 2:4 | 0.967604 | 0.948794 | 0.627044 | 0.660583 | 0.241054 | **0.993319** | 0.739733 | -0.002097 | — |
| SLA input Q/K | SLA + Q 4:8 pairwise | **0.968825** | 0.947749 | 0.611857 | 0.675154 | 0.238930 | 0.992036 | 0.739092 | -0.002738 | — |
| SLA input Q/K | SLA + K 2:4 | 0.964187 | 0.947150 | **0.631279** | 0.657058 | 0.245936 | 0.993234 | 0.739807 | -0.002023 | — |
| SLA input Q/K | SLA + K 4:8 pairwise | 0.966901 | 0.944116 | 0.616120 | 0.675899 | 0.239745 | 0.992439 | 0.739203 | -0.002627 | — |
| SLA input Q/K | SLA + Q 2:4 share-index=2 | 0.964859 | **0.949801** | 0.624470 | 0.653317 | 0.240985 | 0.992964 | 0.737733 | -0.004097 | — |
| SLA input Q/K | SLA + K 2:4 share-index=2 | 0.964040 | 0.948154 | 0.627131 | 0.651485 | 0.244802 | 0.992919 | 0.738088 | -0.003742 | — |
| Linear two-GEMM | SLA + K(token) + Q(feature) 2:4 | 0.966481 | 0.947109 | 0.628657 | 0.676186 | 0.242511 | 0.992331 | **0.742212** | **+0.000382** | [-0.000841, +0.001673] |
| Linear two-GEMM | SLA + K(token) + KV(feature) 2:4 | 0.966234 | 0.940174 | 0.629399 | 0.675549 | 0.244321 | 0.992354 | 0.741338 | -0.000492 | [-0.001561, +0.000283] |
| Linear two-GEMM | SLA + V(token) + Q(feature) 2:4 | 0.966338 | 0.942101 | 0.630959 | 0.674244 | 0.243662 | 0.992374 | 0.741613 | -0.000217 | [-0.001524, +0.000988] |
| Linear two-GEMM | SLA + V(token) + KV(feature) 2:4 | 0.966018 | 0.941903 | 0.627543 | 0.675626 | 0.243802 | 0.992390 | 0.741214 | -0.000616 | [-0.001927, +0.000586] |

## Findings

1. `SLA + K(token) + Q(feature) 2:4` has the highest six-metric point-estimate
   mean, 0.742212. It is 0.000382 above plain SLA, but its paired-bootstrap 95%
   CI crosses zero, so this experiment does not establish a statistically
   reliable improvement.
2. Plain SLA is the second-highest method by mean (0.741830) and retains the
   best imaging-quality score. All four linear two-GEMM combinations are within
   0.000616 mean score of SLA.
3. Among the six single Q/K sparse variants, K 2:4 has the highest mean
   (0.739807), followed by Q 2:4 (0.739733). Both remain below plain SLA.
4. The global metric winners are: SC = Q 4:8 pairwise, BC = Q share-index=2,
   AQ = K 2:4, IQ = SLA, OC = Original, and MC = Q 2:4. No single method wins
   every dimension.
5. Every SLA-based route has a higher six-metric mean than Original, while
   Original retains the strongest OC score. This illustrates why the mean and
   all six component metrics should be reported together.
6. This is a small, single-seed, 8-video experiment. Differences in the third
   or fourth decimal place are directional evidence only. The K+Q combination
   is the best candidate for a larger multi-seed evaluation, not yet a proven
   quality improvement.

## Sparsity-location clarification

- `SLA input Q/K` methods sparsify the Q or K tensor supplied to the complete
  SLA attention module, so both sparse-block attention and linear attention see
  the modified operand.
- `Linear two-GEMM` methods affect only the SLA linear branch. For `K.T @ V`, K
  or V is grouped along the token/reduction dimension. For `Q @ KV`, Q or KV is
  grouped along the head-feature/reduction dimension. The normalization
  denominator stays dense.
