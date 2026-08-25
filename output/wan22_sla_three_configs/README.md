# Wan2.2 SLA 2:4 Configuration Comparison

All three configurations use the same eight Wan2.2 I2V prompts and first-frame images, unquantized high/low A14B checkpoints, 720p, 81 frames, four sampling steps, seed 0, and two-GPU distributed inference.

## 01 - SLA W16A16 dense baseline

- SLA attention enabled.
- No activation sparsity, no weight quantization, no HiF4.
- `videos/`, `audit/`, and `logs/` are copied from `output/hif4_w4a4_full_14b_720p81/sla`.

## 03 - SLA Q 2:4 with Qwen3 weight-norm scoring

- SLA attention enabled across all layers.
- Q-only 2:4 activation sparsity in both SLA branches.
- Qwen3 `ActivateSparseHW_inshare`-style weight-norm scoring: mask selection uses `abs(activation) * input-channel L2 norm of the downstream weight`; retained activation values are not rescaled.
- No weight quantization and no HiF4.
- `videos/`, `audit/`, and `logs/` are copied from `output/vbench_weight_norm_q2to4_14b_720p81/sla_q_2to4_weight_norm`.

The third configuration is implemented by `--sla_q_2to4_weight_norm`. Its audit JSON includes `q_2to4_weight_norm: true`.

## 04 - SLA K 2:4 with Qwen3 weight-norm scoring

- SLA attention enabled across all layers.
- K-only 2:4 activation sparsity in both SLA branches; Q remains dense.
- Sparse-QK branch: K is grouped along its feature dimension and scored using the L2 norm of Q over query tokens.
- Linear K.T@V branch: K.T is grouped along its token reduction dimension and scored using the corresponding V weight norm.
- Weight norms affect mask selection only; retained K values are not rescaled.
- No weight quantization and no HiF4.
- `videos/`, `audit/`, and `logs/` are copied from `output/vbench_weight_norm_k2to4_14b_720p81/sla_k_2to4_weight_norm`.

This configuration is implemented by `--sla_k_2to4_weight_norm`. Its audit JSON includes `k_2to4_weight_norm: true`.

## 05 - SLA K 2:4 weight norm fallback6

- Same SLA K-only Qwen3 weight-norm configuration as 04, using the fallback6 experimental path.
- The eight videos, audits, and logs are copied from `output/vbench_weight_norm_k2to4_fallback6_14b_720p81/sla_k_2to4_weight_norm_fallback6`.
- VBench results are under `vbench_scores/05_sla_k2to4_weight_norm_fallback6/`.

## 06 - SLA K 2:4 weight norm edge3x2

- Same K-only weight-norm 2:4 setup, with dense fallback at 0-based block indices `[0, 1, 2, 37, 38, 39]`.
- This replaces the uniformly distributed fallback indices `[3, 9, 15, 21, 27, 33]` used by configuration 05.
- The eight videos, audits, and logs are copied from `output/vbench_weight_norm_k2to4_edge3x2_14b_720p81/sla_k_2to4_weight_norm_edge3x2`.
- VBench results are under `vbench_scores/06_sla_k2to4_weight_norm_edge3x2/`.

## 07 - SLA K 2:4 RPQ + weight norm fallback6

- Uses the uniformly distributed dense fallback indices `[3, 9, 15, 21, 27, 33]`.
- Each remaining SLA layer calibrates a K-feature permutation on its first forward call using 256 sampled tokens and Top-50% channel co-occurrence conflicts.
- Qwen3-style balanced anti-clustering separates frequently co-important channels into different groups of four.
- The permutation is frozen for later timesteps and applied to Q/K together only in sparse-QK; linear K.T@V retains weight-normal token 2:4 without feature permutation.
- VBench results are under `vbench_scores/07_sla_k2to4_weight_norm_rpq_fallback6/`.

## 08 - SLA HiFloat8 W8A8 dense baseline

- SLA attention remains dense in all 40 layers; all activation-sparsity and dense-fallback modes are disabled.
- All Wan transformer-block `Linear` weights and activations use HiFloat8 round-to-nearest QDQ, except SLA's internal `proj_l`.
- VAE, text encoder, normalization layers, and non-Linear operations retain their original precision.
- The experiment is a numerical QDQ simulation with BF16/FP32 accumulation, not a HiFloat8 hardware-kernel throughput benchmark.
- The eight videos, audits, and logs are copied from `output/vbench_hif8_w8a8_dense_14b_720p81/sla_hif8_w8a8_dense`.
- VBench results are under `vbench_scores/08_sla_hif8_w8a8_dense/`.

## VBench summary

The six-metric evaluation uses 8 videos, 1280x720, 81 frames, 4 steps, and seed 0. Mean is the unweighted mean of SC, BC, AQ, IQ, OC, and MC.

| Configuration | SC | BC | AQ | IQ | OC | MC | Mean | Delta vs SLA dense |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SLA W16A16 dense | 0.964406 | 0.941705 | 0.620544 | 0.675906 | 0.245056 | 0.992412 | 0.740005 | 0.000000 |
| SLA Q-only 2:4, Qwen3 weight norm | 0.963627 | 0.942315 | 0.616410 | 0.669618 | 0.241286 | 0.992648 | 0.737651 | -0.002354 |
| SLA K-only 2:4, Qwen3 weight norm | 0.962949 | 0.946915 | 0.619949 | 0.659748 | 0.244670 | 0.993099 | 0.737888 | -0.002117 |
| SLA K-only 2:4, Qwen3 weight norm, fallback6 | 0.967805 | 0.947791 | 0.621488 | 0.660902 | 0.244973 | 0.993122 | 0.739347 | -0.000658 |
| SLA K-only 2:4, Qwen3 weight norm, edge3x2 | 0.955417 | 0.940486 | 0.609924 | 0.663634 | 0.238483 | 0.991211 | 0.733192 | -0.006813 |
| SLA K-only 2:4, RPQ + weight norm, fallback6 | 0.968093 | 0.947144 | 0.615738 | 0.672161 | 0.243091 | 0.992632 | 0.739810 | -0.000195 |
| SLA HiFloat8 W8A8 dense | 0.963413 | 0.945274 | 0.624659 | 0.674593 | 0.242997 | 0.992248 | 0.740531 | +0.000526 |

Detailed files: `vbench_summary.md`, `vbench_summary.csv`, `vbench_summary.json`; raw per-metric outputs are under `vbench_scores/`.

## Selected six-experiment summary

The six configurations requested for this comparison are extracted into `vbench_selected_six_summary.md`, `vbench_selected_six_summary.csv`, and `vbench_selected_six_summary.json`. Ranking by six-metric mean:

1. SLA W16A16 dense: 0.740005
2. SLA K-only 2:4, RPQ + weight norm, fallback6: 0.739810
3. SLA K-only 2:4, weight norm, fallback6: 0.739347
4. SLA K-only 2:4, weight norm: 0.737888
5. SLA Q-only 2:4, weight norm: 0.737651
6. SLA K-only 2:4, weight norm, edge3x2: 0.733192

Among the sparse configurations, RPQ + weight norm + fallback6 is closest to the W16A16 dense baseline, with a mean difference of -0.000195.
