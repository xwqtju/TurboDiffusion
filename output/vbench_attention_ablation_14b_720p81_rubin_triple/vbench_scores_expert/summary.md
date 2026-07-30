# GEMM-operand activation sparsity: VBench six-metric comparison

All methods use the same eight I2V prompts/images, unquantized TurboWan2.2 14B high/low checkpoints, 720p, 81 frames, four steps, and seed 0. The mean is an internal unweighted six-metric mean, not the official 16-dimension VBench total.

| Method | SC | BC | AQ | IQ | OC | MC | Mean | Delta vs SLA | Delta vs Original | Paired bootstrap 95% CI vs SLA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 0.954199 | 0.932174 | 0.619796 | 0.657766 | 0.254648 | 0.988122 | 0.734451 | -0.007379 | +0.000000 | — |
| SLA | 0.966483 | 0.944287 | 0.627141 | 0.676846 | 0.243953 | 0.992271 | 0.741830 | +0.000000 | +0.007379 | — |
| SLA + Q 2:4 | 0.967726 | 0.949321 | 0.621672 | 0.660394 | 0.242178 | 0.993132 | 0.739071 | -0.002760 | +0.004620 | [-0.007542, +0.001834] |
| SLA + Q 4:8 pairwise | 0.968811 | 0.948506 | 0.610451 | 0.673164 | 0.239856 | 0.992131 | 0.738820 | -0.003010 | +0.004369 | [-0.006793, +0.000056] |
| SLA + Q 2:4 share-index=2 | 0.966059 | 0.950908 | 0.620814 | 0.652090 | 0.241148 | 0.992890 | 0.737318 | -0.004512 | +0.002867 | [-0.009867, +0.000855] |
| SLA + K 2:4 | 0.964020 | 0.946878 | 0.631791 | 0.657707 | 0.244750 | 0.993063 | 0.739701 | -0.002129 | +0.005250 | [-0.007533, +0.003262] |
| SLA + K 4:8 pairwise | 0.966723 | 0.943353 | 0.616066 | 0.675600 | 0.239641 | 0.992473 | 0.738976 | -0.002854 | +0.004525 | [-0.008025, +0.000949] |
| SLA + K 2:4 share-index=2 | 0.964542 | 0.949546 | 0.627740 | 0.649581 | 0.243292 | 0.993091 | 0.737965 | -0.003865 | +0.003515 | [-0.010420, +0.002403] |
| SLA + K(QK) 2:4 + P(PV) 2:4 + K(linear) 2:4 | 0.963967 | 0.949355 | 0.627591 | 0.667828 | 0.245018 | 0.992283 | 0.741007 | -0.000823 | +0.006556 | [-0.007046, +0.004849] |

## Conclusions

1. Plain SLA has the highest six-metric mean, 0.741830. The strongest sparse variant is SLA + K(QK) 2:4 + P(PV) 2:4 + K(linear) 2:4 at 0.741007, -0.000823 versus SLA and +0.006556 versus Original.
2. All 7 GEMM-operand sparse variants score above Original and below SLA by point estimate. Every paired-bootstrap 95% interval versus SLA crosses zero, so this eight-video, single-seed test does not establish a statistically reliable difference from SLA.
3. Metric winners are Q 4:8 pairwise for SC, Q share-index=2 for BC, K 2:4 for AQ, SLA for IQ, Original for OC, and Q 2:4 for MC. No method dominates every dimension.
4. Share-index=2 has the largest mean degradation for both Q and K, driven mainly by lower IQ on this test set.
5. Q is feature-sparse immediately before linear `Q @ KV`; K is token-sparse immediately before `K.T @ V`; sparse-block Q/K is feature-sparse. The Rubin triple path additionally applies score-2:4 masked softmax so P is sparse immediately before `P @ V`. Zeros are materialized in dense-layout tensors, so these are sparse-numerics rather than sparse-kernel speed experiments.
