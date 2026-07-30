# Rubin fused sparse-attention validation

Both runs use the same eight I2V prompts/images, unquantized TurboWan2.2 14B high/low checkpoints, 720p, 81 frames, four steps, and seed 0. `Mean` is the unweighted mean of the six requested VBench dimensions.

| Method | SC | BC | AQ | IQ | OC | MC | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Explicit reference | 0.963967 | 0.949355 | 0.627591 | 0.667828 | 0.245018 | 0.992283 | 0.741007 |
| Fused Triton (rerun) | 0.962691 | 0.946748 | 0.633310 | 0.660875 | 0.244118 | 0.992042 | 0.739964 |
| Fused - reference | -0.001277 | -0.002607 | +0.005719 | -0.006953 | -0.000899 | -0.000241 | -0.001043 |

The paired eight-video bootstrap 95% CI for the six-metric mean difference is `[-0.003711, +0.000931]`, which crosses zero. On this small single-seed set, the fused and explicit paths therefore have no statistically established quality difference. The point estimate is 0.001043 lower for fused, driven mainly by IQ, while AQ is higher.

The normal fused run takes about 78-81 seconds for four diffusion steps per video. The old explicit path took about 617 seconds for the comparable garden run, a roughly 7.6x diffusion-loop speedup. Model loading, VAE encode/decode, and video writing are excluded from that ratio.
