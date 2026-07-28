# Official VBench comparison — unquantized TurboWan2.2 I2V A14B

Generation configuration: unquantized `TurboWan2.2-I2V-A14B-high-720P.pth` and `TurboWan2.2-I2V-A14B-low-720P.pth`, 832x480, 65 frames, 16 fps, 4 sampling steps, seed 0, SLA top-k 0.1. Each method uses the same prompt and the same conditioning image extracted from the previous Original T2V video's first frame.

| Method | Subject consistency | Background consistency | Motion smoothness | Dynamic degree | Aesthetic quality | Imaging quality | Custom 6-dim mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 0.968979 | 0.943833 | 0.990210 | 0.625000 | 0.623766 | 0.669576 | **0.803560** |
| SLA | **0.970282** | **0.957263** | 0.990685 | 0.500000 | 0.615303 | 0.660515 | 0.782341 |
| SLA + Q 2:4 | 0.969381 | 0.955933 | **0.991167** | 0.375000 | 0.614658 | 0.659919 | 0.761010 |

The custom 6-dimension mean is an unweighted arithmetic mean for this experiment, not the official 16-dimension VBench Total Score. Official VBench custom-input mode supports these six dimensions. `dynamic_degree` is a binary decision per video, so its aggregate changes in increments of 0.125 for this 8-video set.

End-to-end generation times include loading the text encoder and both 14B DiT checkpoints for every video: Original 163.2 s/video, SLA 150.1 s/video, and SLA + Q 2:4 155.2 s/video. The Q 2:4 implementation simulates activation sparsity for quality evaluation and is not a Rubin hardware sparse kernel benchmark.
