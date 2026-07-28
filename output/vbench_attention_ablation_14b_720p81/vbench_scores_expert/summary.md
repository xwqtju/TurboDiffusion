# Expert-metric VBench summary

Evaluation set: `vbench_prompt.json` (8 prompts), 720p, 81 frames, 16 fps.

All six columns are raw VBench dimension scores, where higher is better. `Mean` is the unweighted arithmetic mean of SC, BC, AQ, IQ, OC, and MC for this experiment; it is not the official VBench total score.

| Method | SC | BC | AQ | IQ | OC | MC | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 0.954199 | 0.932174 | 0.619796 | 0.657766 | **0.254648** | 0.988122 | 0.734451 |
| **SLA** | 0.966483 | 0.944287 | 0.627141 | **0.676846** | 0.243953 | 0.992271 | **0.741830** |
| SLA + Q 2:4 | 0.967604 | 0.948794 | 0.627044 | 0.660583 | 0.241054 | **0.993319** | 0.739733 |
| SLA + Q 4:8 pairwise | **0.968825** | 0.947749 | 0.611857 | 0.675154 | 0.238930 | 0.992036 | 0.739092 |
| SLA + K 2:4 | 0.964187 | 0.947150 | **0.631279** | 0.657058 | 0.245936 | 0.993234 | 0.739807 |
| SLA + K 4:8 pairwise | 0.966901 | 0.944116 | 0.616120 | 0.675899 | 0.239745 | 0.992439 | 0.739203 |
| SLA + Q 2:4 share-index=2 | 0.964859 | **0.949801** | 0.624470 | 0.653317 | 0.240985 | 0.992964 | 0.737733 |
| SLA + K 2:4 share-index=2 | 0.964040 | 0.948154 | 0.627131 | 0.651485 | 0.244802 | 0.992919 | 0.738088 |

## Interpretation

- SLA has the highest six-metric mean (0.741830), mainly due to the best IQ and strong scores elsewhere.
- Among sparse variants, K 2:4 has the highest mean (0.739807), followed very closely by Q 2:4 (0.739733).
- Metric winners are: SC = Q 4:8 pairwise, BC = Q share-index=2, AQ = K 2:4, IQ = SLA, OC = Original, MC = Q 2:4.
- Every SLA/sparse method beats Original on the unweighted six-metric mean. However, Original has the best OC. The sparse variants generally preserve or improve SC/BC/MC while losing some IQ/OC relative to SLA.
- This is a small 8-video experiment, so differences in the third decimal place should be treated as directional rather than statistically conclusive.

Metric names: SC = subject consistency; BC = background consistency; AQ = aesthetic quality; IQ = imaging quality; OC = overall consistency (ViCLIP); MC = motion smoothness.
