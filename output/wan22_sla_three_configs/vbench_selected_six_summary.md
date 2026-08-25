# Wan2.2 SLA Selected Six Experiments

Protocol: 8 videos per method, 1280x720, 81 frames, 4 steps, seed 0. Mean is the unweighted mean of SC/BC/AQ/IQ/OC/MC.

| Configuration | SC | BC | AQ | IQ | OC | MC | Mean | Delta vs W16A16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SLA W16A16 dense | 0.964406 | 0.941705 | 0.620544 | 0.675906 | 0.245056 | 0.992412 | 0.740005 | 0.000000 |
| SLA Q-only 2:4 + weight norm | 0.963627 | 0.942315 | 0.616410 | 0.669618 | 0.241286 | 0.992648 | 0.737651 | -0.002354 |
| SLA K-only 2:4 + weight norm | 0.962949 | 0.946915 | 0.619949 | 0.659748 | 0.244670 | 0.993099 | 0.737888 | -0.002117 |
| SLA K-only 2:4 + weight norm + fallback6 | 0.967805 | 0.947791 | 0.621488 | 0.660902 | 0.244973 | 0.993122 | 0.739347 | -0.000658 |
| SLA K-only 2:4 + weight norm + edge3x2 | 0.955417 | 0.940486 | 0.609924 | 0.663634 | 0.238483 | 0.991211 | 0.733192 | -0.006813 |
| SLA K-only 2:4 + RPQ + weight norm + fallback6 | 0.968093 | 0.947144 | 0.615738 | 0.672161 | 0.243091 | 0.992632 | 0.739810 | -0.000195 |

Each experiment directory contains `videos/`, `audit/`, and `logs/`; raw metric outputs are under `vbench_scores/<method>/`.
