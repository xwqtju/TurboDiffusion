# Q-only pattern VBench summary

| Config | SC | BC | AQ | IQ | OC | MC | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q-only 2:4 weight norm | 0.963627 | 0.942315 | 0.616410 | 0.669618 | 0.241286 | 0.992648 | 0.737651 |
| Q-only 2:4 + RPQ + weight norm + fallback6 | 0.967002 | 0.944538 | 0.611749 | 0.668982 | 0.241525 | 0.993058 | 0.737809 |
| Q-only 4:8 pairwise + RPQ + weight norm + fallback6 | 0.967849 | 0.947404 | 0.619530 | 0.674535 | 0.241445 | 0.993637 | 0.740734 |
| Q-only 2:4 share-index=2 + RPQ + weight norm + fallback6 | 0.968986 | 0.950576 | 0.614915 | 0.670033 | 0.241787 | 0.992959 | 0.739876 |

Mean is the arithmetic average of the six metrics.
