# Wan2.2 SLA W16A16、K-only、Q-only 汇总

VBench 六项指标及算术平均（8 prompts，720p，81 帧，4 steps，seed 0）。

| 类别 | 配置 | SC | BC | AQ | IQ | OC | MC | Mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| W16A16 | Dense baseline | 0.964406 | 0.941705 | 0.620544 | 0.675906 | 0.245056 | 0.992412 | 0.740005 |
| K-only | 2:4 weight norm | 0.962949 | 0.946915 | 0.619949 | 0.659748 | 0.244670 | 0.993099 | 0.737888 |
| K-only | 2:4 + RPQ + weight norm + fallback6 | 0.968093 | 0.947144 | 0.615738 | 0.672161 | 0.243091 | 0.992632 | 0.739810 |
| K-only | 4:8 pairwise + RPQ + weight norm + fallback6 | 0.967648 | 0.952800 | 0.615577 | 0.674267 | 0.244483 | 0.993356 | 0.741355 |
| K-only | 2:4 share-index=2 + RPQ + weight norm + fallback6 | 0.967222 | 0.950047 | 0.616504 | 0.671962 | 0.245952 | 0.992733 | 0.740737 |
| Q-only | 2:4 weight norm | 0.963627 | 0.942315 | 0.616410 | 0.669618 | 0.241286 | 0.992648 | 0.737651 |
| Q-only | 2:4 + RPQ + weight norm + fallback6 | 0.967002 | 0.944538 | 0.611749 | 0.668982 | 0.241525 | 0.993058 | 0.737809 |
| Q-only | 4:8 pairwise + RPQ + weight norm + fallback6 | 0.967849 | 0.947404 | 0.619530 | 0.674535 | 0.241445 | 0.993637 | 0.740734 |
| Q-only | 2:4 share-index=2 + RPQ + weight norm + fallback6 | 0.968986 | 0.950576 | 0.614915 | 0.670033 | 0.241787 | 0.992959 | 0.739876 |

SC=Subject Consistency，BC=Background Consistency，AQ=Aesthetic Quality，IQ=Imaging Quality，OC=Overall Consistency，MC=Motion Smoothness。
