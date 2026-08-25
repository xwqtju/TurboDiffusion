# Wan2.2 SLA：去除 weight-aware 后的 VBench 汇总

指标顺序：SC=Subject Consistency，BC=Background Consistency，AQ=Aesthetic Quality，IQ=Imaging Quality，OC=Overall Consistency，MC=Motion Smoothness；Mean 为六项算术平均。

| 方向 | 配置（无 weight-aware） | SC | BC | AQ | IQ | OC | MC | Mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| K-only | 2:4 | 0.962804 | 0.945734 | 0.626138 | 0.652657 | 0.244153 | 0.993088 | 0.737429 |
| K-only | 2:4 + RPQ + fallback6 | 0.967506 | 0.944726 | 0.621502 | 0.673881 | 0.243178 | 0.992564 | 0.740560 |
| K-only | 4:8 pairwise + RPQ + fallback6 | 0.967648 | 0.952800 | 0.615577 | 0.674267 | 0.244483 | 0.993356 | 0.741355 |
| K-only | 2:4 share-index=2 + RPQ + fallback6 | 0.966804 | 0.944701 | 0.617415 | 0.671573 | 0.245903 | 0.992678 | 0.739846 |
| Q-only | 2:4 | 0.965359 | 0.945763 | 0.619017 | 0.657426 | 0.243334 | 0.993022 | 0.737320 |
| Q-only | 2:4 + RPQ + fallback6 | 0.968677 | 0.944196 | 0.618501 | 0.673300 | 0.243473 | 0.993214 | 0.740227 |
| Q-only | 4:8 pairwise + RPQ + fallback6 | 0.967849 | 0.947404 | 0.619530 | 0.674535 | 0.241445 | 0.993637 | 0.740734 |
| Q-only | 2:4 share-index=2 + RPQ + fallback6 | 0.968471 | 0.944758 | 0.615191 | 0.672749 | 0.242772 | 0.993123 | 0.739511 |

结果目录：`output/wan22_no_weight_aware_patterns/`。原始 VBench JSON 位于 `vbench_scores_expert/`。
