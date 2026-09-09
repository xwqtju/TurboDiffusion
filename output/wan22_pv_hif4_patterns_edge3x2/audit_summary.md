# edge3x2 实验审计摘要

目标回退层：`[0, 1, 2, 37, 38, 39]`；每个 high/low noise 分支共 40 个 SLA block。
回退层只将 `pv_sparsity` 设为 `none`，仍走 SLA 的 P@V 路径并保留 `pv_hif4=true`；QK 量化由 `pv_qk_hif4` 独立控制。

| 方法 | pv_sparsity | pv_hif4 | pv_qk_hif4 | 回退层 | 稀疏层比例 | 视频数 |
|---|---|---:|---:|---|---:|---:|
| P@V 2:4 + HiF4, edge3x2 | 2to4 | true | false | `[0, 1, 2, 37, 38, 39]` | 34/40=85% | 8 |
| P@V 4:8 pairwise + HiF4, edge3x2 | 4to8_pairwise | true | false | `[0, 1, 2, 37, 38, 39]` | 34/40=85% | 8 |
| P@V 2:4 share2 + HiF4, edge3x2 | 2to4_share2 | true | false | `[0, 1, 2, 37, 38, 39]` | 34/40=85% | 8 |
| P@V 2:4 + P/V QK HiF4, edge3x2 | 2to4 | true | true | `[0, 1, 2, 37, 38, 39]` | 34/40=85% | 8 |
| P@V 4:8 pairwise + P/V QK HiF4, edge3x2 | 4to8_pairwise | true | true | `[0, 1, 2, 37, 38, 39]` | 34/40=85% | 8 |
| P@V 2:4 share2 + P/V QK HiF4, edge3x2 | 2to4_share2 | true | true | `[0, 1, 2, 37, 38, 39]` | 34/40=85% | 8 |
