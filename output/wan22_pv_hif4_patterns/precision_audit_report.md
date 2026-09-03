# Wan2.2 SLA P@V 与 W4A4 dense 精度审计报告

## 审计范围

- 四组均为 8 个相同 prompt/首帧、720p、81 帧、4 steps、seed 0。
- 只改变 SLA sparse-FA 的 P@V 路径；Q/K 与 linear branch 保持 dense 16-bit。
- 评价指标为 VBench 六项：SC（Subject Consistency）、BC（Background Consistency）、AQ（Aesthetic Quality）、IQ（Imaging Quality）、OC（Overall Consistency）、MC（Motion Smoothness）。
- Mean 为六项指标的非加权算术平均；Δ 为相对 dense 参考组的差值。

## VBench 聚合结果

| 实验 | SC | BC | AQ | IQ | OC | MC | Mean | Δ Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SLA W16A16 dense | 0.964406 | 0.941705 | 0.620544 | 0.675906 | 0.245056 | 0.992412 | 0.740005 | 0.000000 |
| P@V 2:4 + HiF4 | 0.967624 | 0.948030 | 0.627253 | 0.676863 | 0.242600 | 0.991542 | 0.742319 | +0.002314 |
| P@V 4:8 pairwise + HiF4 | 0.965564 | 0.940917 | 0.630177 | 0.674538 | 0.246017 | 0.992394 | 0.741601 | +0.001597 |
| P@V 2:4 share-index=2 + HiF4 | 0.966178 | 0.947050 | 0.631757 | 0.681832 | 0.246646 | 0.991440 | 0.744150 | +0.004145 |

## 相对 dense 的逐指标变化

| 实验 | ΔSC | ΔBC | ΔAQ | ΔIQ | ΔOC | ΔMC | 六项中提升项 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P@V 2:4 + HiF4 | +0.003218 | +0.006325 | +0.006709 | +0.000958 | -0.002456 | -0.000870 | 4/6 |
| P@V 4:8 pairwise + HiF4 | +0.001158 | -0.000788 | +0.009633 | -0.001368 | +0.000961 | -0.000018 | 3/6 |
| P@V 2:4 share-index=2 + HiF4 | +0.001772 | +0.005345 | +0.011213 | +0.005926 | +0.001589 | -0.000972 | 5/6 |

## 与已有 W4A4 dense 参考结果的比较

已有的 W4A4 参考评分来自
`output/wan22_sla_three_configs/vbench_scores/11_sla_k_hif4_w4a4_dense`（对应视频目录
`output/wan22_sla_three_configs/hif8_2:4_vs_hif4_dense/09_sla_k_hif4_w4a4_dense`），现已归档到本目录的
`vbench_scores/sla_k_hif4_w4a4_dense`、`sla_k_hif4_w4a4_dense/` 和
`audit/sla_k_hif4_w4a4_dense/`。
该批次六项指标的非加权均值为 **0.740486**。

| 实验 | SC | BC | AQ | IQ | OC | MC | Mean | Δ Mean vs W4A4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W4A4 dense（已有参考） | 0.962241 | 0.942064 | 0.623856 | 0.679494 | 0.242996 | 0.992263 | 0.740486 | 0.000000 |
| P@V 2:4 + HiF4 | 0.967624 | 0.948030 | 0.627253 | 0.676863 | 0.242600 | 0.991542 | 0.742319 | +0.001833 |
| P@V 4:8 pairwise + HiF4 | 0.965564 | 0.940917 | 0.630177 | 0.674538 | 0.246017 | 0.992394 | 0.741601 | +0.001116 |
| P@V 2:4 share-index=2 + HiF4 | 0.966178 | 0.947050 | 0.631757 | 0.681832 | 0.246646 | 0.991440 | 0.744150 | +0.003665 |

按这个已有参考口径，三组 P@V HiF4 稀疏的 Mean 都高于 W4A4 dense：share-index=2 提升最大（+0.003665），其次是 2:4（+0.001833）和 4:8 pairwise（+0.001116）。单项并非全部提升：2:4 的 IQ/OC/MC 略低，4:8 的 BC/IQ 略低；因此应表述为“8 个样例上的总体 VBench 均值更高”，而不是每个质量维度都更好。

### 基线口径限制

`11_sla_k_hif4_w4a4_dense` 的审计字段同时记录了 `hif8_w8a8=true`、
`sla_k_hif4_w4a4=true`。这表明它是“全模型 HiF8 W8A8、SLA K 路径 HiF4 W4A4”的
稠密参考，而不是已确认的“所有相关 Linear 都是 W4A4”的完整全路径基线。所以上表结论
只对这个现有参考批次成立；若要严格证明相对于完整 W4A4 dense 的优势，需要用同一批
prompt/首帧、seed 和当前代码重新生成并评估完整 W4A4 dense，然后重算 Δ。

## 审计结论

1. 三种稀疏配置的 Mean 均高于 dense 参考值；当前六项均值排序为：share-index=2 > 2:4 > 4:8 pairwise > dense。
2. share-index=2 的 Mean 提升最大（+0.004145），且 SC、BC、AQ、IQ、OC 五项提升；MC 略低于 dense。
3. 2:4 的 Mean 提升 +0.002314，但 OC、MC 下降，不能表述为所有维度都改善。
4. 4:8 pairwise 的 Mean 提升 +0.001597，主要来自 AQ；BC、IQ、MC 略低于 dense。
5. 这些结果支持“当前 8 样例集合上，P@V HiF4 稀疏没有造成整体 VBench 下降”，但不能仅凭 8 样例断言普遍精度优势。

## 数据与可复现性说明

- 原始 VBench 分数和逐视频明细位于：[vbench_scores](/home/user/桌面/workspace/TurboDiffusion/output/wan22_pv_hif4_patterns/vbench_scores)。
- 汇总来源：[vbench_summary.json](/home/user/桌面/workspace/TurboDiffusion/output/wan22_pv_hif4_patterns/vbench_summary.json)。
- 四组视频与配置审计：[audit_summary.md](/home/user/桌面/workspace/TurboDiffusion/output/wan22_pv_hif4_patterns/audit_summary.md)。
- 需要特别标注：当前代码下新生成的 dense 视频目录为 `sla_w16a16_dense_current`（2026-09-02），而现有 VBench 汇总中的 dense 分数沿用此前 `sla_w16a16_dense` 评分记录。当前环境没有可直接调用的 VBench 评估环境，因此本报告不伪造新 dense 分数；上述数值适合作为已有结果审计，若要形成严格同批次结论，应在 VBench 环境恢复后仅重新评分 dense current，再替换基线列并重算 Δ。
