# Wan2.2 SLA P@V 与 W4A4 dense 实验审计汇总

- 核对时间：2026-09-02
- 输入：同一 `vbench_prompt.json`、`assets/vbench_baseline_first_frames`
- 推理协议：Wan2.2-A14B，SLA，720p，81 帧，4 steps，seed 0，双卡 distributed batch
- 检查原则：只核对已有视频、运行日志与 `audit/<method>/batch.json`，不因审计更新重新生成视频。

| 实验 | 视频数 | batch.json | attention | pv_sparsity | pv_hif4 | Q/K 稀疏 | 日志完成标记 | 结论 |
|---|---:|---|---|---|---|---|---|---|
| `sla_w16a16_dense_current` | 8/8 | 有 | `sla` | `none` | false | false/false | 有 | 通过 |
| `sla_pv_2to4_hif4` | 8/8 | 有（2026-09-02 重写） | `sla` | `2to4` | true | false/false | 有 | 通过 |
| `sla_pv_4to8_pairwise_hif4` | 8/8 | 有 | `sla` | `4to8_pairwise` | true | false/false | 有（原始完整运行） | 通过 |
| `sla_pv_2to4_share2_hif4` | 8/8 | 有 | `sla` | `2to4_share2` | true | false/false | 有（原始完整运行） | 通过 |
| `sla_k_hif4_w4a4_dense` | 8/8 | 有（补齐） | `sla` | `none` | false | false/false | 有（原始完整运行） | 通过（现有 K-only W4A4 参考） |

## 文件位置

- 视频根目录：[wan22_pv_hif4_patterns](/home/user/桌面/workspace/TurboDiffusion/output/wan22_pv_hif4_patterns)
- Dense 审计：[audit/sla_w16a16_dense_current/batch.json](/home/user/桌面/workspace/TurboDiffusion/output/wan22_pv_hif4_patterns/audit/sla_w16a16_dense_current/batch.json)
- 2:4 审计：[audit/sla_pv_2to4_hif4/batch.json](/home/user/桌面/workspace/TurboDiffusion/output/wan22_pv_hif4_patterns/audit/sla_pv_2to4_hif4/batch.json)
- 4:8 pairwise 审计：[audit/sla_pv_4to8_pairwise_hif4/batch.json](/home/user/桌面/workspace/TurboDiffusion/output/wan22_pv_hif4_patterns/audit/sla_pv_4to8_pairwise_hif4/batch.json)
- share-index=2 审计：[audit/sla_pv_2to4_share2_hif4/batch.json](/home/user/桌面/workspace/TurboDiffusion/output/wan22_pv_hif4_patterns/audit/sla_pv_2to4_share2_hif4/batch.json)
- W4A4 dense 参考审计：[audit/sla_k_hif4_w4a4_dense/batch.json](/home/user/桌面/workspace/TurboDiffusion/output/wan22_pv_hif4_patterns/audit/sla_k_hif4_w4a4_dense/batch.json)

## 说明

五个 `batch.json` 均报告 `task_count=8`、8 个预期输出，且各组有 8 个非空 MP4。三组 P@V 的配置分别是 `2to4`、`4to8_pairwise`、`2to4_share2`，均为 `pv_hif4=true`；W16A16 dense 组为 `pv_sparsity=none`、`pv_hif4=false`。新增的 W4A4 参考组为 `hif8_w8a8=true`、`sla_k_hif4_w4a4=true`，即原有“全模型 HiF8、SLA K 路径 HiF4 W4A4”的 dense 参考，并非完整所有 Linear 均为 W4A4。所有组 Q/K 稀疏开关均为 false，未启用层回退。

当前 fused P@V 路径的逐层 Q/K/attention-output 计数在 `batch.json` 中为零，这是该审计接口未对 fused P 操作暴露逐层计数，并不表示没有执行 P@V；配置级字段、8 个视频输出和 batch 日志完成标记均已核对。

精度指标审计见：[precision_audit_report.md](precision_audit_report.md)。
