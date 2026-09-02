# Wan2.2 SLA fused P@V HiF4 VBench results

Protocol: 8 prompts/images, 1280x720, 81 frames, 4 diffusion steps, seed 0. Only sparse-FA P@V is changed; Q/K and the linear-attention branch remain dense 16-bit. P is selected on S before masked-softmax; P and V use HiF4. No layer fallback.

| Method | SC | BC | AQ | IQ | OC | MC | Mean | Δ vs W16A16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SLA W16A16 dense | 0.964406 | 0.941705 | 0.620544 | 0.675906 | 0.245056 | 0.992412 | 0.740005 | +0.000000 |
| SLA P@V 2:4 + HiF4 | 0.967624 | 0.948030 | 0.627253 | 0.676863 | 0.242600 | 0.991542 | 0.742319 | +0.002314 |
| SLA P@V 4:8 pairwise + HiF4 | 0.965564 | 0.940917 | 0.630177 | 0.674538 | 0.246017 | 0.992394 | 0.741601 | +0.001597 |
| SLA P@V 2:4 share-index=2 + HiF4 | 0.966178 | 0.947050 | 0.631757 | 0.681832 | 0.246646 | 0.991440 | 0.744150 | +0.004145 |

Raw metric outputs are under `vbench_scores/<method>/`.
