# GEMM-operand activation sparsity: VBench six-metric comparison

All methods use the same eight I2V prompts/images, unquantized TurboWan2.2 14B high/low checkpoints, 720p, 81 frames, four steps, and seed 0. The mean is an internal unweighted six-metric mean, not the official 16-dimension VBench total.

| Method | SC | BC | AQ | IQ | OC | MC | Mean | Delta vs SLA | Delta vs Original | Paired bootstrap 95% CI vs SLA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 0.950168 | 0.928579 | 0.617031 | 0.664976 | 0.256395 | 0.989085 | 0.734372 | -0.005633 | +0.000000 | — |
| SLA | 0.964406 | 0.941705 | 0.620544 | 0.675906 | 0.245056 | 0.992412 | 0.740005 | +0.000000 | +0.005633 | — |
| SLA + Q 2:4 + HiF4 | 0.958411 | 0.944183 | 0.611389 | 0.626577 | 0.247186 | 0.992202 | 0.729991 | -0.010014 | -0.004381 | [-0.018452, -0.003265] |
| SLA + Q 4:8 pairwise + HiF4 | 0.958590 | 0.939358 | 0.610491 | 0.643774 | 0.245967 | 0.991258 | 0.731573 | -0.008432 | -0.002799 | [-0.014158, -0.003494] |
| SLA + Q 2:4 share-index=2 + HiF4 | 0.950177 | 0.946007 | 0.602956 | 0.592103 | 0.250428 | 0.991968 | 0.722273 | -0.017732 | -0.012099 | [-0.028214, -0.008273] |
| SLA + K 2:4 + HiF4 | 0.958278 | 0.950930 | 0.618255 | 0.629449 | 0.246877 | 0.992739 | 0.732755 | -0.007250 | -0.001618 | [-0.012715, -0.001932] |
| SLA + K 4:8 pairwise + HiF4 | 0.963462 | 0.942134 | 0.613641 | 0.659698 | 0.242197 | 0.992091 | 0.735537 | -0.004468 | +0.001165 | [-0.009628, +0.000426] |
| SLA + K 2:4 share-index=2 + HiF4 | 0.956399 | 0.946838 | 0.614504 | 0.618958 | 0.247447 | 0.992723 | 0.729478 | -0.010527 | -0.004894 | [-0.016341, -0.004897] |
| SLA + K(token) + Q(feature) 2:4 + HiF4 | 0.931482 | 0.931663 | 0.592058 | 0.654113 | 0.242462 | 0.989191 | 0.723495 | -0.016510 | -0.010877 | [-0.029144, -0.005523] |
| SLA + K(token) + KV(feature) 2:4 + HiF4 | 0.941382 | 0.929873 | 0.615145 | 0.670946 | 0.244124 | 0.990336 | 0.731968 | -0.008037 | -0.002405 | [-0.017281, -0.000482] |
| SLA + V(token) + Q(feature) 2:4 + HiF4 | 0.927462 | 0.926760 | 0.593168 | 0.653058 | 0.241361 | 0.989119 | 0.721821 | -0.018184 | -0.012551 | [-0.032656, -0.005617] |
| SLA + V(token) + KV(feature) 2:4 + HiF4 | 0.957992 | 0.937401 | 0.617186 | 0.678335 | 0.244876 | 0.992250 | 0.738007 | -0.001998 | +0.003634 | [-0.007849, +0.002478] |
| SLA + Rubin triple 2:4 + HiF4 | 0.957824 | 0.950232 | 0.622261 | 0.654089 | 0.245824 | 0.991430 | 0.736943 | -0.003062 | +0.002571 | [-0.011182, +0.004205] |

## Conclusions

Best six-metric mean: SLA (0.740005).
Best sparse+HiF4/HiF4-only method: SLA + V(token) + KV(feature) 2:4 + HiF4 (0.738007); delta vs SLA -0.001998.
These are dense-layout BF16/FP16 HiF4 QDQ numerical simulations; they do not represent packed 4-bit storage or measured sparse Tensor-Core speedups.
