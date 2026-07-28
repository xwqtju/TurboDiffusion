# Wan2.2 I2V 双卡稀疏微调

这套入口用于 Wan2.2 I2V A14B 的 high-noise / low-noise 专家分别进行参数高效微调。

## 训练设计

- 每次只加载一个 14B 专家，以 FSDP2 在两张 GPU 间分片。
- 只训练 40 层 SLA 的 `proj_l.weight` 和 `proj_l.bias`，共 80 个张量、660,480 个参数。
- 同一个模型先关闭 K/Q 2:4，在 `no_grad` 下得到 dense-SLA teacher 输出；再打开 K/Q 2:4 得到 student 输出并反向传播。
- `K.T @ V` 对 K 的 token/reduction 维做 2:4；`Q @ KV` 对 Q 的 head-feature/reduction 维做 2:4。
- 稀疏激活仍是置零后的 dense-layout Tensor，不使用稀疏 kernel。
- high 专家只采样 `[boundary, 1]` 的噪声时间，low 专家只采样 `[0, boundary)`。

这样不需要同时保留 dense teacher 和 sparse student 两份 14B 权重，也不需要给整个模型保存 Adam 状态。

## 数据格式

原始 manifest 是 JSON list，每项包含：

```json
{
  "id": "sample_name",
  "image": "conditioning_image.png",
  "video": "target_video.mp4",
  "prompt": "text prompt"
}
```

仓库已经提供基于现有 8-prompt 实验结果的 smoke manifest：
`data/wan22_i2v_smoke_manifest.json`。这 8 条合成视频只用于验证和小规模过拟合，不能用于声称泛化质量提升。正式实验应替换成更大且有授权的 I2V 数据集。

先缓存 VAE latent、首帧条件和 umT5 embedding：

```bash
PYTHONPATH=turbodiffusion python scripts/prepare_wan22_i2v_finetune_data.py \
  --manifest data/wan22_i2v_smoke_manifest.json \
  --output_dir output/wan22_i2v_finetune_cache_448x256x17 \
  --vae_path checkpoints/Wan2.1_VAE.pth \
  --text_encoder_path checkpoints/models_t5_umt5-xxl-enc-bf16.pth \
  --height 256 --width 448 --frames 17
```

训练 crop 可以低于最终推理的 720p×81；SLA `proj_l` 与 token 数无关，合并后的权重仍能用于 720p×81 推理。

## 分别训练两个专家

high-noise：

```bash
PYTHONPATH=turbodiffusion torchrun --standalone --nproc_per_node=2 \
  scripts/train_wan22_i2v_sparse_adapter.py \
  --model_path checkpoints/TurboWan2.2-I2V-A14B-high-720P.pth \
  --cache_manifest output/wan22_i2v_finetune_cache_448x256x17/cache_manifest.json \
  --output_dir output/wan22_sparse_finetune/high \
  --expert high --boundary 0.9 --steps 1000 --save_every 100
```

low-noise：

```bash
PYTHONPATH=turbodiffusion torchrun --standalone --nproc_per_node=2 \
  scripts/train_wan22_i2v_sparse_adapter.py \
  --model_path checkpoints/TurboWan2.2-I2V-A14B-low-720P.pth \
  --cache_manifest output/wan22_i2v_finetune_cache_448x256x17/cache_manifest.json \
  --output_dir output/wan22_sparse_finetune/low \
  --expert low --boundary 0.9 --steps 1000 --save_every 100
```

两个任务必须顺序运行，不能在同一双卡节点上同时启动。可用 `--resume_adapter` 从一个 adapter 权重继续；当前实现只恢复模型 adapter，不恢复 Adam 动量。

## 合并并推理

分别合并到对应专家，不能交叉：

```bash
PYTHONPATH=turbodiffusion python scripts/merge_wan22_sparse_adapter.py \
  --base checkpoints/TurboWan2.2-I2V-A14B-high-720P.pth \
  --adapter output/wan22_sparse_finetune/high/adapter_step_001000.pt \
  --output checkpoints/TurboWan2.2-I2V-A14B-high-720P-KQ-ft.pth

PYTHONPATH=turbodiffusion python scripts/merge_wan22_sparse_adapter.py \
  --base checkpoints/TurboWan2.2-I2V-A14B-low-720P.pth \
  --adapter output/wan22_sparse_finetune/low/adapter_step_001000.pt \
  --output checkpoints/TurboWan2.2-I2V-A14B-low-720P-KQ-ft.pth
```

推理时必须继续开启与训练一致的参数：

```text
--attention_type sla
--linear_kv_2to4_operand k
--linear_qkv_2to4_operand q
```

## 本机验证记录

使用 448×256×17 缓存已在双 GPU 上分别完成 high 和 low 的 1-step 实测：

- high：80/80 个 adapter 张量更新，最大绝对更新约 `1.53e-5`。
- low：80/80 个 adapter 张量更新，最大绝对更新约 `1.14e-5`。
- 两个 adapter 均包含 660,480 个有限参数。

smoke adapter 位于 `output/wan22_sparse_finetune/high_smoke` 和
`output/wan22_sparse_finetune/low_smoke`。
