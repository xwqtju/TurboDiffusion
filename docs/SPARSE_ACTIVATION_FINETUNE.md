# Dense-layout activation-sparsity fine-tuning

## What this path trains

The `wan2pt1_14B_res480p_t2v_SLA_KQ_2to4` experiment performs white-box
distillation from a frozen dense-attention teacher to an SLA student. In the
student linear-attention branch it applies:

- `K.T @ V`: 2:4 activation sparsity on K along the token/reduction dimension.
- `Q @ KV`: 2:4 activation sparsity on Q along the head-feature/reduction
  dimension.

The selected activations are set to zero with ordinary PyTorch operations.
Tensor shapes, storage layout and matrix multiplications remain dense. This is
therefore a differentiable numerical simulation, not a sparse-kernel speedup.
The normalization denominator stays dense, matching the ablation inference
path.

## Current support boundary

The repository training stack currently supports Wan2.1 T2V WebDataset and DCP
checkpoints. Wan2.2 I2V uses separate high-noise and low-noise experts and is
not yet supported by this trainer. Do not point this experiment at the two
Wan2.2 `.pth` files: doing so would silently be the wrong model/data contract.

The 14B white-box setup holds a trainable student and a frozen teacher. Its
reference configuration uses 32-way FSDP plus context parallelism; two 32 GB
GPUs are not sufficient for the unchanged full-parameter 14B trainer. Use the
1.3B setup for a local pipeline smoke test, or run the 14B job on an adequately
sized cluster. A memory-reduced Wan2.2 expert-wise trainer will additionally
need an I2V training dataset and a deliberate parameter-efficient strategy.

## Required data and checkpoints

Follow the main README training section to prepare:

- `assets/checkpoints/Wan2.1-T2V-14B.dcp`
- `assets/checkpoints/Wan2.1_VAE.pth`
- `assets/checkpoints/models_t5_umt5-xxl-enc-bf16.pth`
- `assets/checkpoints/umT5_wan_negative_emb.pt`
- the Wan2.1 synthetic WebDataset shards under `assets/datasets/`

## Start the K+Q experiment

```bash
WORKDIR=$PWD
CHECKPOINT_ROOT=$WORKDIR/assets/checkpoints
DATASET_ROOT=$WORKDIR/assets/datasets/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K

torchrun --nproc_per_node=32 --master_port=12341 \
  -m scripts.train \
  --config=turbodiffusion/rcm/configs/registry.py \
  -- experiment=wan2pt1_14B_res480p_t2v_SLA_KQ_2to4 \
  model.config.teacher_ckpt=$CHECKPOINT_ROOT/Wan2.1-T2V-14B.dcp \
  model.config.tokenizer.vae_pth=$CHECKPOINT_ROOT/Wan2.1_VAE.pth \
  model.config.text_encoder_path=$CHECKPOINT_ROOT/models_t5_umt5-xxl-enc-bf16.pth \
  model.config.neg_embed_path=$CHECKPOINT_ROOT/umT5_wan_negative_emb.pt \
  dataloader_train.tar_path_pattern="$DATASET_ROOT/shard*.tar"
```

The two relevant Hydra values are independently overridable:

```text
model.config.linear_kv_2to4_operand=k
model.config.linear_qkv_2to4_operand=q
```

Valid values are `none|k|v` for the first GEMM and `none|q|kv` for the second.
Do not combine legacy `model.config.linear_q_2to4=true` with a non-`none`
`linear_qkv_2to4_operand`; the implementation rejects that ambiguous setup.

## Preflight verification

Run the CPU unit test before launching a cluster job:

```bash
PYTHONPATH=turbodiffusion python -m unittest \
  tests.test_structured_sparsity.StructuredSparsityTest.test_dense_layout_kq_sparsity_supports_backward
```

It verifies that K+Q masks change the forward computation and that finite,
nonzero gradients reach Q, K and V through the dense-layout simulation.
