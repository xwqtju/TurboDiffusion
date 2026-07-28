#!/usr/bin/env python3
"""Two-GPU parameter-efficient Wan2.2 I2V SLA sparsity fine-tuning."""

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh

from inference.modify_model import create_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--cache_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--expert", choices=["high", "low"], required=True)
    parser.add_argument("--boundary", type=float, default=0.9)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--sla_topk", type=float, default=0.1)
    parser.add_argument("--linear_kv_2to4_operand", choices=["k", "v"], default="k")
    parser.add_argument("--linear_qkv_2to4_operand", choices=["q", "kv"], default="q")
    parser.add_argument("--resume_adapter", type=Path)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    return parser.parse_args()


def model_args(args):
    return SimpleNamespace(
        model="Wan2.2-A14B", attention_type="sla", sla_topk=args.sla_topk,
        linear_q_2to4=False,
        linear_kv_2to4_operand=args.linear_kv_2to4_operand,
        linear_qkv_2to4_operand=args.linear_qkv_2to4_operand,
        sla_q_2to4=False, sla_q_4to8_pairwise=False, sla_k_2to4=False,
        sla_k_4to8_pairwise=False, sla_q_2to4_share2=False,
        sla_k_2to4_share2=False, sparsity_profile_path=None,
        # Match the validated Wan2.2 inference path: FastRMSNorm avoids the
        # reference norm's FP32 weight promoting Q/K back to FP32.
        quant_linear=False, default_norm=False, sac_mode="none",
    )


def sparse_modules(model):
    for module in model.modules():
        if hasattr(module, "linear_kv_2to4_operand") and hasattr(module, "linear_qkv_2to4_operand"):
            yield module


@contextmanager
def sparsity_enabled(model, enabled: bool):
    saved = []
    for module in sparse_modules(model):
        saved.append((module, module.linear_kv_2to4_operand, module.linear_qkv_2to4_operand))
        if not enabled:
            module.linear_kv_2to4_operand = "none"
            module.linear_qkv_2to4_operand = "none"
    try:
        yield
    finally:
        for module, kv, qkv in saved:
            module.linear_kv_2to4_operand = kv
            module.linear_qkv_2to4_operand = qkv


def adapter_parameters(model):
    selected = []
    for name, parameter in model.named_parameters():
        trainable = ".attn_op.local_attn.proj_l." in name
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(parameter)
    if not selected:
        raise RuntimeError("No SLA proj_l parameters found")
    return selected


def load_adapter(model, path: Path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state["adapter"], strict=False)
    bad_missing = [name for name in missing if ".attn_op.local_attn.proj_l." in name]
    if bad_missing or unexpected:
        raise RuntimeError(f"Adapter mismatch: missing={bad_missing}, unexpected={unexpected}")


def save_adapter(model, output_dir: Path, step: int, rank: int, metadata: dict):
    adapter = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            value = parameter.full_tensor() if isinstance(parameter, DTensor) else parameter.detach()
            if rank == 0:
                adapter[name] = value.detach().cpu()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"adapter": adapter, "step": step, "metadata": metadata}, output_dir / f"adapter_step_{step:06d}.pt")


def sample_timestep(expert: str, boundary: float, device: torch.device, generator: torch.Generator):
    low, high = (boundary, 1.0) if expert == "high" else (0.0, boundary)
    return torch.rand(1, 1, device=device, generator=generator).mul_(high - low).add_(low)


def main():
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"This entry point requires exactly two GPUs, got {world_size}")
    device = torch.device("cuda", local_rank)

    records = json.loads(args.cache_manifest.read_text(encoding="utf-8"))
    if not records:
        raise ValueError("Empty cache manifest")
    model = create_model(args.model_path, model_args(args), target_device="cpu")
    if args.resume_adapter:
        load_adapter(model, args.resume_adapter)
    adapter_parameters(model)
    model.train()
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("shard",))
    policy = MixedPrecisionPolicy(reduce_dtype=torch.float32)
    model.fully_shard(mesh=mesh, mp_policy=policy)
    model = fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    model.enable_context_parallel(dist.group.WORLD)
    model.cuda()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    metadata = vars(args).copy()
    metadata["cache_manifest"] = str(args.cache_manifest)
    metadata["output_dir"] = str(args.output_dir)
    metadata["resume_adapter"] = str(args.resume_adapter) if args.resume_adapter else None

    for step in range(1, args.steps + 1):
        sample = torch.load(records[(step - 1) % len(records)], map_location="cpu", weights_only=False)
        x0 = sample["latents"].to(device=device, dtype=torch.bfloat16)
        condition = sample["image_condition"].to(device=device, dtype=torch.bfloat16)
        text = sample["text_embedding"].to(device=device, dtype=torch.bfloat16)
        if text.ndim == 2:
            text = text.unsqueeze(0)
        t = sample_timestep(args.expert, args.boundary, device, generator)
        noise = torch.randn(x0.shape, device=device, dtype=torch.bfloat16, generator=generator)
        xt = ((1 - t.view(1, 1, 1, 1, 1)) * x0 + t.view(1, 1, 1, 1, 1) * noise).to(torch.bfloat16)
        kwargs = dict(
            x_B_C_T_H_W=xt,
            timesteps_B_T=(t * 1000).to(torch.bfloat16),
            crossattn_emb=text,
            y_B_C_T_H_W=condition,
        )
        with torch.no_grad(), sparsity_enabled(model, False):
            target = model(**kwargs).float()
        with sparsity_enabled(model, True):
            prediction = model(**kwargs).float()
        loss = (prediction - target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        loss_value = loss.detach()
        dist.all_reduce(loss_value, op=dist.ReduceOp.AVG)
        if rank == 0:
            print(f"expert={args.expert} step={step} loss={loss_value.item():.8g} grad_norm={float(grad_norm):.6g}", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_adapter(model, args.output_dir, step, rank, metadata)
        del sample, x0, condition, text, noise, xt, target, prediction, loss
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
