#!/usr/bin/env python3
"""Two-GPU Wan2.2 I2V inference with FSDP2 parameter sharding and context parallelism."""

import gc
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.v2 as T
from einops import rearrange, repeat
from PIL import Image
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import init_device_mesh
from tqdm import tqdm

from imaginaire.utils import log
from imaginaire.utils.io import save_image_or_video
from modify_model import collect_sparsity_profiles, create_model, tensor_kwargs
from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding

_single_gpu_path = Path(__file__).with_name("wan2.2_i2v_infer.py")
_single_gpu_spec = importlib.util.spec_from_file_location("wan22_i2v_single_gpu", _single_gpu_path)
if _single_gpu_spec is None or _single_gpu_spec.loader is None:
    raise ImportError(f"Cannot load argument parser from {_single_gpu_path}")
_single_gpu_module = importlib.util.module_from_spec(_single_gpu_spec)
_single_gpu_spec.loader.exec_module(_single_gpu_module)
parse_arguments = _single_gpu_module.parse_arguments


def merge_rank_profiles(rank_profiles: list[dict]) -> dict:
    """Sum sufficient statistics across context-parallel ranks."""

    merged = {}
    for rank_profile in rank_profiles:
        for model_key in ("high_noise", "low_noise"):
            model_profile = rank_profile[model_key]
            destination = merged.setdefault(model_key, {"model": model_profile["model"], "layers": {}})
            for layer_name, measurements in model_profile["layers"].items():
                layer = destination["layers"].setdefault(layer_name, {})
                for label, values in measurements.items():
                    stats = layer.setdefault(label, {
                        "calls": 0, "elements": 0, "zeros_before": 0, "zeros_after": 0,
                        "error_sq_sum": 0.0, "reference_sq_sum": 0.0,
                    })
                    for key in stats:
                        stats[key] += values[key]
    for model_profile in merged.values():
        for measurements in model_profile["layers"].values():
            for stats in measurements.values():
                elements = stats["elements"]
                reference_sq_sum = stats["reference_sq_sum"]
                stats["zero_rate_before"] = stats["zeros_before"] / elements if elements else None
                stats["zero_rate_after"] = stats["zeros_after"] / elements if elements else None
                stats["relative_l2_error"] = (
                    (stats["error_sq_sum"] / reference_sq_sum) ** 0.5
                    if reference_sq_sum > 0 else None
                )
    return merged


def init_distributed() -> tuple[int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", init_method="env://")
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"This inference path currently requires exactly 2 ranks, got {world_size}")
    return local_rank, world_size


def shard_model(model: torch.nn.Module, world_size: int) -> torch.nn.Module:
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("shard",))
    # Match the repository's training FSDP policy. Setting param_dtype here
    # also casts the FP32 timestep modulation input and violates Wan's block
    # invariant (`e.dtype == torch.float32`). Checkpoint parameters are already
    # BF16, so only the collective reduction dtype needs to be specified.
    mp_policy = MixedPrecisionPolicy(reduce_dtype=torch.float32)
    model.fully_shard(mesh=mesh, mp_policy=mp_policy)
    model = fully_shard(model, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)
    model.enable_context_parallel(dist.group.WORLD)
    return model


def build_tasks(args) -> list[tuple[str, str, str]]:
    if args.prompt_file:
        if not args.image_dir or not args.output_dir:
            raise ValueError("--prompt_file requires --image_dir and --output_dir")
        prompts = json.loads(Path(args.prompt_file).read_text(encoding="utf-8"))
        if not isinstance(prompts, dict) or not prompts:
            raise ValueError("--prompt_file must contain a non-empty JSON object")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tasks = []
        for filename, prompt in prompts.items():
            stem = Path(filename).stem
            images = [
                path for path in Path(args.image_dir).glob(f"{stem}.*")
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            ]
            if len(images) != 1:
                raise FileNotFoundError(f"Expected one conditioning image for {filename}, found {images}")
            tasks.append((prompt, str(images[0]), str(output_dir / filename)))
        return tasks
    if args.prompt is None or args.image_path is None:
        raise ValueError("--prompt and --image_path are required outside batch mode")
    return [(args.prompt, args.image_path, args.save_path)]


def main() -> None:
    args = parse_arguments()
    tasks = build_tasks(args)
    if args.sparsity_profile_path is not None and len(tasks) != 1:
        raise ValueError("Distributed --sparsity_profile_path currently requires exactly one task")
    # The umT5 loader uses a meta-parameter broadcast when torch.distributed is
    # initialized. Compute the small final embedding first on each local GPU,
    # then establish the DiT process group after releasing the encoder.
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    with torch.no_grad():
        text_embeddings = get_umt5_embedding(
            checkpoint_path=args.text_encoder_path,
            prompts=[task[0] for task in tasks],
        ).to(**tensor_kwargs)
    clear_umt5_memory()
    rank, world_size = init_distributed()

    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)
    if args.adaptive_resolution and len(tasks) > 1:
        raise ValueError("--adaptive_resolution is not supported for batch mode")
    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
    frames = args.num_frames
    lat_h = h // tokenizer.spatial_compression_factor
    lat_w = w // tokenizer.spatial_compression_factor
    lat_t = tokenizer.get_latent_num_frames(frames)
    image_transforms = T.Compose([
        T.ToImage(), T.Resize(size=(h, w), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    log.info(f"Rank {rank}: loading and sharding high-noise model", rank0_only=False)
    high_model = create_model(args.high_noise_model_path, args, target_device="cpu")
    high_model = shard_model(high_model, world_size).cuda().eval()
    log.info(f"Rank {rank}: loading and sharding low-noise model", rank0_only=False)
    low_model = create_model(args.low_noise_model_path, args, target_device="cpu")
    low_model = shard_model(low_model, world_size).cpu().eval()
    torch.cuda.empty_cache()

    mid_t = [1.5, 1.4, 1.0][: args.num_steps - 1]
    t_steps = torch.tensor([math.atan(args.sigma_max), *mid_t, 0], dtype=torch.float64, device="cuda")
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

    for task_idx, ((prompt, image_path, save_path), text_emb) in enumerate(zip(tasks, text_embeddings)):
        if rank == 0:
            log.info(f"Batch item {task_idx + 1}/{len(tasks)}: {Path(save_path).name}")
        input_image = Image.open(image_path).convert("RGB")
        image_tensor = image_transforms(input_image).unsqueeze(0).cuda()
        with torch.no_grad():
            frames_to_encode = torch.cat([
                image_tensor.unsqueeze(2),
                torch.zeros(1, 3, frames - 1, h, w, device="cuda"),
            ], dim=2)
            encoded_latents = tokenizer.encode(frames_to_encode)
        del frames_to_encode, image_tensor
        torch.cuda.empty_cache()

        mask = torch.zeros(1, 4, lat_t, lat_h, lat_w, device="cuda", dtype=tensor_kwargs["dtype"])
        mask[:, :, 0] = 1
        image_condition = torch.cat([mask, encoded_latents.to(**tensor_kwargs)], dim=1).repeat(args.num_samples, 1, 1, 1, 1)
        condition = {
            "crossattn_emb": repeat(text_emb.unsqueeze(0).to(**tensor_kwargs), "b l d -> (k b) l d", k=args.num_samples),
            "y_B_C_T_H_W": image_condition,
        }

        high_model.cuda()
        low_model.cpu()
        torch.cuda.empty_cache()
        state_shape = [tokenizer.latent_ch, lat_t, lat_h, lat_w]
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        init_noise = torch.randn(args.num_samples, *state_shape, dtype=torch.float32, device="cuda", generator=generator)
        x = init_noise.to(torch.float64) * t_steps[0]
        ones = torch.ones(x.size(0), 1, device="cuda", dtype=x.dtype)
        net = high_model
        switched = False
        iterator = zip(t_steps[:-1], t_steps[1:])
        if rank == 0:
            iterator = tqdm(list(iterator), desc=Path(save_path).stem, total=args.num_steps)
        for t_cur, t_next in iterator:
            if t_cur.item() < args.boundary and not switched:
                high_model.cpu()
                torch.cuda.empty_cache()
                low_model.cuda()
                net = low_model
                switched = True
            with torch.no_grad():
                v_pred = net(
                    x_B_C_T_H_W=x.to(**tensor_kwargs),
                    timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
                    **condition,
                ).to(torch.float64)
                if args.ode:
                    x = x - (t_cur - t_next) * v_pred
                else:
                    x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                        *x.shape, dtype=torch.float32, device="cuda", generator=generator
                    )

        samples = x.float()
        high_model.cpu()
        low_model.cpu()
        del net, condition, image_condition, encoded_latents, mask, init_noise, x
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
        if rank == 0:
            with torch.no_grad():
                video = tokenizer.decode(samples)
            output = (1.0 + video.float().cpu().clamp(-1, 1)) / 2.0
            save_image_or_video(rearrange(output.unsqueeze(0), "n b c t h w -> c t (n h) (b w)"), save_path, fps=16)
            log.success(f"Saved distributed result to {save_path}")
        del samples
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()

    if args.sparsity_profile_path is not None:
        local_profile = {
            "high_noise": collect_sparsity_profiles(high_model, "high_noise"),
            "low_noise": collect_sparsity_profiles(low_model, "low_noise"),
        }
        rank_profiles = [None for _ in range(world_size)]
        dist.all_gather_object(rank_profiles, local_profile)
        if rank == 0:
            profile = {
                "schema_version": 1,
                "world_size": world_size,
                "attention_type": args.attention_type,
                "sparsity_modes": {
                    "q_2to4": args.sla_q_2to4,
                    "q_4to8_pairwise": args.sla_q_4to8_pairwise,
                    "q_2to4_share_index_2": args.sla_q_2to4_share2,
                    "k_2to4": args.sla_k_2to4,
                    "k_4to8_pairwise": args.sla_k_4to8_pairwise,
                    "k_2to4_share_index_2": args.sla_k_2to4_share2,
                },
                **merge_rank_profiles(rank_profiles),
            }
            profile_path = Path(args.sparsity_profile_path)
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
            log.success(f"Saved distributed sparsity profile to {profile_path}")

    del high_model, low_model
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
