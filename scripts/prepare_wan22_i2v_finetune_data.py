#!/usr/bin/env python3
"""Cache video/image/prompt triples for Wan2.2 I2V expert fine-tuning."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.io import read_video
from torchvision.transforms.functional import pil_to_tensor

from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--vae_path", required=True)
    parser.add_argument("--text_encoder_path", required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resize_normalize(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert uint8 [T,H,W,C] to normalized float [1,C,T,H,W]."""
    frames = frames.permute(0, 3, 1, 2).float().div_(255)
    frames = F.interpolate(frames, size=(height, width), mode="bilinear", align_corners=False, antialias=True)
    return frames.mul_(2).sub_(1).permute(1, 0, 2, 3).unsqueeze(0).contiguous()


def load_video(path: Path, count: int, height: int, width: int) -> torch.Tensor:
    frames, _, _ = read_video(str(path), pts_unit="sec", output_format="THWC")
    if frames.shape[0] < count:
        raise ValueError(f"{path} has {frames.shape[0]} frames, need at least {count}")
    indices = torch.linspace(0, frames.shape[0] - 1, count).round().long()
    return resize_normalize(frames[indices], height, width)


def load_image(path: Path, height: int, width: int) -> torch.Tensor:
    image = pil_to_tensor(Image.open(path).convert("RGB")).unsqueeze(0)
    image = F.interpolate(image.float().div_(255), size=(height, width), mode="bilinear", align_corners=False, antialias=True)
    return image.mul_(2).sub_(1)


def main():
    args = parse_args()
    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("manifest must be a non-empty JSON list")
    if (args.frames - 1) % 4:
        raise ValueError("--frames must satisfy frames = 4*n + 1 for Wan VAE")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompts = [record["prompt"] for record in records]
    with torch.no_grad():
        text_embeddings = get_umt5_embedding(args.text_encoder_path, prompts).cpu()
    clear_umt5_memory()
    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)

    cached_manifest = []
    for index, (record, text_embedding) in enumerate(zip(records, text_embeddings)):
        video = load_video(Path(record["video"]), args.frames, args.height, args.width).to(args.device)
        image = load_image(Path(record["image"]), args.height, args.width).to(args.device)
        with torch.no_grad():
            latents = tokenizer.encode(video).cpu()
            image_video = torch.cat([
                image.unsqueeze(2),
                torch.zeros(1, 3, args.frames - 1, args.height, args.width, device=args.device),
            ], dim=2)
            image_latents = tokenizer.encode(image_video).cpu()
        latent_t, latent_h, latent_w = latents.shape[-3:]
        mask = torch.zeros(1, 4, latent_t, latent_h, latent_w, dtype=torch.bfloat16)
        mask[:, :, 0] = 1
        destination = args.output_dir / f"{index:04d}_{record.get('id', Path(record['video']).stem)}.pt"
        torch.save({
            "latents": latents.to(torch.bfloat16),
            "image_condition": torch.cat([mask, image_latents.to(torch.bfloat16)], dim=1),
            "text_embedding": text_embedding.to(torch.bfloat16),
            "prompt": record["prompt"],
            "source": record,
        }, destination)
        cached_manifest.append(str(destination.resolve()))
        print(f"[{index + 1}/{len(records)}] {destination}", flush=True)
    (args.output_dir / "cache_manifest.json").write_text(json.dumps(cached_manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
