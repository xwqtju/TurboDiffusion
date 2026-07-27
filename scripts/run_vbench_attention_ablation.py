#!/usr/bin/env python3
"""Run reproducible original/SLA/SLA-Q-2:4 generation comparisons."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path


METHOD_ARGS = {
    "original": ["--attention_type", "original"],
    "sla": ["--attention_type", "sla"],
    "sla_q_2to4": ["--attention_type", "sla", "--sla_q_2to4"],
}


@dataclass(frozen=True)
class Task:
    method: str
    filename: str
    prompt: str
    output_path: Path
    image_path: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=Path("vbench_prompt.json"))
    parser.add_argument("--pipeline", choices=("t2v-1.3b", "i2v-a14b"), default="t2v-1.3b")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/TurboWan2.1-T2V-1.3B-480P.pth"),
    )
    parser.add_argument(
        "--high-noise-checkpoint",
        type=Path,
        default=Path("checkpoints/TurboWan2.2-I2V-A14B-high-720P.pth"),
    )
    parser.add_argument(
        "--low-noise-checkpoint",
        type=Path,
        default=Path("checkpoints/TurboWan2.2-I2V-A14B-low-720P.pth"),
    )
    parser.add_argument(
        "--input-image-dir",
        type=Path,
        help="For i2v-a14b, contains one image per prompt using the MP4 stem (for example creature.png)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/vbench_attention_ablation"))
    parser.add_argument("--gpus", default="0,1", help="Comma-separated physical GPU IDs")
    parser.add_argument("--methods", nargs="+", choices=METHOD_ARGS, default=list(METHOD_ARGS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=4, choices=(1, 2, 3, 4))
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--resolution", choices=("480p", "720p"), help="Defaults to 480p for T2V and 720p for I2V")
    parser.add_argument("--sla-topk", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N prompts")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_prompts(path: Path, limit: int | None) -> list[tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path} must contain a non-empty JSON object")
    prompts = []
    for filename, prompt in data.items():
        if not isinstance(filename, str) or not filename.endswith(".mp4"):
            raise ValueError(f"Invalid output filename: {filename!r}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Prompt for {filename!r} must be a non-empty string")
        prompts.append((filename, prompt))
    return prompts[:limit]


def output_is_complete(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def build_command(args: argparse.Namespace, task: Task, repo_root: Path) -> list[str]:
    common = [
        "--prompt",
        task.prompt,
        "--num_samples",
        "1",
        "--num_steps",
        str(args.num_steps),
        "--num_frames",
        str(args.num_frames),
        "--seed",
        str(args.seed),
        "--save_path",
        str(task.output_path.resolve()),
        "--sla_topk",
        str(args.sla_topk),
        *METHOD_ARGS[task.method],
    ]
    if args.pipeline == "i2v-a14b":
        assert task.image_path is not None
        return [
            sys.executable,
            str(repo_root / "turbodiffusion/inference/wan2.2_i2v_infer.py"),
            "--model",
            "Wan2.2-A14B",
            "--high_noise_model_path",
            str(args.high_noise_checkpoint.resolve()),
            "--low_noise_model_path",
            str(args.low_noise_checkpoint.resolve()),
            "--vae_path",
            str((repo_root / "checkpoints/Wan2.1_VAE.pth").resolve()),
            "--text_encoder_path",
            str((repo_root / "checkpoints/models_t5_umt5-xxl-enc-bf16.pth").resolve()),
            "--resolution",
            args.resolution,
            "--aspect_ratio",
            "16:9",
            "--image_path",
            str(task.image_path.resolve()),
            *common,
        ]
    return [
        sys.executable,
        str(repo_root / "turbodiffusion/inference/wan2.1_t2v_infer.py"),
        "--model",
        "Wan2.1-1.3B",
        "--dit_path",
        str(args.checkpoint.resolve()),
        "--vae_path",
        str((repo_root / "checkpoints/Wan2.1_VAE.pth").resolve()),
        "--text_encoder_path",
        str((repo_root / "checkpoints/models_t5_umt5-xxl-enc-bf16.pth").resolve()),
        "--resolution",
        "480p",
        "--aspect_ratio",
        "16:9",
        *common,
    ]


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    args.prompts = (repo_root / args.prompts).resolve() if not args.prompts.is_absolute() else args.prompts
    args.checkpoint = (repo_root / args.checkpoint).resolve() if not args.checkpoint.is_absolute() else args.checkpoint
    args.high_noise_checkpoint = (repo_root / args.high_noise_checkpoint).resolve() if not args.high_noise_checkpoint.is_absolute() else args.high_noise_checkpoint
    args.low_noise_checkpoint = (repo_root / args.low_noise_checkpoint).resolve() if not args.low_noise_checkpoint.is_absolute() else args.low_noise_checkpoint
    if args.input_image_dir is not None:
        args.input_image_dir = (repo_root / args.input_image_dir).resolve() if not args.input_image_dir.is_absolute() else args.input_image_dir
    args.output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    if args.resolution is None:
        args.resolution = "480p" if args.pipeline == "t2v-1.3b" else "720p"

    if args.pipeline == "t2v-1.3b":
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    else:
        for checkpoint in (args.high_noise_checkpoint, args.low_noise_checkpoint):
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        if args.input_image_dir is None or not args.input_image_dir.is_dir():
            raise FileNotFoundError("--input-image-dir is required for i2v-a14b")
    prompts = load_prompts(args.prompts, args.limit)
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU ID is required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks: queue.Queue[Task] = queue.Queue()
    scheduled = []
    for method in args.methods:
        method_dir = args.output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        for filename, prompt in prompts:
            image_path = None
            if args.pipeline == "i2v-a14b":
                matches = [path for path in args.input_image_dir.glob(f"{Path(filename).stem}.*") if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
                if len(matches) != 1:
                    raise FileNotFoundError(f"Expected exactly one input image for {filename}, found: {matches}")
                image_path = matches[0]
            task = Task(method, filename, prompt, method_dir / filename, image_path)
            if args.overwrite or not output_is_complete(task.output_path):
                tasks.put(task)
                scheduled.append(task)

    config = {
        "pipeline": args.pipeline,
        "checkpoint": str(args.checkpoint) if args.pipeline == "t2v-1.3b" else None,
        "high_noise_checkpoint": str(args.high_noise_checkpoint) if args.pipeline == "i2v-a14b" else None,
        "low_noise_checkpoint": str(args.low_noise_checkpoint) if args.pipeline == "i2v-a14b" else None,
        "quantized": False,
        "input_image_dir": str(args.input_image_dir) if args.input_image_dir else None,
        "prompts": str(args.prompts),
        "output_dir": str(args.output_dir),
        "methods": args.methods,
        "gpus": gpu_ids,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "num_frames": args.num_frames,
        "resolution": args.resolution,
        "aspect_ratio": "16:9",
        "sla_topk": args.sla_topk,
        "prompt_count": len(prompts),
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    manifest_path = args.output_dir / "manifest.jsonl"
    manifest_lock = threading.Lock()
    failures: list[dict] = []

    def worker(gpu_id: str) -> None:
        while True:
            try:
                task = tasks.get_nowait()
            except queue.Empty:
                return

            command = build_command(args, task, repo_root)
            log_path = task.output_path.with_suffix(".log")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            python_path = str(repo_root / "turbodiffusion")
            env["PYTHONPATH"] = python_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            started_at = time.time()
            with log_path.open("w", encoding="utf-8") as log_file:
                result = subprocess.run(
                    command,
                    cwd=repo_root,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            record = {
                **asdict(task),
                "output_path": str(task.output_path),
                "image_path": str(task.image_path) if task.image_path else None,
                "gpu": gpu_id,
                "command": command,
                "started_at_unix": started_at,
                "duration_seconds": time.time() - started_at,
                "returncode": result.returncode,
                "output_bytes": task.output_path.stat().st_size if task.output_path.exists() else 0,
                "log_path": str(log_path),
            }
            with manifest_lock:
                with manifest_path.open("a", encoding="utf-8") as manifest:
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                if result.returncode != 0 or not output_is_complete(task.output_path):
                    failures.append(record)
                status = "OK" if result.returncode == 0 and output_is_complete(task.output_path) else "FAILED"
                print(f"[{status}] gpu={gpu_id} {task.method}/{task.filename} ({record['duration_seconds']:.1f}s)", flush=True)
            tasks.task_done()

    print(f"Scheduling {len(scheduled)} generations across GPUs {gpu_ids}", flush=True)
    workers = [threading.Thread(target=worker, args=(gpu_id,), daemon=False) for gpu_id in gpu_ids]
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join()

    if failures:
        print(f"{len(failures)} generation(s) failed; inspect per-video logs", file=sys.stderr)
        return 1
    print(f"Completed {len(scheduled)} generation(s). Results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
