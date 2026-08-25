#!/usr/bin/env python3
"""Run resumable two-GPU Wan2.2 branch-aware K ablations one video at a time."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


METHODS = {
    "original": [],
    "sla": [],
    "sla_q_2to4_hif4": ["--sla_q_2to4", "--hif4_sparse_upgrade"],
    "sla_q_4to8_pairwise_hif4": ["--sla_q_4to8_pairwise", "--hif4_sparse_upgrade"],
    "sla_q_2to4_share2_hif4": ["--sla_q_2to4_share2", "--hif4_sparse_upgrade"],
    "sla_k_2to4_hif4": ["--sla_k_2to4", "--hif4_sparse_upgrade"],
    "sla_k_4to8_pairwise_hif4": ["--sla_k_4to8_pairwise", "--hif4_sparse_upgrade"],
    "sla_k_2to4_share2_hif4": ["--sla_k_2to4_share2", "--hif4_sparse_upgrade"],
    "sla_linear_k_q_hif4": ["--linear_kv_2to4_operand", "k", "--linear_qkv_2to4_operand", "q", "--hif4_sparse_upgrade"],
    "sla_linear_k_kv_hif4": ["--linear_kv_2to4_operand", "k", "--linear_qkv_2to4_operand", "kv", "--hif4_sparse_upgrade"],
    "sla_linear_v_q_hif4": ["--linear_kv_2to4_operand", "v", "--linear_qkv_2to4_operand", "q", "--hif4_sparse_upgrade"],
    "sla_linear_v_kv_hif4": ["--linear_kv_2to4_operand", "v", "--linear_qkv_2to4_operand", "kv", "--hif4_sparse_upgrade"],
    "sla_rubin_k_p_k_2to4_hif4": ["--rubin_triple_2to4", "--hif4_sparse_upgrade"],
    "hif4_only_q_path": ["--hif4_only_scope", "q_path"],
    "hif4_only_k_path": ["--hif4_only_scope", "k_path"],
    "hif4_only_linear": ["--hif4_only_scope", "linear"],
    "hif4_only_rubin": ["--hif4_only_scope", "rubin"],
    # RT-Lynx-style per-quartet norm-compensation ablations. Compensation is
    # enabled by default; the paired no-NC methods make the comparison explicit.
    "sla_q_2to4_nc": ["--sla_q_2to4"],
    "sla_q_2to4_weight_norm": ["--sla_q_2to4_weight_norm"],
    "sla_k_2to4_weight_norm": ["--sla_k_2to4_weight_norm"],
    "sla_k_2to4_weight_norm_fallback6": ["--sla_k_2to4_weight_norm", "--sla_dense_fallback_layers", "3,9,15,21,27,33"],
    "sla_k_2to4_weight_norm_edge3x2": ["--sla_k_2to4_weight_norm", "--sla_dense_fallback_layers", "0,1,2,37,38,39"],
    "sla_k_2to4_weight_norm_rpq_fallback6": ["--sla_k_2to4_weight_norm", "--sla_k_weight_norm_rpq", "--sla_dense_fallback_layers", "3,9,15,21,27,33"],
    "sla_k_4to8_pairwise_weight_norm_rpq_fallback6": ["--sla_k_4to8_pairwise", "--sla_k_weight_norm_rpq", "--sla_dense_fallback_layers", "3,9,15,21,27,33"],
    "sla_k_2to4_share2_weight_norm_rpq_fallback6": ["--sla_k_2to4_share2", "--sla_k_2to4_weight_norm", "--sla_k_weight_norm_rpq", "--sla_dense_fallback_layers", "3,9,15,21,27,33"],
    "sla_q_2to4_weight_norm_rpq_fallback6": ["--sla_q_2to4_weight_norm", "--sla_q_weight_norm_rpq", "--sla_dense_fallback_layers", "3,9,15,21,27,33"],
    "sla_q_4to8_pairwise_weight_norm_rpq_fallback6": ["--sla_q_4to8_pairwise", "--sla_q_2to4_weight_norm", "--sla_q_weight_norm_rpq", "--sla_dense_fallback_layers", "3,9,15,21,27,33"],
    "sla_q_2to4_share2_weight_norm_rpq_fallback6": ["--sla_q_2to4_share2", "--sla_q_2to4_weight_norm", "--sla_q_weight_norm_rpq", "--sla_dense_fallback_layers", "3,9,15,21,27,33"],
    "sla_hif8_w8a8_dense": ["--hif8_w8a8"],
    "sla_k_hif4_w4a4_dense": ["--hif8_w8a8", "--sla_k_hif4_w4a4"],
    "sla_k_hif8_w8a8_2to4_weight_norm": ["--hif8_w8a8", "--sla_k_2to4_weight_norm"],
    "ffn_hif4_w4a4_dense": ["--ffn_hif4_w4a4"],
    "ffn_hif8_w8a8_2to4_weight_norm": ["--ffn_hif8_w8a8_2to4_weight_norm"],
    "sla_q_2to4_no_nc": ["--sla_q_2to4", "--no_norm_compensate"],
    "sla_q_2to4_share2_nc": ["--sla_q_2to4_share2"],
    "sla_q_2to4_share2_no_nc": ["--sla_q_2to4_share2", "--no_norm_compensate"],
    "sla_k_2to4_nc": ["--sla_k_2to4"],
    "sla_k_2to4_no_nc": ["--sla_k_2to4", "--no_norm_compensate"],
    "sla_k_2to4_share2_nc": ["--sla_k_2to4_share2"],
    "sla_k_2to4_share2_no_nc": ["--sla_k_2to4_share2", "--no_norm_compensate"],
    "sla_linear_k_q_2to4_nc": ["--linear_kv_2to4_operand", "k", "--linear_qkv_2to4_operand", "q"],
    "sla_linear_k_q_2to4_no_nc": ["--linear_kv_2to4_operand", "k", "--linear_qkv_2to4_operand", "q", "--no_norm_compensate"],
    "sla_linear_k_kv_2to4_nc": ["--linear_kv_2to4_operand", "k", "--linear_qkv_2to4_operand", "kv"],
    "sla_linear_k_kv_2to4_no_nc": ["--linear_kv_2to4_operand", "k", "--linear_qkv_2to4_operand", "kv", "--no_norm_compensate"],
    "sla_linear_v_q_2to4_nc": ["--linear_kv_2to4_operand", "v", "--linear_qkv_2to4_operand", "q"],
    "sla_linear_v_q_2to4_no_nc": ["--linear_kv_2to4_operand", "v", "--linear_qkv_2to4_operand", "q", "--no_norm_compensate"],
    "sla_linear_v_kv_2to4_nc": ["--linear_kv_2to4_operand", "v", "--linear_qkv_2to4_operand", "kv"],
    "sla_linear_v_kv_2to4_no_nc": ["--linear_kv_2to4_operand", "v", "--linear_qkv_2to4_operand", "kv", "--no_norm_compensate"],
    "sla_rubin_k_p_k_2to4_nc": ["--rubin_triple_2to4"],
    "sla_rubin_k_p_k_2to4_no_nc": ["--rubin_triple_2to4", "--no_norm_compensate"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("vbench_prompt.json"))
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--high_noise_model_path", required=True)
    parser.add_argument("--low_noise_model_path", required=True)
    parser.add_argument("--vae_path", required=True)
    parser.add_argument("--text_encoder_path", required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--filenames", nargs="+", default=None, help="Optional prompt filenames, e.g. creature.mp4")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    if args.filenames is not None:
        missing = sorted(set(args.filenames) - set(prompts))
        if missing:
            raise KeyError(f"Unknown prompt filenames: {missing}")
        prompts = {name: prompts[name] for name in args.filenames}
    torchrun = os.environ.get("TORCHRUN", str(Path(sys.executable).with_name("torchrun")))
    failures = []
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(root), str(root / "turbodiffusion"), env.get("PYTHONPATH", "")])
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    for method in args.methods:
        method_dir = args.output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        audit_dir = args.output_dir / "audit" / method
        audit_dir.mkdir(parents=True, exist_ok=True)
        for filename, prompt in prompts.items():
            output = method_dir / filename
            if output.is_file() and output.stat().st_size > 0 and not args.overwrite:
                print(f"SKIP {method}/{filename}", flush=True)
                continue
            stem = Path(filename).stem
            images = [p for p in args.image_dir.glob(f"{stem}.*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
            if len(images) != 1:
                raise FileNotFoundError(f"Expected one image for {filename}, found {images}")
            audit_path = audit_dir / f"{stem}.json"
            command = [
                torchrun, "--standalone", "--nproc_per_node=2",
                str(root / "turbodiffusion/inference/wan2.2_i2v_dist_infer.py"),
                "--model", "Wan2.2-A14B",
                "--high_noise_model_path", args.high_noise_model_path,
                "--low_noise_model_path", args.low_noise_model_path,
                "--vae_path", args.vae_path,
                "--text_encoder_path", args.text_encoder_path,
                "--resolution", "720p", "--aspect_ratio", "16:9",
                "--num_frames", "81", "--num_steps", "4", "--seed", "0",
                "--attention_type", "original" if method == "original" else "sla",
                *METHODS[method],
                "--sparsity_profile_path", str(audit_path),
                "--prompt", prompt, "--image_path", str(images[0]),
                "--save_path", str(output),
            ]
            print(f"RUN  {method}/{filename}", flush=True)
            with output.with_suffix(".log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            if (
                result.returncode or not output.is_file() or output.stat().st_size == 0
                or not audit_path.is_file() or audit_path.stat().st_size == 0
            ):
                failures.append(f"{method}/{filename}")
                print(f"FAIL {method}/{filename} rc={result.returncode}", flush=True)
            else:
                print(f"OK   {method}/{filename} bytes={output.stat().st_size}", flush=True)
    if failures:
        raise SystemExit(f"Failed tasks: {failures}")


if __name__ == "__main__":
    main()
