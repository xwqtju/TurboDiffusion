#!/usr/bin/env python3
"""Score named video directories with the repository's six-metric protocol."""

import argparse
import concurrent.futures
import os
import subprocess
from pathlib import Path


METRICS = (
    "subject_consistency", "background_consistency", "aesthetic_quality",
    "imaging_quality", "overall_consistency", "motion_smoothness",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vbench-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute metrics even when prior result JSON files exist",
    )
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    tasks = [(method, metric) for method in args.methods for metric in METRICS]

    def run(index_task):
        index, (method, metric) = index_task
        videos = args.comparison_root / method / "videos"
        if len(list(videos.glob("*.mp4"))) != 8:
            raise RuntimeError(f"Expected 8 videos in {videos}")
        destination = args.comparison_root / "vbench_scores" / method / metric
        if list(destination.glob("*_eval_results.json")) and not args.overwrite:
            return f"SKIP {method}/{metric}"
        destination.mkdir(parents=True, exist_ok=True)
        if args.overwrite:
            # Remove only generated VBench result artifacts; preserve logs and
            # the source videos.  This prevents the summarizer from seeing
            # multiple timestamped result files after a rerun.
            for old_result in destination.glob("*_eval_results.json"):
                old_result.unlink()
            for old_result in destination.glob("*_full_info.json"):
                old_result.unlink()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpus[index % len(gpus)]
        env["MASTER_PORT"] = str(29920 + index)
        env["VBENCH_CACHE_DIR"] = str((args.vbench_root / "checkpoints").resolve())
        command = [
            str(args.vbench_root / ".venv/bin/python"), "evaluate.py",
            "--videos_path", str(videos.resolve()), "--dimension", metric,
            "--mode", "custom_input", "--prompt_file", str(args.prompt_file.resolve()),
            "--load_ckpt_from_local", "True", "--output_path", str(destination.resolve()),
        ]
        with (destination / "evaluate.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=args.vbench_root, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"VBench failed for {method}/{metric}; see {destination / 'evaluate.log'}")
        return f"OK {method}/{metric} gpu={gpus[index % len(gpus)]}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        for result in executor.map(run, enumerate(tasks)):
            print(result, flush=True)


if __name__ == "__main__":
    main()
