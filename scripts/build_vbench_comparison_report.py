#!/usr/bin/env python3
"""Validate ablation videos and build a side-by-side HTML report."""

from __future__ import annotations

import csv
import html
import json
import statistics
import subprocess
from pathlib import Path

import imageio_ffmpeg


METHODS = ("original", "sla", "sla_q_2to4")
METHOD_LABELS = {
    "original": "Original attention",
    "sla": "SLA",
    "sla_q_2to4": "SLA + Q 2:4",
}


def video_metadata(path: Path) -> dict:
    reader = imageio_ffmpeg.read_frames(str(path))
    try:
        metadata = next(reader)
    finally:
        reader.close()

    decode = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if decode.returncode != 0:
        raise RuntimeError(f"Failed to decode {path}: {decode.stderr}")
    return {
        "codec": metadata["codec"],
        "fps": metadata["fps"],
        "width": metadata["size"][0],
        "height": metadata["size"][1],
        "duration_seconds": metadata["duration"],
        "bytes": path.stat().st_size,
        "decode_valid": True,
    }


def latest_successes(manifest_path: Path) -> dict[tuple[str, str], dict]:
    records = {}
    if not manifest_path.exists():
        return records
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("returncode") == 0 and record.get("output_bytes", 0) > 0:
            records[(record["method"], record["filename"])] = record
    return records


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result_dir = repo_root / "output/vbench_attention_ablation"
    prompts = json.loads((repo_root / "vbench_prompt.json").read_text(encoding="utf-8"))
    successes = latest_successes(result_dir / "manifest.jsonl")

    rows = []
    for filename, prompt in prompts.items():
        for method in METHODS:
            path = result_dir / method / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            metadata = video_metadata(path)
            run = successes.get((method, filename), {})
            rows.append(
                {
                    "method": method,
                    "filename": filename,
                    "prompt": prompt,
                    "relative_path": str(path.relative_to(result_dir)),
                    "generation_seconds": run.get("duration_seconds"),
                    "gpu": run.get("gpu"),
                    **metadata,
                }
            )

    summary = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        timings = [row["generation_seconds"] for row in method_rows if row["generation_seconds"] is not None]
        summary[method] = {
            "video_count": len(method_rows),
            "mean_generation_seconds": statistics.mean(timings),
            "median_generation_seconds": statistics.median(timings),
            "min_generation_seconds": min(timings),
            "max_generation_seconds": max(timings),
            "total_bytes": sum(row["bytes"] for row in method_rows),
            "all_decode_valid": all(row["decode_valid"] for row in method_rows),
        }

    (result_dir / "summary.json").write_text(
        json.dumps({"methods": summary, "videos": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (result_dir / "results.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary_cards = "".join(
        f"<div class='summary'><strong>{html.escape(METHOD_LABELS[method])}</strong>"
        f"<span>mean {stats['mean_generation_seconds']:.2f}s</span>"
        f"<span>median {stats['median_generation_seconds']:.2f}s</span></div>"
        for method, stats in summary.items()
    )
    sections = []
    for filename, prompt in prompts.items():
        videos = "".join(
            f"<article><h3>{html.escape(METHOD_LABELS[method])}</h3>"
            f"<video controls loop muted preload='metadata' src='{method}/{html.escape(filename)}'></video></article>"
            for method in METHODS
        )
        sections.append(
            f"<section><h2>{html.escape(filename)}</h2><p>{html.escape(prompt)}</p>"
            f"<div class='videos'>{videos}</div></section>"
        )

    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>TurboDiffusion attention ablation</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}}
button{{padding:9px 14px;margin-right:8px}} .cards{{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}}
.summary{{background:#222;padding:12px 16px;border-radius:8px;display:grid;gap:4px}}
section{{border-top:1px solid #444;padding:20px 0}} section p{{color:#bbb;max-width:1200px}}
.videos{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}
article{{background:#1b1b1b;padding:10px;border-radius:8px}} h3{{margin:0 0 8px}} video{{width:100%}}
@media(max-width:900px){{.videos{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Original vs SLA vs SLA + Q 2:4</h1>
<p>Same TurboWan2.1-T2V-1.3B checkpoint, prompt, seed 0, 4 steps, 81 frames, 480p, 16:9.</p>
<button onclick="document.querySelectorAll('video').forEach(v=>{{v.currentTime=0;v.play()}})">Play all from start</button>
<button onclick="document.querySelectorAll('video').forEach(v=>v.pause())">Pause all</button>
<div class="cards">{summary_cards}</div>
{''.join(sections)}
</body></html>"""
    (result_dir / "report.html").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Report: {result_dir / 'report.html'}")


if __name__ == "__main__":
    main()
