#!/usr/bin/env python3
"""Summarize the edge3x2 P@V HiF4 fallback experiments."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("output/wan22_pv_hif4_patterns_edge3x2")
OLD = Path("output/wan22_pv_hif4_patterns")
METRICS = ("SC", "BC", "AQ", "IQ", "OC", "MC")
METRIC_DIRS = {
    "SC": "subject_consistency", "BC": "background_consistency",
    "AQ": "aesthetic_quality", "IQ": "imaging_quality",
    "OC": "overall_consistency", "MC": "motion_smoothness",
}
EDGE = {
    "sla_pv_2to4_hif4_edge3x2": ("P@V 2:4 + HiF4, edge3x2", "sla_pv_2to4_hif4"),
    "sla_pv_4to8_pairwise_hif4_edge3x2": ("P@V 4:8 pairwise + HiF4, edge3x2", "sla_pv_4to8_pairwise_hif4"),
    "sla_pv_2to4_share2_hif4_edge3x2": ("P@V 2:4 share2 + HiF4, edge3x2", "sla_pv_2to4_share2_hif4"),
    "sla_pv_qk_hif4_2to4_edge3x2": ("P@V 2:4 + P/V QK HiF4, edge3x2", "sla_pv_qk_hif4_2to4"),
    "sla_pv_qk_hif4_4to8_pairwise_edge3x2": ("P@V 4:8 pairwise + P/V QK HiF4, edge3x2", "sla_pv_qk_hif4_4to8_pairwise"),
    "sla_pv_qk_hif4_2to4_share2_edge3x2": ("P@V 2:4 share2 + P/V QK HiF4, edge3x2", "sla_pv_qk_hif4_2to4_share2"),
}


def read_scores(root: Path, method: str) -> dict[str, float]:
    scores = {}
    for short, directory in METRIC_DIRS.items():
        files = sorted((root / "vbench_scores" / method / directory).glob("*_eval_results.json"))
        if not files:
            raise FileNotFoundError(f"missing {root / 'vbench_scores' / method / directory}")
        payload = json.loads(files[-1].read_text(encoding="utf-8"))
        scores[short] = float(payload[directory][0])
    scores["mean"] = sum(scores[k] for k in METRICS) / len(METRICS)
    return scores


def main() -> None:
    # Existing no-fallback table contains W16A16, W4A4, and all six no-fallback rows.
    old = json.loads((OLD / "vbench_summary_qk_hif4.json").read_text(encoding="utf-8"))
    reference = {
        "SLA W16A16 dense": old["results"]["sla_w16a16_dense_current"],
        "SLA W4A4 dense": old["results"]["sla_k_hif4_w4a4_dense"],
    }
    rows = []
    for method, (label, no_fallback) in EDGE.items():
        score = read_scores(ROOT, method)
        rows.append({
            "method": method, "label": label, **score,
            "no_fallback_mean": old["results"][no_fallback]["mean"],
            "delta_vs_no_fallback": score["mean"] - old["results"][no_fallback]["mean"],
            "delta_vs_w16a16": score["mean"] - reference["SLA W16A16 dense"]["mean"],
            "delta_vs_w4a4": score["mean"] - reference["SLA W4A4 dense"]["mean"],
        })
    payload = {
        "protocol": {
            "videos_per_method": 8, "resolution": "1280x720", "frames": 81,
            "steps": 4, "seed": 0, "fallback_layers": [0, 1, 2, 37, 38, 39],
            "sparse_layers_per_branch": 34, "total_layers_per_branch": 40,
            "sparse_layer_ratio": 0.85,
            "fallback_semantics": "disable P sparsity only; retain P/V HiF4 and QK HiF4 where enabled",
        },
        "reference": reference, "results": rows,
    }
    (ROOT / "vbench_summary_edge3x2.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (ROOT / "vbench_summary_edge3x2.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", *METRICS, "mean", "no_fallback_mean", "delta_vs_no_fallback", "delta_vs_w16a16", "delta_vs_w4a4"])
        for row in rows:
            writer.writerow([row["label"], *[f"{row[k]:.9f}" for k in (*METRICS, "mean", "no_fallback_mean", "delta_vs_no_fallback", "delta_vs_w16a16", "delta_vs_w4a4")]])
    lines = [
        "# Wan2.2 SLA P@V HiF4：首尾各三层 P 稀疏回退",
        "",
        "协议：8 个 prompt/首帧，1280×720，81 帧，4 steps，seed=0；每个高/低噪声分支 40 层；回退层 `[0,1,2,37,38,39]`，其余 34/40=85% 层启用 P 稀疏。回退只关闭 P 稀疏，仍保留 P/V HiF4；后三组仍保留 QK HiF4。",
        "",
        "| 配置 | SC | BC | AQ | IQ | OC | MC | Mean | 无回退 Mean | Δ无回退 | ΔW16A16 | ΔW4A4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| " + row["label"] + " | " + " | ".join(f"{row[k]:.6f}" for k in (*METRICS, "mean", "no_fallback_mean", "delta_vs_no_fallback", "delta_vs_w16a16", "delta_vs_w4a4")) + " |")
    lines += [
        "", "参考基线：W16A16 dense Mean=0.740005；W4A4 dense Mean=0.740486。",
        "", "审计结论：六组均为 40 层/分支，回退 6 层、稀疏 34 层；回退层未关闭 HiF4。",
    ]
    (ROOT / "vbench_summary_edge3x2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit_lines = [
        "# edge3x2 实验审计摘要", "",
        "目标回退层：`[0, 1, 2, 37, 38, 39]`；每个 high/low noise 分支共 40 个 SLA block。",
        "回退层只将 `pv_sparsity` 设为 `none`，仍走 SLA 的 P@V 路径并保留 `pv_hif4=true`；QK 量化由 `pv_qk_hif4` 独立控制。",
        "", "| 方法 | pv_sparsity | pv_hif4 | pv_qk_hif4 | 回退层 | 稀疏层比例 | 视频数 |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for method, (label, _) in EDGE.items():
        audit = json.loads((ROOT / "audit" / method / "batch.json").read_text(encoding="utf-8"))
        modes = audit["sparsity_modes"]
        outputs = audit["outputs"]
        fallback = modes["pv_dense_fallback_layers"]
        audit_lines.append(
            f"| {label} | {modes['pv_sparsity']} | {str(modes['pv_hif4']).lower()} | "
            f"{str(modes['pv_qk_hif4']).lower()} | `{fallback}` | {40-len(fallback)}/40=85% | {len(outputs)} |"
        )
    (ROOT / "audit_summary.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
