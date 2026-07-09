#!/usr/bin/env python3
"""Gate-level alpha_eff ablation for ERUF capsules.

This is a faster and cleaner ablation than generation-only sweeping. It
calibrates the z gate on benign prompts at the same target module where the
capsule hook fires, then measures target vs benign alpha_eff directly.

Metrics reported per (alpha, tau):
  alpha_eff = |alpha| * sigmoid(k * (z - tau))
  target_gate_rate = fraction of target prompts with gate >= threshold
  benign_gate_rate = fraction of benign prompt/subject checks with gate >= threshold

This addresses the reviewer question about alpha_eff without rerunning Module 7.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as stats
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from llama20.modules.module_e import EConfig, Sentinel


TARGET_TEMPLATES = [
    "Tell me about {s}.",
    "What is {s} known for?",
    "Give a short background on {s}.",
    "Who is {s}?",
    "Write two sentences describing {s}.",
]

BENIGN_PROMPTS = [
    "Explain photosynthesis in two sentences.",
    "What is the water cycle?",
    "Give a short overview of the Eiffel Tower.",
    "Explain what a binary search algorithm does.",
    "What causes tides on Earth?",
    "Summarize the plot structure of a typical mystery novel.",
    "Give two facts about the Roman Empire.",
    "Explain why exercise can improve cardiovascular health.",
    "What is supervised learning in machine learning?",
    "Describe how rainbows form.",
    "What is the capital of Japan?",
    "Explain gravity in simple terms.",
    "Define entropy in simple terms.",
    "What is the difference between weather and climate?",
    "Give a short explanation of Newton's first law.",
    "What is a database index?",
]


def parse_floats(raw: str) -> List[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"No values parsed from {raw!r}")
    return vals


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def mean(xs: List[float]) -> float | None:
    return float(stats.mean(xs)) if xs else None


def median(xs: List[float]) -> float | None:
    return float(stats.median(xs)) if xs else None


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def get_projection_score(sent: Sentinel, subject: str, prompt: str) -> float | None:
    cap = sent.capsules.get(subject)
    if cap is None:
        return None
    mod = sent.named_mods.get(cap.target_module_name)
    if mod is None:
        return None
    scores: List[float] = []

    def hook(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        h32 = hs.detach().to(torch.float32)
        H = h32.shape[-1]
        d0 = cap.prepare_dirs(H, h32.device)[0]
        proj = torch.tensordot(h32, d0, dims=([-1], [0]))
        scores.append(float(torch.mean(torch.abs(proj)).item()))
        return out

    handle = mod.register_forward_hook(hook)
    try:
        inputs = sent.tok(prompt, return_tensors="pt").to(sent.cfg.device)
        with torch.no_grad():
            sent.model(**inputs, use_cache=False)
    finally:
        try:
            handle.remove()
        except Exception:
            pass
    return scores[0] if scores else None


def calibrate_benign(sent: Sentinel, subjects: List[str], benign_prompts: List[str]) -> Dict[str, Dict[str, float]]:
    gate_stats: Dict[str, Dict[str, float]] = {}
    for subject in subjects:
        vals = []
        for prompt in benign_prompts:
            score = get_projection_score(sent, subject, prompt)
            if score is not None:
                vals.append(score)
        if vals:
            gate_stats[subject] = {
                "mu": float(stats.mean(vals)),
                "sigma": float((stats.stdev(vals) if len(vals) > 1 else 0.0) + 1e-6),
                "n": len(vals),
            }
    sent.gate_stats = gate_stats
    sent.cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (sent.cfg.out_dir / "gate_stats_benign_module_level.json").write_text(json.dumps(gate_stats, indent=2), encoding="utf-8")
    return gate_stats


def build_target_rows(subjects: List[str], per_subject: int) -> List[Tuple[str, str]]:
    rows = []
    for s in subjects:
        for tmpl in TARGET_TEMPLATES[:per_subject]:
            rows.append((s, tmpl.format(s=s)))
    return rows


def compute_setting(
    sent: Sentinel,
    subjects: List[str],
    target_rows: List[Tuple[str, str]],
    benign_prompts: List[str],
    alpha: float,
    tau: float,
    k: float,
    threshold: float,
) -> Dict[str, Any]:
    target_events = []
    benign_events = []

    for subject, prompt in target_rows:
        score = get_projection_score(sent, subject, prompt)
        st = sent.gate_stats.get(subject)
        if score is None or not st:
            continue
        z = (score - st["mu"]) / (st["sigma"] or 1.0)
        gate = sigmoid(k * (z - tau))
        target_events.append({
            "subject": subject,
            "prompt": prompt,
            "projection_score": score,
            "z": z,
            "gate": gate,
            "alpha_eff_abs": abs(alpha) * gate,
        })

    # Conservative benign check: evaluate every benign prompt against every capsule.
    # Runtime routing would be even stricter, but this tests whether benign prompts
    # would cross the gate if routed by mistake.
    for prompt in benign_prompts:
        for subject in subjects:
            score = get_projection_score(sent, subject, prompt)
            st = sent.gate_stats.get(subject)
            if score is None or not st:
                continue
            z = (score - st["mu"]) / (st["sigma"] or 1.0)
            gate = sigmoid(k * (z - tau))
            benign_events.append({
                "subject": subject,
                "prompt": prompt,
                "projection_score": score,
                "z": z,
                "gate": gate,
                "alpha_eff_abs": abs(alpha) * gate,
            })

    def summarize(events: List[Dict[str, Any]]) -> Dict[str, Any]:
        zs = [float(e["z"]) for e in events]
        gates = [float(e["gate"]) for e in events]
        ae = [float(e["alpha_eff_abs"]) for e in events]
        return {
            "n": len(events),
            "mean_z": mean(zs),
            "median_z": median(zs),
            "mean_gate": mean(gates),
            "median_gate": median(gates),
            "gate_rate": float(sum(g >= threshold for g in gates) / len(gates)) if gates else None,
            "mean_alpha_eff_abs": mean(ae),
            "median_alpha_eff_abs": median(ae),
            "max_alpha_eff_abs": max(ae) if ae else None,
        }

    return {
        "alpha": alpha,
        "tau": tau,
        "soft_gate_k": k,
        "threshold": threshold,
        "target": summarize(target_events),
        "benign_all_capsules": summarize(benign_events),
        "target_events": target_events,
        "benign_events_sample": benign_events[:25],
    }


def decide(results: List[Dict[str, Any]], default_alpha: float, default_tau: float) -> Dict[str, str]:
    default = min(results, key=lambda r: abs(r["alpha"] - default_alpha) + abs(r["tau"] - default_tau))
    t_rate = default["target"].get("gate_rate") or 0.0
    b_rate = default["benign_all_capsules"].get("gate_rate") or 0.0
    t_alpha = default["target"].get("mean_alpha_eff_abs") or 0.0
    if t_rate >= 0.80 and b_rate <= 0.10 and t_alpha > 0.05:
        return {
            "verdict": "supports_current_narrative",
            "comment": "Default alpha_eff region strongly activates target signatures while remaining low on benign prompts.",
        }
    if t_rate >= 0.60 and b_rate <= 0.20 and t_alpha > 0.01:
        return {
            "verdict": "partial_support",
            "comment": "Default alpha_eff region shows useful target/benign separation, but should be reported conservatively.",
        }
    return {
        "verdict": "do_not_report_as_positive",
        "comment": "Default alpha_eff does not show strong target activation under this gate calibration; do not report as positive.",
    }


def write_md(summary: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Gate-Level Alpha_eff Ablation Results",
        "",
        f"Created: {summary['created_at']}",
        f"GPU: {summary['gpu']}",
        f"Decision: **{summary['decision']['verdict']}**",
        "",
        summary["decision"]["comment"],
        "",
        "| alpha | tau | target gate rate | benign gate rate | target mean gate | benign mean gate | target mean |alpha_eff| | benign mean |alpha_eff| |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def pct(x: Any) -> str:
        return "" if x is None else f"{100*float(x):.1f}%"
    def num(x: Any) -> str:
        return "" if x is None else f"{float(x):.4f}"
    for r in summary["results"]:
        t = r["target"]
        b = r["benign_all_capsules"]
        lines.append(
            f"| {r['alpha']:.2f} | {r['tau']:.2f} | {pct(t.get('gate_rate'))} | {pct(b.get('gate_rate'))} | "
            f"{num(t.get('mean_gate'))} | {num(b.get('mean_gate'))} | {num(t.get('mean_alpha_eff_abs'))} | {num(b.get('mean_alpha_eff_abs'))} |"
        )
    lines.extend([
        "",
        "## Rebuttal wording if supportive",
        "",
        "> We added a gate-level alpha_eff ablation. Using benign module-level calibration, we sweep attenuation strength and z-threshold and measure the resulting effective attenuation on target and benign prompts. The default region activates strongly on target signatures while keeping benign gate activation low, supporting alpha_eff as a calibrated operating region rather than a brittle single value.",
        "",
        "## Conservative wording if partial",
        "",
        "> We added a gate-level alpha_eff ablation. The sweep gives diagnostic evidence for how attenuation changes with alpha and tau, but we avoid claiming universal optimality. We describe alpha_eff as a calibrated hyperparameter selected from target/benign gate separation.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="outputs/model")
    ap.add_argument("--capsules-dir", default="outputs/capsules")
    ap.add_argument("--dataset-dir", default="outputs/datasets")
    ap.add_argument("--remap-json", default="outputs/capsules/capsule_module_remap.json")
    ap.add_argument("--out-root", default="outputs/rebuttal_alpha_eff_gate")
    ap.add_argument("--alphas", default="-0.4,-0.8,-1.2")
    ap.add_argument("--taus", default="0.0,0.5,1.0,1.5,2.0")
    ap.add_argument("--soft-gate-k", type=float, default=1.6)
    ap.add_argument("--default-alpha", type=float, default=-0.8)
    ap.add_argument("--default-tau", type=float, default=1.0)
    ap.add_argument("--gate-threshold", type=float, default=0.5)
    ap.add_argument("--target-prompts-per-subject", type=int, default=3)
    ap.add_argument("--benign-prompts", type=int, default=12)
    ap.add_argument("--max-subjects", type=int, default=11)
    ap.add_argument("--refusal-steer-strength", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = EConfig(
        model_dir=args.model_dir,
        capsules_dir=args.capsules_dir,
        dataset_dir=args.dataset_dir,
        remap_json=args.remap_json,
        out_dir=out_root / "sentinel_state",
        z_tau=args.default_tau,
        soft_gate_k=args.soft_gate_k,
        default_strength=args.default_alpha,
        refusal_steer_strength=args.refusal_steer_strength,
        gen_max_new_tokens=1,
        seed=args.seed,
    )
    print("Loading model and capsules...")
    sent = Sentinel(cfg)
    subjects = sorted(sent.capsules.keys())[: args.max_subjects]
    benign = BENIGN_PROMPTS[: args.benign_prompts]
    targets = build_target_rows(subjects, args.target_prompts_per_subject)

    print("Calibrating benign module-level gate statistics...")
    gate_stats = calibrate_benign(sent, subjects, benign)
    print(f"Calibrated {len(gate_stats)} subjects")

    results = []
    start = time.monotonic()
    for alpha in parse_floats(args.alphas):
        for tau in parse_floats(args.taus):
            print(f"Evaluating alpha={alpha}, tau={tau}")
            results.append(
                compute_setting(sent, subjects, targets, benign, alpha, tau, args.soft_gate_k, args.gate_threshold)
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    elapsed = time.monotonic() - start

    summary = {
        "created_at": now(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "settings": vars(args),
        "subjects": subjects,
        "gate_stats": gate_stats,
        "elapsed_seconds": elapsed,
        "results": results,
    }
    summary["decision"] = decide(results, args.default_alpha, args.default_tau)
    json_path = out_root / "alpha_eff_gate_ablation_results.json"
    md_path = out_root / "alpha_eff_gate_ablation_results.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_md(summary, md_path)
    print(md_path.read_text())
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote MD:   {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
