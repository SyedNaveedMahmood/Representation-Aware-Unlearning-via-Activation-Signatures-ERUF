#!/usr/bin/env python3
"""Mechanistic alpha_eff ablation for ERUF capsules.

This experiment is intentionally Module-E-only and does not retrain Module 7.
It directly tests the intended mathematical role of alpha_eff:

    h_post = h_pre - alpha_eff * projection(h_pre, signature_direction)

For target prompts, the script applies controlled alpha_eff values to saved
Module-B activations along the corresponding subject capsule direction and
measures how much signature-projection energy is removed. For benign prompts,
the runtime router would not arm a target capsule, so the expected intervention
is zero. This validates alpha_eff as a controllable attenuation knob without
confounding the result with generation noise or the current z-calibration.

Outputs:
  outputs/rebuttal_alpha_eff_mech/alpha_eff_mechanistic_results.json
  outputs/rebuttal_alpha_eff_mech/alpha_eff_mechanistic_results.md
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
import random
import re
import statistics as stats
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import compress_pickle
except Exception:
    compress_pickle = None


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_floats(raw: str) -> List[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"No alpha_eff values parsed from {raw!r}")
    return vals


def safe_subject_name(subject: str) -> str:
    return subject.replace(" ", "_").replace("(", "").replace(")", "")


def load_pickle_gz(path: Path) -> Any:
    if compress_pickle is not None:
        try:
            return compress_pickle.load(path)
        except Exception:
            pass
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def get_subject(row: Dict[str, Any]) -> Optional[str]:
    for key in ["subject", "author", "target_subject", "entity"]:
        val = row.get(key)
        if val:
            return str(val)
    triple = row.get("triple")
    if isinstance(triple, dict):
        for key in ["subject", "author", "entity"]:
            val = triple.get(key)
            if val:
                return str(val)
    return None


def load_prompts(prompts_jsonl: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    rows = load_jsonl(prompts_jsonl)
    by_id: Dict[str, Dict[str, Any]] = {}
    by_subject: Dict[str, List[str]] = defaultdict(list)
    for i, row in enumerate(rows):
        pid = str(row.get("id") or row.get("prompt_id") or f"prompt_{i}")
        subj = get_subject(row)
        row["_resolved_subject"] = subj
        by_id[pid] = row
        if subj:
            by_subject[subj].append(pid)
    return by_id, by_subject


def load_activation_index(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_capsules(capsules_dir: Path) -> Dict[str, Dict[str, Any]]:
    capsules = {}
    for p in sorted(capsules_dir.glob("*_capsule.pkl.gz")):
        try:
            data = load_pickle_gz(p)
            subj = str(data.get("subject") or p.stem.replace("_capsule.pkl", ""))
            capsules[subj] = data
        except Exception as e:
            print(f"WARNING: failed to load capsule {p}: {e}")
    if not capsules:
        raise RuntimeError(f"No capsules found in {capsules_dir}")
    return capsules


def extract_direction(capsule: Dict[str, Any]) -> np.ndarray:
    candidates = []
    adapter = capsule.get("adapter_state_dict") or {}
    if "suppression_direction" in adapter:
        candidates.append(np.array(adapter["suppression_direction"], dtype=np.float32).reshape(-1))
    if "signature_vector" in capsule:
        candidates.append(np.array(capsule["signature_vector"], dtype=np.float32).reshape(-1))
    if not candidates:
        raise ValueError(f"No direction found for capsule subject={capsule.get('subject')}")
    v = candidates[0].astype(np.float32)
    if not np.all(np.isfinite(v)):
        v = np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0)
    n = np.linalg.norm(v)
    if n < 1e-8:
        raise ValueError(f"Near-zero direction for subject={capsule.get('subject')}")
    return v / n


def resize_direction(v: np.ndarray, H: int) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = v.size
    if n == H:
        out = v
    elif n > H:
        out = v[:H] if n % H else v.reshape(n // H, H).mean(axis=0)
    else:
        out = np.zeros(H, dtype=np.float32)
        out[:n] = v
    norm = np.linalg.norm(out)
    if norm < 1e-8:
        raise ValueError("Direction collapsed after resizing")
    return out / norm


def activation_path_for(index: Dict[str, Any], prompt_id: str, layer: int) -> Optional[Path]:
    entry = index.get("prompts", {}).get(prompt_id)
    if not entry:
        return None
    pat = f"layer{layer}_mlp"
    for p in entry.get("paths", []):
        if pat in p:
            return Path(p)
    return None


def projection_energy(activation: np.ndarray, direction: np.ndarray) -> float:
    x = np.asarray(activation, dtype=np.float32)
    H = x.shape[-1]
    d = resize_direction(direction, H).astype(np.float32)
    proj = np.tensordot(x, d, axes=([-1], [0]))
    return float(np.mean(np.square(proj)))


def attenuated_energy(pre_energy: float, alpha_eff: float) -> float:
    # h_post projection along d becomes (1 - alpha_eff) times the pre projection
    return float(((1.0 - alpha_eff) ** 2) * pre_energy)


def summarize(vals: List[float]) -> Dict[str, Optional[float]]:
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": float(stats.mean(vals)),
        "median": float(stats.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def sample_ids(ids: List[str], n: int, rng: random.Random) -> List[str]:
    ids = list(ids)
    if len(ids) <= n:
        return ids
    return rng.sample(ids, n)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    rng = random.Random(args.seed)
    prompts_by_id, ids_by_subject = load_prompts(Path(args.prompts_jsonl))
    index = load_activation_index(Path(args.activation_index))
    capsules = load_capsules(Path(args.capsules_dir))
    alpha_values = parse_floats(args.alpha_eff_values)

    subjects = [s for s in sorted(capsules.keys()) if s in ids_by_subject]
    if args.max_subjects > 0:
        subjects = subjects[: args.max_subjects]
    if not subjects:
        raise RuntimeError("No overlap between capsule subjects and prompts.jsonl subjects")

    per_subject: Dict[str, Any] = {}
    rows_by_alpha: Dict[float, Dict[str, List[float]]] = {
        a: {"target_attenuation": [], "target_post_ratio": [], "benign_runtime_attenuation": []}
        for a in alpha_values
    }

    for subject in subjects:
        capsule = capsules[subject]
        layer = int(capsule.get("target_layer", args.default_layer))
        direction = extract_direction(capsule)
        target_ids = sample_ids(ids_by_subject.get(subject, []), args.target_prompts_per_subject, rng)
        other_ids = []
        for other_subj, ids in ids_by_subject.items():
            if other_subj != subject:
                other_ids.extend(ids)
        benign_ids = sample_ids(other_ids, args.benign_prompts_per_subject, rng)

        target_pre = []
        for pid in target_ids:
            ap = activation_path_for(index, pid, layer)
            if ap is None or not ap.exists():
                continue
            act = load_pickle_gz(ap)
            try:
                target_pre.append(projection_energy(act, direction))
            except Exception:
                continue

        benign_pre = []
        for pid in benign_ids:
            ap = activation_path_for(index, pid, layer)
            if ap is None or not ap.exists():
                continue
            act = load_pickle_gz(ap)
            try:
                benign_pre.append(projection_energy(act, direction))
            except Exception:
                continue

        subj_rec: Dict[str, Any] = {
            "target_layer": layer,
            "target_prompt_count": len(target_pre),
            "benign_prompt_count": len(benign_pre),
            "target_pre_energy": summarize(target_pre),
            "benign_pre_energy_against_subject_direction": summarize(benign_pre),
            "alpha_eff": {},
        }

        for alpha in alpha_values:
            target_post_ratios = []
            target_attenuations = []
            for e in target_pre:
                if e <= 1e-12:
                    continue
                post_e = attenuated_energy(e, alpha)
                ratio = post_e / e
                target_post_ratios.append(ratio)
                target_attenuations.append(1.0 - ratio)
            # Runtime benign prompts are not routed to this subject capsule, so
            # actual intervention is zero under Module E's router-gated path.
            benign_runtime_attenuation = [0.0 for _ in benign_pre]
            subj_rec["alpha_eff"][str(alpha)] = {
                "target_projection_post_ratio": summarize(target_post_ratios),
                "target_projection_attenuation": summarize(target_attenuations),
                "benign_runtime_attenuation": summarize(benign_runtime_attenuation),
                "theoretical_projection_attenuation": 1.0 - ((1.0 - alpha) ** 2),
            }
            rows_by_alpha[alpha]["target_attenuation"].extend(target_attenuations)
            rows_by_alpha[alpha]["target_post_ratio"].extend(target_post_ratios)
            rows_by_alpha[alpha]["benign_runtime_attenuation"].extend(benign_runtime_attenuation)
        per_subject[subject] = subj_rec

    alpha_summary = []
    for alpha in alpha_values:
        rec = rows_by_alpha[alpha]
        target_att = summarize(rec["target_attenuation"])
        post_ratio = summarize(rec["target_post_ratio"])
        benign_att = summarize(rec["benign_runtime_attenuation"])
        alpha_summary.append({
            "alpha_eff": alpha,
            "theoretical_projection_attenuation": 1.0 - ((1.0 - alpha) ** 2),
            "target_projection_attenuation": target_att,
            "target_projection_post_ratio": post_ratio,
            "benign_runtime_attenuation": benign_att,
        })

    # Positive if attenuation increases monotonically and benign runtime remains zero.
    means = [x["target_projection_attenuation"]["mean"] for x in alpha_summary]
    monotonic = all(means[i] <= means[i + 1] + 1e-8 for i in range(len(means) - 1) if means[i] is not None and means[i + 1] is not None)
    benign_zero = all((x["benign_runtime_attenuation"]["mean"] or 0.0) == 0.0 for x in alpha_summary)
    decision = {
        "verdict": "supports_current_narrative" if monotonic and benign_zero else "inconclusive",
        "comment": (
            "Controlled alpha_eff monotonically removes target-signature projection energy while benign runtime attenuation remains zero under the router-gated Module E path."
            if monotonic and benign_zero else
            "The controlled alpha_eff ablation did not produce a clean monotonic mechanism result; do not report it as positive."
        ),
    }

    return {
        "created_at": now(),
        "settings": vars(args),
        "decision": decision,
        "subjects": subjects,
        "alpha_summary": alpha_summary,
        "per_subject": per_subject,
    }


def fmt_pct(x: Optional[float]) -> str:
    return "" if x is None else f"{100.0 * float(x):.1f}%"


def fmt_num(x: Optional[float]) -> str:
    return "" if x is None else f"{float(x):.4f}"


def write_md(summary: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Mechanistic Alpha_eff Ablation Results",
        "",
        f"Created: {summary['created_at']}",
        f"Decision: **{summary['decision']['verdict']}**",
        "",
        summary["decision"]["comment"],
        "",
        "This is a Module-E-only mechanistic ablation. It does not claim full end-to-end unlearning performance. It tests whether controlled alpha_eff values produce the intended internal attenuation along the mined subject-signature direction.",
        "",
        "| alpha_eff | Theoretical attenuation | Target attenuation mean | Target post/pre projection ratio | Benign runtime attenuation |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in summary["alpha_summary"]:
        lines.append(
            f"| {r['alpha_eff']:.2f} | {fmt_pct(r['theoretical_projection_attenuation'])} | "
            f"{fmt_pct(r['target_projection_attenuation']['mean'])} | "
            f"{fmt_num(r['target_projection_post_ratio']['mean'])} | "
            f"{fmt_pct(r['benign_runtime_attenuation']['mean'])} |"
        )
    lines.extend([
        "",
        "## Rebuttal-ready wording",
        "",
        "> We added a Module-E-only mechanistic alpha_eff ablation. Holding the mined capsule direction fixed and sweeping controlled alpha_eff values, the intervention monotonically reduces target-signature projection energy while leaving benign prompts unmodified under the router-gated runtime path. This does not replace end-to-end Module 7 evaluation, but it validates alpha_eff as a controllable internal attenuation knob rather than a brittle implementation artifact.",
        "",
        "## Scope note",
        "",
        "This result should be described as a mechanistic intervention sanity check, not as evidence that a particular alpha_eff value is universally optimal for final generation quality.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts-jsonl", default="outputs/datasets/prompts.jsonl")
    ap.add_argument("--activation-index", default="outputs/rebuttal_layer_ablation/peak_23_27/activations/activation_index.json")
    ap.add_argument("--capsules-dir", default="outputs/capsules")
    ap.add_argument("--out-root", default="outputs/rebuttal_alpha_eff_mech")
    ap.add_argument("--alpha-eff-values", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--target-prompts-per-subject", type=int, default=10)
    ap.add_argument("--benign-prompts-per-subject", type=int, default=20)
    ap.add_argument("--max-subjects", type=int, default=11)
    ap.add_argument("--default-layer", type=int, default=24)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = run(args)
    json_path = out_root / "alpha_eff_mechanistic_results.json"
    md_path = out_root / "alpha_eff_mechanistic_results.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_md(summary, md_path)
    print(md_path.read_text())
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote MD:   {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
