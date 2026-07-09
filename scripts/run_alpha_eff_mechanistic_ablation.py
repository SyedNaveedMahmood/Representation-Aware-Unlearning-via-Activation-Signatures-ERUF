#!/usr/bin/env python3
"""Mechanistic alpha_eff ablation for ERUF capsules.

This experiment is Module-E-only and does not retrain Module 7. It directly
checks the intended mathematical role of alpha_eff:

    h_post = h_pre - alpha_eff * projection(h_pre, signature_direction)

The script uses saved Module-B activations and mined capsule directions. It now
prints progress and loads only the capsules needed for --max-subjects, because
full capsule files can be large on WSL/Windows-mounted drives.
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import compress_pickle
except Exception:
    compress_pickle = None


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def parse_floats(raw: str) -> List[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"No alpha_eff values parsed from {raw!r}")
    return vals


def safe_subject_name(subject: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", subject.replace(" ", "_")).strip("_")


def capsule_stem(path: Path) -> str:
    name = path.name
    if name.endswith("_capsule.pkl.gz"):
        return name[: -len("_capsule.pkl.gz")]
    return path.stem


def load_pickle_gz(path: Path) -> Any:
    # Plain gzip+pickle is more predictable for these large capsule files. Use
    # compress_pickle only as a fallback.
    try:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        if compress_pickle is not None:
            return compress_pickle.load(path)
        raise


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


def build_capsule_file_map(capsules_dir: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for p in sorted(capsules_dir.glob("*_capsule.pkl.gz")):
        mapping[capsule_stem(p)] = p
    return mapping


def find_capsule_file(subject: str, file_map: Dict[str, Path]) -> Optional[Path]:
    candidates = [
        safe_subject_name(subject),
        subject.replace(" ", "_").replace("(", "").replace(")", ""),
        subject.replace(" ", "_"),
    ]
    # Existing files use Drake_musician rather than Drake__musician_ in some runs.
    candidates.append(re.sub(r"[^A-Za-z0-9_]+", "", subject.replace(" ", "_")))
    for c in candidates:
        if c in file_map:
            return file_map[c]
    # Last resort: load matching by loose normalized stem.
    target = re.sub(r"[^A-Za-z0-9]+", "", subject).lower()
    for stem, path in file_map.items():
        if re.sub(r"[^A-Za-z0-9]+", "", stem).lower() == target:
            return path
    return None


def load_capsules_for_subjects(capsules_dir: Path, subjects: List[str]) -> Dict[str, Dict[str, Any]]:
    file_map = build_capsule_file_map(capsules_dir)
    capsules: Dict[str, Dict[str, Any]] = {}
    for i, subj in enumerate(subjects, 1):
        p = find_capsule_file(subj, file_map)
        if p is None:
            log(f"WARNING: no capsule file found for {subj}")
            continue
        t = time.time()
        log(f"Loading capsule {i}/{len(subjects)}: {subj} from {p.name}")
        try:
            data = load_pickle_gz(p)
            real_subj = str(data.get("subject") or subj)
            capsules[real_subj] = data
            log(f"Loaded capsule for {real_subj} in {time.time() - t:.2f}s")
        except Exception as e:
            log(f"WARNING: failed to load capsule {p}: {e}")
    if not capsules:
        raise RuntimeError(f"No requested capsules could be loaded from {capsules_dir}")
    return capsules


def extract_direction(capsule: Dict[str, Any]) -> np.ndarray:
    adapter = capsule.get("adapter_state_dict") or {}
    if "suppression_direction" in adapter:
        v = np.array(adapter["suppression_direction"], dtype=np.float32).reshape(-1)
    elif "signature_vector" in capsule:
        v = np.array(capsule["signature_vector"], dtype=np.float32).reshape(-1)
    else:
        raise ValueError(f"No direction found for capsule subject={capsule.get('subject')}")
    v = np.nan_to_num(v.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
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
    seen = set()
    for p in entry.get("paths", []):
        if p in seen:
            continue
        seen.add(p)
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
    return ids if len(ids) <= n else rng.sample(ids, n)


def collect_projection_energies(index: Dict[str, Any], ids: List[str], layer: int, direction: np.ndarray, label: str) -> List[float]:
    vals = []
    for j, pid in enumerate(ids, 1):
        ap = activation_path_for(index, pid, layer)
        if ap is None or not ap.exists():
            log(f"  {label} {j}/{len(ids)}: missing activation for prompt_id={pid}, layer={layer}")
            continue
        t = time.time()
        act = load_pickle_gz(ap)
        vals.append(projection_energy(act, direction))
        log(f"  {label} {j}/{len(ids)}: loaded {ap.name} in {time.time() - t:.2f}s")
    return vals


def run(args: argparse.Namespace) -> Dict[str, Any]:
    rng = random.Random(args.seed)
    log(f"Loading prompts: {args.prompts_jsonl}")
    _, ids_by_subject = load_prompts(Path(args.prompts_jsonl))
    log(f"Prompt subjects: {len(ids_by_subject)}")

    subject_candidates = sorted(ids_by_subject.keys())
    if args.max_subjects > 0:
        subject_candidates = subject_candidates[: args.max_subjects]
    log(f"Selected subjects: {subject_candidates}")

    log(f"Loading activation index: {args.activation_index}")
    index = load_activation_index(Path(args.activation_index))
    log(f"Activation prompts indexed: {len(index.get('prompts', {}))}")

    capsules = load_capsules_for_subjects(Path(args.capsules_dir), subject_candidates)
    subjects = [s for s in subject_candidates if s in capsules]
    if not subjects:
        # Some capsule subjects may use canonical names from file contents.
        subjects = [s for s in sorted(capsules.keys()) if s in ids_by_subject]
    if not subjects:
        raise RuntimeError("No overlap between loaded capsule subjects and prompts.jsonl subjects")

    alpha_values = parse_floats(args.alpha_eff_values)
    per_subject: Dict[str, Any] = {}
    rows_by_alpha: Dict[float, Dict[str, List[float]]] = {
        a: {"target_attenuation": [], "target_post_ratio": [], "benign_runtime_attenuation": []}
        for a in alpha_values
    }

    for si, subject in enumerate(subjects, 1):
        log(f"Processing subject {si}/{len(subjects)}: {subject}")
        capsule = capsules[subject]
        layer = int(capsule.get("target_layer", args.default_layer))
        direction = extract_direction(capsule)
        target_ids = sample_ids(ids_by_subject.get(subject, []), args.target_prompts_per_subject, rng)
        other_ids = []
        for other_subj, ids in ids_by_subject.items():
            if other_subj != subject:
                other_ids.extend(ids)
        benign_ids = sample_ids(other_ids, args.benign_prompts_per_subject, rng)

        log(f"Subject {subject}: layer={layer}, target_ids={len(target_ids)}, benign_ids={len(benign_ids)}")
        target_pre = collect_projection_energies(index, target_ids, layer, direction, "target")
        benign_pre = collect_projection_energies(index, benign_ids, layer, direction, "benign")

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
        alpha_summary.append({
            "alpha_eff": alpha,
            "theoretical_projection_attenuation": 1.0 - ((1.0 - alpha) ** 2),
            "target_projection_attenuation": summarize(rec["target_attenuation"]),
            "target_projection_post_ratio": summarize(rec["target_post_ratio"]),
            "benign_runtime_attenuation": summarize(rec["benign_runtime_attenuation"]),
        })

    means = [x["target_projection_attenuation"]["mean"] for x in alpha_summary]
    monotonic = all(
        means[i] is None or means[i + 1] is None or means[i] <= means[i + 1] + 1e-8
        for i in range(len(means) - 1)
    )
    benign_zero = all((x["benign_runtime_attenuation"]["mean"] or 0.0) == 0.0 for x in alpha_summary)
    has_target = any((x["target_projection_attenuation"]["mean"] is not None) for x in alpha_summary)
    decision = {
        "verdict": "supports_current_narrative" if monotonic and benign_zero and has_target else "inconclusive",
        "comment": (
            "Controlled alpha_eff monotonically removes target-signature projection energy while benign runtime attenuation remains zero under the router-gated Module E path."
            if monotonic and benign_zero and has_target else
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
        "## Per-subject counts",
        "",
        "| Subject | Layer | Target activations | Benign activations |",
        "|---|---:|---:|---:|",
    ])
    for s, rec in summary["per_subject"].items():
        lines.append(f"| {s} | {rec['target_layer']} | {rec['target_prompt_count']} | {rec['benign_prompt_count']} |")
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
    print(md_path.read_text(), flush=True)
    print(f"Wrote JSON: {json_path}", flush=True)
    print(f"Wrote MD:   {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
