#!/usr/bin/env python3
"""Run ERUF layer-band ablation end-to-end.

This script runs Module B activation collection and Module C signature mining
for multiple layer bands, then writes a single JSON and Markdown summary.

Default bands:
  early_05_09   -> layers 5-9
  middle_14_18 -> layers 14-18
  peak_23_27   -> layers 23-27

Example:
  python scripts/run_layer_band_ablation.py \
    --batch-size 8 \
    --out-root outputs/rebuttal_layer_ablation \
    --skip-existing
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics as stats
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch

from llama20.modules.module_c import SignatureMiningConfig, ROMEHyperParams, SignatureExtractor


DEFAULT_BANDS: Dict[str, List[int]] = {
    "early_05_09": [5, 6, 7, 8, 9],
    "middle_14_18": [14, 15, 16, 17, 18],
    "peak_23_27": [23, 24, 25, 26, 27],
}


def parse_layer_spec(spec: str) -> List[int]:
    layers = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"Invalid descending layer range: {part}")
            layers.update(range(lo, hi + 1))
        else:
            layers.add(int(part))
    if not layers:
        raise ValueError(f"Empty layer spec: {spec!r}")
    return sorted(layers)


def parse_bands(values: Iterable[str] | None) -> Dict[str, List[int]]:
    if not values:
        return dict(DEFAULT_BANDS)
    bands: Dict[str, List[int]] = {}
    for value in values:
        if ":" not in value:
            raise ValueError(f"Band must have format name:layers, got {value!r}")
        name, spec = value.split(":", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid empty band name in {value!r}")
        bands[name] = parse_layer_spec(spec)
    return bands


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def run_command(cmd: List[str], *, env: Dict[str, str], log_path: Path, cwd: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(f"# command_started: {now()}\n")
        log_f.write("# command: " + " ".join(cmd) + "\n\n")
        log_f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
        ret = proc.wait()
        elapsed = time.monotonic() - start
        log_f.write(f"\n# command_finished: {now()}\n")
        log_f.write(f"# return_code: {ret}\n")
        log_f.write(f"# elapsed_seconds: {elapsed:.3f}\n")
    if ret != 0:
        raise RuntimeError(f"Command failed with code {ret}: {' '.join(cmd)}")
    return elapsed


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_module_b_for_band(
    *,
    band_name: str,
    layers: List[int],
    band_dir: Path,
    batch_size: int,
    capture_scope: str,
    max_length: int,
    model_dir: str,
    prompts_file: str,
    cleanup_every_batches: int,
    force: bool,
    repo_root: Path,
) -> Dict[str, Any]:
    activations_dir = band_dir / "activations"
    report_path = activations_dir / "collection_report.json"
    index_path = activations_dir / "activation_index.json"
    log_path = band_dir / "module_b.log"

    if force and activations_dir.exists():
        shutil.rmtree(activations_dir)
    if report_path.exists() and index_path.exists() and not force:
        return {
            "status": "skipped_existing",
            "elapsed_seconds": None,
            "log_path": str(log_path),
            "activation_index": str(index_path),
            "collection_report": str(report_path),
        }

    env = os.environ.copy()
    env.update(
        {
            "KIF_MODULE_B_LAYERS": ",".join(map(str, layers)),
            "KIF_MODULE_B_BATCH_SIZE": str(batch_size),
            "KIF_MODULE_B_CAPTURE_SCOPE": capture_scope,
            "KIF_MODULE_B_OUTPUT_DIR": str(activations_dir),
            "KIF_MODULE_B_MAX_LENGTH": str(max_length),
            "KIF_MODULE_B_MODEL_DIR": model_dir,
            "KIF_MODULE_B_PROMPTS_FILE": prompts_file,
            "KIF_MODULE_B_CLEANUP_EVERY_BATCHES": str(cleanup_every_batches),
        }
    )
    elapsed = run_command(["llama20", "module_b"], env=env, log_path=log_path, cwd=repo_root)
    return {
        "status": "completed",
        "elapsed_seconds": elapsed,
        "log_path": str(log_path),
        "activation_index": str(index_path),
        "collection_report": str(report_path),
    }


def run_module_c_for_band(
    *,
    band_name: str,
    layers: List[int],
    band_dir: Path,
    significance_threshold: float,
    top_k_directions: int,
    force: bool,
) -> Dict[str, Any]:
    activations_dir = band_dir / "activations"
    signatures_dir = band_dir / "signatures"
    index_path = signatures_dir / "signature_index.json"
    summary_report = signatures_dir / "summary_report.json"
    log_path = band_dir / "module_c_runtime.json"

    if force and signatures_dir.exists():
        shutil.rmtree(signatures_dir)
    if index_path.exists() and summary_report.exists() and not force:
        return {
            "status": "skipped_existing",
            "elapsed_seconds": None,
            "signature_index": str(index_path),
            "summary_report": str(summary_report),
        }

    if not activations_dir.exists():
        raise FileNotFoundError(f"Missing activations for {band_name}: {activations_dir}")

    start = time.monotonic()
    config = SignatureMiningConfig(
        activations_dir=activations_dir,
        output_dir=signatures_dir,
        rome_hparams=ROMEHyperParams(
            layers=layers,
            layer_selection="top_k",
            target_module="mlp",
            significance_threshold=significance_threshold,
        ),
        top_k_directions=top_k_directions,
        min_prompts_per_subject=2,
        use_semantic_negatives=True,
        min_controls_per_subject=1,
        allow_synthetic_fallback=True,
        enable_oversampling=False,
        negative_pool_mode="match_positives",
        fixed_negative_pool_size=100,
        synthetic_fraction=0.10,
        activation_strategy="mean_token",
        standardize_dims=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_half_precision=False,
        enable_memory_cleanup=True,
        cleanup_frequency=5,
    )
    extractor = SignatureExtractor(config)
    signatures = extractor.extract_all_signatures()
    extractor.save_signature_index(signatures)
    extractor.create_summary_report()
    elapsed = time.monotonic() - start

    log_path.write_text(
        json.dumps(
            {
                "band": band_name,
                "layers": layers,
                "started_or_completed": now(),
                "elapsed_seconds": elapsed,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "status": "completed",
        "elapsed_seconds": elapsed,
        "signature_index": str(index_path),
        "summary_report": str(summary_report),
    }


def summarize_band(band_name: str, band_dir: Path, layers: List[int], module_b: Dict[str, Any], module_c: Dict[str, Any]) -> Dict[str, Any]:
    sig_index_path = band_dir / "signatures" / "signature_index.json"
    act_report_path = band_dir / "activations" / "collection_report.json"
    act_index_path = band_dir / "activations" / "activation_index.json"

    row: Dict[str, Any] = {
        "band": band_name,
        "layers": layers,
        "module_b_status": module_b.get("status"),
        "module_b_elapsed_seconds": module_b.get("elapsed_seconds"),
        "module_c_status": module_c.get("status"),
        "module_c_elapsed_seconds": module_c.get("elapsed_seconds"),
        "activation_index": str(act_index_path),
        "collection_report": str(act_report_path),
        "signature_index": str(sig_index_path),
    }

    if act_report_path.exists():
        act_report = load_json(act_report_path)
        row["activation_prompts_processed"] = act_report.get("results", {}).get("prompts_processed")
        row["activation_total_files"] = act_report.get("results", {}).get("total_files")
        row["activation_storage_gb"] = act_report.get("results", {}).get("storage_gb")

    if sig_index_path.exists():
        sig_index = load_json(sig_index_path)
        subjects = sig_index.get("subjects", {})
        scores: List[float] = []
        best_layers: List[int] = []
        successes = 0
        per_subject = {}
        for subj, rec in subjects.items():
            status = rec.get("status")
            best_score = rec.get("best_score")
            best_layer = rec.get("best_layer")
            if status == "success":
                successes += 1
            if best_score is not None:
                scores.append(float(best_score))
            if best_layer is not None:
                best_layers.append(int(best_layer))
            per_subject[subj] = {
                "status": status,
                "best_layer": best_layer,
                "best_score": best_score,
                "actual_layers_mined": rec.get("actual_layers_mined"),
            }
        row.update(
            {
                "subjects_total": len(subjects),
                "subjects_successful": successes,
                "mean_best_score": stats.mean(scores) if scores else None,
                "median_best_score": stats.median(scores) if scores else None,
                "min_best_score": min(scores) if scores else None,
                "max_best_score": max(scores) if scores else None,
                "best_layers": best_layers,
                "per_subject": per_subject,
                "signature_stats": sig_index.get("stats", {}),
            }
        )
    return row


def decide_narrative(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    usable = [r for r in rows if r.get("mean_best_score") is not None]
    if not usable:
        return {
            "verdict": "inconclusive",
            "comment": "No usable signature scores were produced; do not report this ablation yet.",
        }
    best = max(usable, key=lambda r: float(r["mean_best_score"]))
    peak = next((r for r in usable if "peak" in r["band"]), None)
    if peak is None:
        return {
            "verdict": "inconclusive",
            "comment": "Peak band was not included; use this only as a diagnostic, not as the layer-selection rebuttal.",
        }
    peak_score = float(peak["mean_best_score"])
    best_score = float(best["mean_best_score"])
    peak_success = peak.get("subjects_successful", 0)
    best_success = best.get("subjects_successful", 0)
    if best["band"] == peak["band"] or (peak_score >= 0.95 * best_score and peak_success >= best_success):
        return {
            "verdict": "supports_current_narrative",
            "comment": (
                "The peak 23-27 band is the strongest or essentially tied with the strongest band while preserving subject coverage. "
                "This supports the paper's Cohen's-d localization narrative and can be reported as a layer-band sensitivity check."
            ),
        }
    return {
        "verdict": "soften_claim",
        "comment": (
            f"The strongest band is {best['band']}, not peak_23_27. Do not claim fixed peak-layer superiority. "
            "Use this to soften the paper: ERUF should be described as using data-driven layer localization rather than a universal fixed layer band."
        ),
    }


def write_outputs(rows: List[Dict[str, Any]], out_root: Path) -> None:
    decision = decide_narrative(rows)
    summary = {
        "created_at": now(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "decision": decision,
        "bands": rows,
    }
    json_path = out_root / "layer_band_ablation_results.json"
    md_path = out_root / "layer_band_ablation_results.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Layer-Band Ablation Results",
        "",
        f"Created: {summary['created_at']}",
        f"GPU: {summary['gpu']}",
        "",
        f"Decision: **{decision['verdict']}**",
        "",
        decision["comment"],
        "",
        "| Band | Layers | Subjects | Mean best score | Median | Min | Max | Module B sec | Module C sec | Activation GB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        subj = ""
        if r.get("subjects_successful") is not None:
            subj = f"{r.get('subjects_successful')}/{r.get('subjects_total')}"
        def fmt(x: Any, nd: int = 4) -> str:
            return "" if x is None else f"{float(x):.{nd}f}"
        lines.append(
            "| {band} | {layers} | {subj} | {mean} | {median} | {minv} | {maxv} | {bsec} | {csec} | {gb} |".format(
                band=r["band"],
                layers=",".join(map(str, r["layers"])),
                subj=subj,
                mean=fmt(r.get("mean_best_score")),
                median=fmt(r.get("median_best_score")),
                minv=fmt(r.get("min_best_score")),
                maxv=fmt(r.get("max_best_score")),
                bsec=fmt(r.get("module_b_elapsed_seconds"), 1),
                csec=fmt(r.get("module_c_elapsed_seconds"), 1),
                gb=fmt(r.get("activation_storage_gb"), 3),
            )
        )
    lines.extend(
        [
            "",
            "## Per-subject details",
            "",
        ]
    )
    for r in rows:
        lines.append(f"### {r['band']}")
        lines.append("")
        lines.append("| Subject | Status | Best layer | Best score | Actual layers mined |")
        lines.append("|---|---|---:|---:|---|")
        for subj, rec in sorted(r.get("per_subject", {}).items()):
            best_score = rec.get("best_score")
            best_score_s = "" if best_score is None else f"{float(best_score):.4f}"
            lines.append(
                f"| {subj} | {rec.get('status')} | {rec.get('best_layer')} | {best_score_s} | {rec.get('actual_layers_mined')} |"
            )
        lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path.read_text())
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote MD:   {md_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Module B+C layer-band ablation and write one final JSON/MD summary.")
    ap.add_argument("--out-root", default="outputs/rebuttal_layer_ablation")
    ap.add_argument("--bands", nargs="*", help="Optional bands as name:layers, e.g. early:5-9 peak:23-27")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--capture-scope", default="full", choices=["full", "last_token"])
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--model-dir", default="outputs/model")
    ap.add_argument("--prompts-file", default="outputs/datasets/prompts.jsonl")
    ap.add_argument("--cleanup-every-batches", type=int, default=10)
    ap.add_argument("--significance-threshold", type=float, default=1.5)
    ap.add_argument("--top-k-directions", type=int, default=3)
    ap.add_argument("--skip-existing", action="store_true", help="Reuse completed activations/signatures if present.")
    ap.add_argument("--only-summary", action="store_true", help="Do not run modules; summarize existing outputs only.")
    ap.add_argument("--force", action="store_true", help="Delete and rerun existing outputs.")
    args = ap.parse_args()

    repo_root = Path.cwd().resolve()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    bands = parse_bands(args.bands)

    rows: List[Dict[str, Any]] = []
    for band_name, layers in bands.items():
        print(f"\n========== BAND {band_name}: layers {layers} ==========")
        band_dir = out_root / band_name
        band_dir.mkdir(parents=True, exist_ok=True)
        module_b_result: Dict[str, Any] = {"status": "not_run", "elapsed_seconds": None}
        module_c_result: Dict[str, Any] = {"status": "not_run", "elapsed_seconds": None}
        if not args.only_summary:
            module_b_result = run_module_b_for_band(
                band_name=band_name,
                layers=layers,
                band_dir=band_dir,
                batch_size=args.batch_size,
                capture_scope=args.capture_scope,
                max_length=args.max_length,
                model_dir=args.model_dir,
                prompts_file=args.prompts_file,
                cleanup_every_batches=args.cleanup_every_batches,
                force=args.force,
                repo_root=repo_root,
            )
            module_c_result = run_module_c_for_band(
                band_name=band_name,
                layers=layers,
                band_dir=band_dir,
                significance_threshold=args.significance_threshold,
                top_k_directions=args.top_k_directions,
                force=args.force,
            )
        else:
            if (band_dir / "activations" / "collection_report.json").exists():
                module_b_result["status"] = "existing"
            if (band_dir / "signatures" / "signature_index.json").exists():
                module_c_result["status"] = "existing"
        rows.append(summarize_band(band_name, band_dir, layers, module_b_result, module_c_result))

    write_outputs(rows, out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
