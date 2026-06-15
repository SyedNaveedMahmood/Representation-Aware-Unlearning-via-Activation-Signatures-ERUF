#!/usr/bin/env python3
"""Batched evaluator that runs all KIF entity audits on the pretrained base model
and the ERUF/KIF model only.

This is the optimized suite wrapper. For each requested model label it loads the
model once, evaluates rows in batches, computes generation-based metrics and
EL-style target mass in batches, checkpoints JSONL rows, and writes paper-facing
summaries.

Only two models are ever loaded:
  pre  -> the pretrained base model (--model_dir)
  kif  -> the ERUF/KIF saved model or adapter (--kif_adapter_path)

No other-paper baseline models (optout / simnpo / reglu / lunar) are discovered
or evaluated.

Suites evaluated for each model:
  1. fast_entity_eval_bundle.py
     - name-agnostic forget robustness
     - mixed-query / BLUR-style forget-retain robustness
     - syntactic locality
  2. adversarial_forget_recovery_eval.py
     - overall adversarial forget-recovery
     - by-attack-family recovery and alias-hit summaries
  3. rwku_style_entity_robustness.py
     - RWKU-style entity robustness generation/mass rows

By default the wrapper tries all remaining rows in one job. Checkpointing remains
active, so a re-submit resumes safely if a cluster time limit interrupts it.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

import fast_entity_eval_bundle as fast  # type: ignore
import adversarial_forget_recovery_eval as adv  # type: ignore
import rwku_style_entity_robustness as rwku  # type: ignore
from adversarial_forget_recovery_fast_baseline import eval_batch as eval_fastlike_batch  # type: ignore


def log(msg: str) -> None:
    print(f"[ERUF-SUITES] {msg}", flush=True)


def safe_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip().lower()).strip("_")


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def max_rows_to_run(todo: Sequence[Dict[str, Any]], max_rows: int) -> int:
    return len(todo) if max_rows == 0 else min(len(todo), max_rows)


def soft_stop(start: float, soft_minutes: float, reserve_seconds: float = 45.0) -> bool:
    if soft_minutes <= 0:
        return False
    return (time.time() - start) > max(1.0, soft_minutes * 60.0 - reserve_seconds)


@torch.inference_mode()
def rwku_batch_generate(model, tok, prompts: Sequence[str], device: str, max_new_tokens: int) -> List[str]:
    enc = tok(list(prompts), return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    input_len = enc["input_ids"].shape[1]
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    return tok.batch_decode(out[:, input_len:], skip_special_tokens=True)


@torch.inference_mode()
def rwku_batch_mass(model, tok, prompts: Sequence[str], token_id_lists: Sequence[Sequence[int]], device: str, steps: int) -> List[float]:
    if not prompts:
        return []
    cur = tok(list(prompts), return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    vals = torch.zeros((len(prompts),), dtype=torch.float32, device=device)
    cleaned = [[int(x) for x in ids if int(x) >= 0] for ids in token_id_lists]
    active = torch.tensor([1.0 if ids else 0.0 for ids in cleaned], dtype=torch.float32, device=device)
    for _ in range(max(1, int(steps))):
        out = model(**cur)
        probs = torch.softmax(out.logits[:, -1, :].float(), dim=-1)
        row_vals = []
        for i, ids in enumerate(cleaned):
            row_vals.append(probs[i, ids].sum() if ids else torch.tensor(0.0, dtype=torch.float32, device=device))
        vals += torch.stack(row_vals) * active
        nxt = torch.argmax(probs, dim=-1)
        input_ids = torch.cat([cur["input_ids"], nxt.unsqueeze(1)], dim=1)
        cur = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids, device=device)}
    return (vals / max(1, int(steps))).detach().cpu().float().tolist()


def eval_rwku_batch(model, tok, rows: Sequence[Dict[str, Any]], label: str, device: str, max_new_tokens: int, el_steps: int, max_keywords: int, el_batch_size: int) -> List[Dict[str, Any]]:
    prompts = [r["prompt"] for r in rows]
    try:
        gens = rwku_batch_generate(model, tok, prompts, device, max_new_tokens)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or len(rows) <= 1:
            raise
        torch.cuda.empty_cache()
        mid = len(rows) // 2
        return eval_rwku_batch(model, tok, rows[:mid], label, device, max_new_tokens, el_steps, max_keywords, el_batch_size) + eval_rwku_batch(model, tok, rows[mid:], label, device, max_new_tokens, el_steps, max_keywords, el_batch_size)

    token_lists = [rwku.keyword_token_ids(tok, r, max_keywords) for r in rows]
    masses: List[float] = []
    for i in range(0, len(rows), el_batch_size):
        bp = prompts[i:i + el_batch_size]
        bt = token_lists[i:i + el_batch_size]
        try:
            masses.extend(rwku_batch_mass(model, tok, bp, bt, device, el_steps))
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or len(bp) <= 1:
                raise
            torch.cuda.empty_cache()
            for j in range(i, min(i + el_batch_size, len(rows))):
                masses.extend(rwku_batch_mass(model, tok, [prompts[j]], [token_lists[j]], device, el_steps))

    out: List[Dict[str, Any]] = []
    for r, gen, mass in zip(rows, gens, masses):
        out.append({
            "model": label,
            "row_id": r["row_id"],
            "subject": r["subject"],
            "prompt_entity": r["prompt_entity"],
            "control_type": r["control_type"],
            "probe_family": r["probe_family"],
            "probe_type": r["probe_type"],
            "prompt": r["prompt"],
            "alias_hit": rwku.alias_hit(gen, r.get("target_aliases", [])),
            "keyword_hit_rate": rwku.keyword_hit_rate(gen, r.get("target_keywords", [])),
            "target_mass": float(mass),
            "generation_preview": gen[:400],
            "status": "ok",
        })
    return out


def append_many(path: Path, records: Sequence[Dict[str, Any]], append_fn) -> None:
    for rec in records:
        append_fn(path, rec)


def kif_path_for(label: str, model_path: str):
    return model_path if label == "kif" else None


def eval_fast_suite(model, tok, label: str, model_path: str, subjects: List[str], args: argparse.Namespace, start: float) -> Dict[str, Any]:
    out_dir = Path(args.out_root) / label / "fast_entity_eval_bundle"
    rows = fast.build_dataset(subjects, out_dir / "datasets" / "fast_entity_eval_bundle.jsonl", rebuild=args.rebuild_datasets)
    eval_path = out_dir / "checkpoints" / f"eval_rows_{label}.jsonl"
    done = fast.completed_ids(eval_path)
    todo = [r for r in rows if r["row_id"] not in done]
    n = max_rows_to_run(todo, args.max_fast_rows)
    log(f"[{label}] FAST batched rows: done={len(done)}/{len(rows)} this_run={n}")
    for start_i in range(0, n, args.batch_size):
        if soft_stop(start, args.soft_time_limit_minutes):
            log(f"[{label}] FAST soft stop at row {start_i}/{n}")
            break
        batch = todo[start_i:start_i + args.batch_size]
        records = eval_fastlike_batch(model, tok, batch, label, args.device, args.fast_max_new_tokens, args.fast_el_steps, args.fast_max_keywords, args.el_batch_size)
        append_many(eval_path, records, fast.append_jsonl)
        if (start_i // args.batch_size) % max(1, args.progress_every_batches) == 0:
            log(f"[{label}] FAST batch_end={start_i + len(batch)}/{n}; completed_total={len(done) + start_i + len(batch)}/{len(rows)}")
    ns = Namespace(model_dir=args.model_dir, kif_adapter_path=kif_path_for(label, model_path), prompts_jsonl=args.prompts_jsonl, out_dir=str(out_dir), models=label, load_mode=args.load_mode, device=args.device, seed=args.seed)
    return fast.write_summary(out_dir, rows, [label], ns)


def eval_adv_suite(model, tok, label: str, model_path: str, subjects: List[str], args: argparse.Namespace, start: float) -> Dict[str, Any]:
    out_dir = Path(args.out_root) / label / "adversarial_forget_recovery"
    rows = adv.build_adversarial_dataset(subjects, out_dir / "datasets" / "adversarial_forget_recovery.jsonl", rebuild=args.rebuild_datasets)
    eval_path = out_dir / "checkpoints" / f"eval_rows_{label}.jsonl"
    done = adv.completed_ids(eval_path)
    todo = [r for r in rows if r["row_id"] not in done]
    n = max_rows_to_run(todo, args.max_adv_rows)
    log(f"[{label}] ADV batched rows: done={len(done)}/{len(rows)} this_run={n}")
    for start_i in range(0, n, args.batch_size):
        if soft_stop(start, args.soft_time_limit_minutes):
            log(f"[{label}] ADV soft stop at row {start_i}/{n}")
            break
        batch = todo[start_i:start_i + args.batch_size]
        records = eval_fastlike_batch(model, tok, batch, label, args.device, args.adv_max_new_tokens, args.adv_el_steps, args.adv_max_keywords, args.el_batch_size)
        append_many(eval_path, records, adv.append_jsonl)
        if (start_i // args.batch_size) % max(1, args.progress_every_batches) == 0:
            log(f"[{label}] ADV batch_end={start_i + len(batch)}/{n}; completed_total={len(done) + start_i + len(batch)}/{len(rows)}")
    ns = Namespace(model_dir=args.model_dir, kif_adapter_path=kif_path_for(label, model_path), prompts_jsonl=args.prompts_jsonl, out_dir=str(out_dir), models=label, load_mode=args.load_mode, device=args.device, recovery_keyword_threshold=args.recovery_keyword_threshold, recovery_mass_threshold=args.recovery_mass_threshold, seed=args.seed)
    return adv.write_summary(out_dir, rows, [label], ns)


def eval_rwku_suite(model, tok, label: str, model_path: str, subjects: List[str], args: argparse.Namespace, start: float) -> Dict[str, Any]:
    out_dir = Path(args.out_root) / label / "rwku_style_entity_robustness"
    cap = args.rwku_rows_per_family_cap if args.rwku_rows_per_family_cap > 0 else None
    rows = rwku.load_or_build_dataset(subjects, out_dir / "dataset" / "rwku_style_entity_robustness.jsonl", args.rebuild_datasets, cap)
    eval_path = out_dir / "checkpoints" / f"eval_rows_{label}.jsonl"
    done = rwku.completed_ids(eval_path)
    todo = [r for r in rows if r["row_id"] not in done]
    n = max_rows_to_run(todo, args.max_rwku_rows)
    log(f"[{label}] RWKU batched rows: done={len(done)}/{len(rows)} this_run={n}")
    for start_i in range(0, n, args.batch_size):
        if soft_stop(start, args.soft_time_limit_minutes):
            log(f"[{label}] RWKU soft stop at row {start_i}/{n}")
            break
        batch = todo[start_i:start_i + args.batch_size]
        records = eval_rwku_batch(model, tok, batch, label, args.device, args.rwku_max_new_tokens, args.rwku_el_steps, args.rwku_max_keywords, args.el_batch_size)
        append_many(eval_path, records, rwku.append_jsonl)
        if (start_i // args.batch_size) % max(1, args.progress_every_batches) == 0:
            log(f"[{label}] RWKU batch_end={start_i + len(batch)}/{n}; completed_total={len(done) + start_i + len(batch)}/{len(rows)}")
    ns = Namespace(model_dir=args.model_dir, kif_adapter_path=kif_path_for(label, model_path), prompts_jsonl=args.prompts_jsonl, out_dir=str(out_dir), models=label, load_mode=args.load_mode, device=args.device, seed=args.seed)
    return rwku.write_summary(out_dir, rows, [label], ns)


def compact_summary(label: str, model_path: str, summaries: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "label": label,
        "model_path": model_path,
        "out_root": args.out_root,
        "fast_completion": summaries.get("fast", {}).get("completion"),
        "fast_paper_key_results": summaries.get("fast", {}).get("paper_key_results"),
        "adversarial_completion": summaries.get("adversarial", {}).get("completion"),
        "adversarial_paper_key_results": summaries.get("adversarial", {}).get("paper_key_results"),
        "rwku_completion": summaries.get("rwku", {}).get("completion"),
        "rwku_eval_summary": summaries.get("rwku", {}).get("evaluation_summary"),
        "args": vars(args),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B", help="Pretrained base model (pre)")
    ap.add_argument("--kif_adapter_path", required=True, help="ERUF/KIF saved model or adapter path (kif)")
    ap.add_argument("--models", default="pre,kif", help="comma list: pre,kif")
    ap.add_argument("--prompts_jsonl", default="outputs/datasets/prompts.jsonl")
    ap.add_argument("--out_root", default="analysis/outputs_eruf_suite_evals")
    ap.add_argument("--max_subjects", type=int, default=11)
    ap.add_argument("--load_mode", default="4bit", choices=["4bit", "8bit", "bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--rebuild_datasets", action="store_true")
    ap.add_argument("--soft_time_limit_minutes", type=float, default=0.0, help="0 disables soft stop; Slurm still enforces wall time")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--el_batch_size", type=int, default=16)
    ap.add_argument("--progress_every_batches", type=int, default=5)

    ap.add_argument("--max_fast_rows", type=int, default=0, help="0 = all remaining")
    ap.add_argument("--fast_max_new_tokens", type=int, default=48)
    ap.add_argument("--fast_el_steps", type=int, default=8)
    ap.add_argument("--fast_max_keywords", type=int, default=10)

    ap.add_argument("--max_adv_rows", type=int, default=0, help="0 = all remaining")
    ap.add_argument("--adv_max_new_tokens", type=int, default=64)
    ap.add_argument("--adv_el_steps", type=int, default=8)
    ap.add_argument("--adv_max_keywords", type=int, default=12)
    ap.add_argument("--recovery_keyword_threshold", type=float, default=0.15)
    ap.add_argument("--recovery_mass_threshold", type=float, default=0.02)

    ap.add_argument("--max_rwku_rows", type=int, default=0, help="0 = all remaining")
    ap.add_argument("--rwku_rows_per_family_cap", type=int, default=0)
    ap.add_argument("--rwku_max_new_tokens", type=int, default=48)
    ap.add_argument("--rwku_el_steps", type=int, default=8)
    ap.add_argument("--rwku_max_keywords", type=int, default=10)
    ap.add_argument("--rwku_mia_max_pairs", type=int, default=300)
    ap.add_argument("--rwku_max_length", type=int, default=192)
    ap.add_argument("--rwku_min_k_frac", type=float, default=0.2)
    args = ap.parse_args()

    start = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tok = fast.load_tokenizer(args.model_dir)
    subjects = fast.parse_subjects(Path(args.prompts_jsonl), args.max_subjects) or fast.PREFERRED_FORGET_SUBJECTS[:args.max_subjects]
    log(f"subjects={subjects}")

    model_paths = {"pre": args.model_dir, "kif": args.kif_adapter_path}
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    labels = [m for m in requested if model_paths.get(m)]
    if not labels:
        raise ValueError("No valid models requested. Use --models pre,kif")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    all_summaries: Dict[str, Any] = {}
    for label in labels:
        model_path = model_paths[label]
        log(f"=== Evaluating label={label} model_path={model_path} ===")
        write_json(out_root / label / "resolved_model.json", {"label": label, "model_path": model_path})
        model = fast.load_model(model_path, args.model_dir, args.device, args.load_mode)
        summaries: Dict[str, Any] = {}
        try:
            summaries["fast"] = eval_fast_suite(model, tok, label, model_path, subjects, args, start)
            summaries["adversarial"] = eval_adv_suite(model, tok, label, model_path, subjects, args, start)
            summaries["rwku"] = eval_rwku_suite(model, tok, label, model_path, subjects, args, start)
        finally:
            fast.free_model(model)
        combined = compact_summary(label, model_path, summaries, args)
        write_json(out_root / label / "eruf_suite_summary.json", combined)
        all_summaries[label] = combined
        log(f"[{label}] summary written to {out_root / label / 'eruf_suite_summary.json'}")

    write_json(out_root / "eruf_suite_summary.json", {"models": labels, "by_model": all_summaries})
    log("Combined summary written")
    log(json.dumps({
        label: {
            "model_path": all_summaries[label].get("model_path"),
            "fast_completion": all_summaries[label].get("fast_completion"),
            "adversarial_completion": all_summaries[label].get("adversarial_completion"),
            "rwku_completion": all_summaries[label].get("rwku_completion"),
        } for label in labels
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
