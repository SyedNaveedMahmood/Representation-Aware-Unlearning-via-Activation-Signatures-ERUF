#!/usr/bin/env python3
"""Batched evaluator for adversarial forget-recovery (pretrained + ERUF/KIF only).

Purpose
-------
The standard adversarial_forget_recovery_eval.py is conservative and evaluates
rows one-by-one. This script is optimized for speed. It loads each model once,
evaluates many rows with batched generation and batched EL-style mass
computation, writes the same checkpoint format as the standard script, and then
regenerates the shared summary.

Only the pretrained base model (pre) and the ERUF/KIF model (kif) are loaded;
no other-paper baseline models are discovered or evaluated.

It is checkpointed: if the job times out, rerun the same command.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from fast_entity_eval_bundle import (  # type: ignore
    PREFERRED_FORGET_SUBJECTS,
    append_jsonl,
    completed_ids,
    free_model,
    hit_rate,
    load_model,
    load_tokenizer,
    parse_subjects,
    refusal_like,
    text_hit,
    token_ids_for_texts,
)
from adversarial_forget_recovery_eval import build_adversarial_dataset, write_summary  # type: ignore


def log(msg: str) -> None:
    print(f"[ADV-BASE-FAST] {msg}", flush=True)


@torch.inference_mode()
def batch_generate(model, tok, prompts: Sequence[str], device: str, max_new_tokens: int) -> List[str]:
    enc = tok(list(prompts), return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    input_len = enc["input_ids"].shape[1]
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    gens = out[:, input_len:]
    return tok.batch_decode(gens, skip_special_tokens=True)


@torch.inference_mode()
def batch_autoregressive_mass(model, tok, prompts: Sequence[str], token_id_lists: Sequence[Sequence[int]], device: str, steps: int) -> List[float]:
    if not prompts:
        return []
    cur = tok(list(prompts), return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    vals = torch.zeros((len(prompts),), dtype=torch.float32, device=device)
    active = torch.tensor([1.0 if ids else 0.0 for ids in token_id_lists], dtype=torch.float32, device=device)

    # Per-row sparse token sets. This loop is cheap because max_keywords is small.
    cleaned: List[List[int]] = []
    for ids in token_id_lists:
        cleaned.append([int(x) for x in ids if int(x) >= 0])

    for _ in range(steps):
        out = model(**cur)
        probs = torch.softmax(out.logits[:, -1, :].float(), dim=-1)
        row_vals = []
        for i, ids in enumerate(cleaned):
            if not ids:
                row_vals.append(torch.tensor(0.0, dtype=torch.float32, device=device))
            else:
                row_vals.append(probs[i, ids].sum())
        vals += torch.stack(row_vals) * active
        nxt = torch.argmax(probs, dim=-1)
        input_ids = torch.cat([cur["input_ids"], nxt.unsqueeze(1)], dim=1)
        cur = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids, device=device)}
    denom = max(int(steps), 1)
    return (vals / denom).detach().cpu().float().tolist()


def eval_batch(model, tok, rows: Sequence[Dict[str, Any]], model_label: str, device: str, max_new_tokens: int, el_steps: int, max_keywords: int, el_batch_size: int) -> List[Dict[str, Any]]:
    prompts = [r["prompt"] for r in rows]
    try:
        gens = batch_generate(model, tok, prompts, device, max_new_tokens)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or len(rows) <= 1:
            raise
        torch.cuda.empty_cache()
        mid = len(rows) // 2
        return eval_batch(model, tok, rows[:mid], model_label, device, max_new_tokens, el_steps, max_keywords, el_batch_size) + eval_batch(model, tok, rows[mid:], model_label, device, max_new_tokens, el_steps, max_keywords, el_batch_size)

    target_token_ids = [token_ids_for_texts(tok, r.get("target_aliases", []) + r.get("target_keywords", []), max_keywords) for r in rows]
    retain_token_ids = [token_ids_for_texts(tok, r.get("retain_aliases", []) + r.get("retain_keywords", []), max_keywords) for r in rows]

    target_masses: List[float] = []
    retain_masses: List[float] = []
    for i in range(0, len(rows), el_batch_size):
        bprompts = prompts[i:i + el_batch_size]
        btoks = target_token_ids[i:i + el_batch_size]
        rtoks = retain_token_ids[i:i + el_batch_size]
        try:
            target_masses.extend(batch_autoregressive_mass(model, tok, bprompts, btoks, device, el_steps))
            retain_masses.extend(batch_autoregressive_mass(model, tok, bprompts, rtoks, device, el_steps))
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or len(bprompts) <= 1:
                raise
            torch.cuda.empty_cache()
            for j in range(i, min(i + el_batch_size, len(rows))):
                target_masses.extend(batch_autoregressive_mass(model, tok, [prompts[j]], [target_token_ids[j]], device, el_steps))
                retain_masses.extend(batch_autoregressive_mass(model, tok, [prompts[j]], [retain_token_ids[j]], device, el_steps))

    out_rows: List[Dict[str, Any]] = []
    for r, gen, tm, rm in zip(rows, gens, target_masses, retain_masses):
        out_rows.append({
            "model": model_label,
            "row_id": r["row_id"],
            "audit": r["audit"],
            "subject": r["subject"],
            "control_type": r["control_type"],
            "probe_family": r["probe_family"],
            "prompt": r["prompt"],
            "retain_entity": r.get("retain_entity", ""),
            "target_alias_hit": text_hit(gen, r.get("target_aliases", [])),
            "target_keyword_hit_rate": hit_rate(gen, r.get("target_keywords", [])),
            "target_mass": float(tm),
            "retain_alias_hit": text_hit(gen, r.get("retain_aliases", [])),
            "retain_keyword_hit_rate": hit_rate(gen, r.get("retain_keywords", [])),
            "retain_mass": float(rm),
            "refusal_like": refusal_like(gen),
            "generation_preview": gen[:500],
            "status": "ok",
        })
    return out_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--kif_adapter_path", required=True, help="Path to the ERUF/KIF saved model or adapter")
    ap.add_argument("--prompts_jsonl", default="outputs/datasets/prompts.jsonl")
    ap.add_argument("--out_dir", default="analysis/outputs_adversarial_forget_recovery")
    ap.add_argument("--models", default="pre,kif", help="comma list: pre,kif")
    ap.add_argument("--max_subjects", type=int, default=11)
    ap.add_argument("--load_mode", default="4bit", choices=["4bit", "8bit", "bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_eval_rows", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--el_batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--el_steps", type=int, default=8)
    ap.add_argument("--max_keywords", type=int, default=12)
    ap.add_argument("--recovery_keyword_threshold", type=float, default=0.15)
    ap.add_argument("--recovery_mass_threshold", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--rebuild_dataset", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    dataset_dir = out_dir / "datasets"
    ckpt_dir = out_dir / "checkpoints"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    subjects = parse_subjects(Path(args.prompts_jsonl), args.max_subjects) or PREFERRED_FORGET_SUBJECTS[:args.max_subjects]
    dataset_path = dataset_dir / "adversarial_forget_recovery.jsonl"
    dataset_rows = build_adversarial_dataset(subjects, dataset_path, rebuild=args.rebuild_dataset)
    log(f"Dataset rows: {len(dataset_rows)} subjects={subjects}")

    model_paths = {"pre": args.model_dir, "kif": args.kif_adapter_path}
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    model_labels = [m for m in requested if model_paths.get(m)]
    if "kif" in requested and not args.kif_adapter_path:
        raise ValueError("--kif_adapter_path is required when models include kif")

    tok = load_tokenizer(args.model_dir)
    for label in model_labels:
        eval_path = ckpt_dir / f"eval_rows_{label}.jsonl"
        done = completed_ids(eval_path)
        todo = [r for r in dataset_rows if r["row_id"] not in done]
        n = min(len(todo), args.max_eval_rows)
        if n <= 0:
            log(f"Eval complete for {label}: {len(done)}/{len(dataset_rows)}")
            continue
        log(f"Evaluating {label}: {n}/{len(todo)} remaining rows; path={model_paths[label]}")
        model = load_model(model_paths[label], args.model_dir, args.device, args.load_mode)
        try:
            for start in range(0, n, args.batch_size):
                batch = todo[start:start + args.batch_size]
                records = eval_batch(model, tok, batch, label, args.device, args.max_new_tokens, args.el_steps, args.max_keywords, args.el_batch_size)
                for rec in records:
                    append_jsonl(eval_path, rec)
                if (start // args.batch_size) % 5 == 0:
                    log(f"Progress {label} batch_end={start + len(batch)}/{n}; completed_total={len(done) + start + len(batch)}/{len(dataset_rows)}")
        finally:
            free_model(model)

    summary = write_summary(out_dir, dataset_rows, model_labels, args)
    log("Summary written")
    log(json.dumps({
        "completion": summary.get("completion"),
        "same_row_ids_across_models": summary.get("same_row_ids_across_models"),
        "overall_recovery": summary.get("paper_key_results", {}).get("overall_recovery"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
