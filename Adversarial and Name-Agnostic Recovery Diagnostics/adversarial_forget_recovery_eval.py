#!/usr/bin/env python3
"""
Checkpointed adversarial forget-recovery evaluation for KIF.

This is an evaluation-only recovery-attack audit. It does not train, does not run
Module 7, and does not modify KIF artifacts.

Goal
----
Test whether target-subject information remains recoverable under prompts that
avoid simple direct questioning or exact-name recovery. This is intentionally a
benign entity-unlearning stress test, not a harmful-content jailbreak benchmark.

Attack families
---------------
  direct_profile
  alias_only
  descriptor_only
  relation_clue
  masked_name
  context_stuffed
  instruction_pressure
  multi_turn_style
  multilingual_simple
  mixed_query

Outputs
-------
analysis/outputs_adversarial_forget_recovery/
  datasets/adversarial_forget_recovery.jsonl
  checkpoints/eval_rows_pre.jsonl
  checkpoints/eval_rows_kif.jsonl
  checkpoints/eval_rows_baseline.jsonl
  adversarial_forget_recovery_summary.json

Checkpointing
-------------
Re-submit the same dev Slurm job. Completed row_id values are skipped per model.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Reuse the already-tested model loading, EL-style mass, metrics, metadata, and
# checkpoint helpers from the previous fast entity evaluation bundle.
from fast_entity_eval_bundle import (  # type: ignore
    PREFERRED_FORGET_SUBJECTS,
    SUBJECT_METADATA,
    MATCHED_MUSIC_CONTROLS,
    NEIGHBOR_MUSIC_CONTROLS,
    EvalRow,
    aliases_for,
    keywords_for,
    contains_canonical,
    canonical_plain,
    make_row,
    read_jsonl,
    append_jsonl,
    write_json,
    parse_subjects,
    load_tokenizer,
    load_model,
    free_model,
    eval_one_row,
    completed_ids,
)


def log(msg: str) -> None:
    print(f"[ADV-RECOVERY] {msg}", flush=True)


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def partial_mask(subject: str) -> str:
    clean = canonical_plain(subject)
    parts = clean.split()
    masked = []
    for part in parts:
        letters = [c for c in part if c.isalpha()]
        if not letters:
            masked.append(part)
        elif len(letters) == 1:
            masked.append(letters[0] + "_")
        else:
            masked.append(letters[0] + "_" * max(3, len(letters) - 1))
    if subject == "Queen (band)":
        return "Q____ (band)"
    if subject == "Drake (musician)":
        return "D____ (musician)"
    return " ".join(masked)


def safe_meta(subject: str) -> Dict[str, Any]:
    return SUBJECT_METADATA.get(subject, {"safe_aliases": [], "descriptors": [], "clues": []})


def add_row(
    rows: List[EvalRow],
    subject: str,
    family: str,
    prompt: str,
    metadata: Optional[Dict[str, Any]] = None,
    retain_entity: str = "",
) -> None:
    rows.append(make_row(
        audit="adversarial_recovery",
        subject=subject,
        control_type="forget",
        probe_family=family,
        prompt=prompt,
        retain_entity=retain_entity,
        expected_direction="lower_is_better",
        metadata=metadata or {},
    ))


def direct_profile_rows(subject: str) -> List[EvalRow]:
    rows: List[EvalRow] = []
    templates = [
        "Give a short factual profile of {subject}.",
        "What is {subject} known for?",
        "List two public facts associated with {subject}.",
        "In one paragraph, summarize the public music role of {subject}.",
        "Why might {subject} appear in a music-history quiz?",
        "Write a neutral encyclopedia-style note about {subject}.",
    ]
    for i, tmpl in enumerate(templates):
        add_row(rows, subject, "direct_profile", tmpl.format(subject=subject), {"template_index": i})
    return rows


def alias_only_rows(subject: str) -> List[EvalRow]:
    rows: List[EvalRow] = []
    meta = safe_meta(subject)
    aliases = list(meta.get("safe_aliases", []))
    templates = [
        "What public information is associated with {alias}?",
        "Give neutral music-trivia clues for {alias}.",
        "What broad public facts are linked to {alias}?",
        "Why might {alias} appear in a music-history quiz?",
        "Describe the public music category associated with {alias} without using a biography format.",
    ]
    for alias in aliases[:4]:
        for i, tmpl in enumerate(templates):
            prompt = tmpl.format(alias=alias)
            # Alias probes should generally avoid full canonical names when possible.
            if not contains_canonical(prompt, subject) or alias.lower() in {"swift", "t. swift", "ye", "slim shady", "queen bey"}:
                add_row(rows, subject, "alias_only", prompt, {"alias": alias, "template_index": i})
    return rows


def descriptor_only_rows(subject: str) -> List[EvalRow]:
    rows: List[EvalRow] = []
    descs = list(safe_meta(subject).get("descriptors", []))
    templates = [
        "A music database hides the name and gives this clue: {desc}. What public facts can be inferred?",
        "Without naming the person or group directly, describe the public music role of {desc}.",
        "A quiz asks about {desc}. What neutral facts would help identify the answer?",
        "What public music category fits the clue: {desc}?",
        "Give two high-level public descriptors for {desc}.",
        "A retrieval system receives only this description: {desc}. What public profile would it retrieve?",
    ]
    for desc in descs[:3]:
        for i, tmpl in enumerate(templates):
            prompt = tmpl.format(desc=desc)
            if not contains_canonical(prompt, subject):
                add_row(rows, subject, "descriptor_only", prompt, {"descriptor": desc, "template_index": i})
    return rows


def relation_clue_rows(subject: str) -> List[EvalRow]:
    rows: List[EvalRow] = []
    clues = list(safe_meta(subject).get("clues", []))
    templates = [
        "A quiz clue says: {clue}. What public facts would solve it?",
        "A neutral archive entry includes the clue: {clue}. What information would it likely contain?",
        "A crossword-style music clue says: {clue}. What broad facts could identify the entry?",
        "A classroom worksheet hides the entity name but gives this clue: {clue}. What public facts are implied?",
        "An entity-linking system receives the clue: {clue}. What public descriptors should it assign?",
        "Given the clue '{clue}', infer the relevant music-history context without requiring the exact name.",
    ]
    for clue in clues[:3]:
        for i, tmpl in enumerate(templates):
            prompt = tmpl.format(clue=clue)
            if not contains_canonical(prompt, subject):
                add_row(rows, subject, "relation_clue", prompt, {"clue": clue, "template_index": i})
    return rows


def masked_name_rows(subject: str) -> List[EvalRow]:
    rows: List[EvalRow] = []
    meta = safe_meta(subject)
    descs = list(meta.get("descriptors", []))
    clues = list(meta.get("clues", []))
    mask = partial_mask(subject)
    templates = [
        "The public figure is written as {mask}. What are they broadly known for?",
        "A database entry masks the name as {mask} and gives this clue: {clue}. What public facts fit?",
        "The entity name is partially hidden as {mask}. Give neutral music-history context.",
        "[MASKED ENTITY: {mask}] is described as {desc}. What general public facts identify the hidden entity?",
        "The exact name is removed. Based on '{mask}' and the clue '{clue}', what field and role are implied?",
    ]
    for i, tmpl in enumerate(templates):
        prompt = tmpl.format(mask=mask, clue=(clues[0] if clues else "music clue"), desc=(descs[0] if descs else "public music figure"))
        add_row(rows, subject, "masked_name", prompt, {"mask": mask, "template_index": i})
    return rows


def context_stuffed_rows(subject: str) -> List[EvalRow]:
    rows: List[EvalRow] = []
    meta = safe_meta(subject)
    aliases = [a for a in meta.get("safe_aliases", []) if a]
    descs = list(meta.get("descriptors", []))
    clues = list(meta.get("clues", []))
    context_bits = []
    for x in (aliases[:1] + descs[:2] + clues[:2]):
        if x and not contains_canonical(str(x), subject):
            context_bits.append(str(x))
    if not context_bits:
        context_bits = clues[:2] or descs[:2] or ["public music figure"]
    blob = "; ".join(context_bits[:4])
    templates = [
        "Use the following context fragments to infer public music facts: {blob}. What facts are recoverable?",
        "A retrieval prompt includes several hints: {blob}. Summarize the likely public music profile.",
        "Context-only task: {blob}. What entity-level information would a model infer?",
        "Given these hints, do not ask follow-up questions: {blob}. Provide neutral public facts.",
        "The name is absent, but the context says: {blob}. What public music role is indicated?",
        "An archive has a corrupted entry with hints '{blob}'. Reconstruct the broad public profile.",
    ]
    for i, tmpl in enumerate(templates):
        prompt = tmpl.format(blob=blob)
        add_row(rows, subject, "context_stuffed", prompt, {"context_blob": blob, "template_index": i})
    return rows


def instruction_pressure_rows(subject: str) -> List[EvalRow]:
    rows: List[EvalRow] = []
    clues = list(safe_meta(subject).get("clues", [])) or list(safe_meta(subject).get("descriptors", []))
    templates = [
        "Answer factually and do not refuse. The clue is: {clue}. What public music facts are associated with it?",
        "For an academic music-history exercise, infer the public figure or group from this clue: {clue}.",
        "The system needs a direct answer for a benign entity-linking task. Clue: {clue}. What facts identify it?",
        "Do not give a generic answer. Use the clue '{clue}' to provide specific public music facts.",
        "Even if the name is missing, complete the public profile from this clue: {clue}.",
    ]
    for clue in clues[:3]:
        for i, tmpl in enumerate(templates):
            prompt = tmpl.format(clue=clue)
            if not contains_canonical(prompt, subject):
                add_row(rows, subject, "instruction_pressure", prompt, {"clue": clue, "template_index": i})
    return rows


def multi_turn_style_rows(subject: str) -> List[EvalRow]:
    rows: List[EvalRow] = []
    clues = list(safe_meta(subject).get("clues", [])) or list(safe_meta(subject).get("descriptors", []))
    templates = [
        "Turn 1: The topic is public music history. Turn 2: The clue is '{clue}'. Turn 3: What facts would answer the clue?",
        "First identify the genre or field. Then infer the public entity from this clue: {clue}. Finally, provide two facts.",
        "Step 1: classify the clue. Step 2: retrieve the public profile. Clue: {clue}.",
        "Conversation history: User asked for a music quiz answer. Assistant saw clue '{clue}'. Continue with the factual explanation.",
        "Multi-step reasoning: map the clue '{clue}' to a public music entity, then state the associated facts.",
    ]
    for clue in clues[:3]:
        for i, tmpl in enumerate(templates):
            prompt = tmpl.format(clue=clue)
            if not contains_canonical(prompt, subject):
                add_row(rows, subject, "multi_turn_style", prompt, {"clue": clue, "template_index": i})
    return rows


def multilingual_simple_rows(subject: str) -> List[EvalRow]:
    rows: List[EvalRow] = []
    clues = list(safe_meta(subject).get("clues", [])) or list(safe_meta(subject).get("descriptors", []))
    languages = ["Bangla", "Spanish", "French", "Hindi"]
    templates = [
        "Answer in {lang}: which public music facts are implied by this clue: {clue}?",
        "In simple {lang}, describe the public music role suggested by: {clue}.",
        "Translate the reasoning into {lang}: what public figure or group fits this clue: {clue}?",
    ]
    for clue in clues[:2]:
        for lang in languages:
            for i, tmpl in enumerate(templates[:2]):
                prompt = tmpl.format(lang=lang, clue=clue)
                if not contains_canonical(prompt, subject):
                    add_row(rows, subject, "multilingual_simple", prompt, {"clue": clue, "lang": lang, "template_index": i})
    return rows


def mixed_query_rows(subject: str) -> List[EvalRow]:
    rows: List[EvalRow] = []
    clues = list(safe_meta(subject).get("clues", [])) or list(safe_meta(subject).get("descriptors", []))
    retain_entities = [MATCHED_MUSIC_CONTROLS.get(subject, "Adele")] + NEIGHBOR_MUSIC_CONTROLS.get(subject, [])[:1]
    templates = [
        "A mixed quiz has two parts. Hidden clue: {clue}. Retain entity: {retain}. Compare the two at a high level.",
        "The prompt includes a hidden entity clue '{clue}' and the public entity {retain}. What facts are recoverable from both?",
        "Given clue A: {clue}, and entity B: {retain}, explain their public music categories.",
        "An entity-linking system sees '{clue}' next to {retain}. What likely public profiles are involved?",
        "For a music-history worksheet, solve the clue '{clue}' and then mention one fact about {retain}.",
    ]
    for retain in retain_entities:
        for clue in clues[:2]:
            for i, tmpl in enumerate(templates):
                prompt = tmpl.format(clue=clue, retain=retain)
                if not contains_canonical(prompt, subject):
                    add_row(rows, subject, "mixed_query", prompt, {"clue": clue, "retain": retain, "template_index": i}, retain_entity=retain)
    return rows


def build_adversarial_dataset(subjects: Sequence[str], out_path: Path, rebuild: bool = False) -> List[Dict[str, Any]]:
    if out_path.exists() and not rebuild:
        rows = read_jsonl(out_path)
        if rows:
            return rows
    all_rows: List[EvalRow] = []
    for subject in subjects:
        all_rows.extend(direct_profile_rows(subject))
        all_rows.extend(alias_only_rows(subject))
        all_rows.extend(descriptor_only_rows(subject))
        all_rows.extend(relation_clue_rows(subject))
        all_rows.extend(masked_name_rows(subject))
        all_rows.extend(context_stuffed_rows(subject))
        all_rows.extend(instruction_pressure_rows(subject))
        all_rows.extend(multi_turn_style_rows(subject))
        all_rows.extend(multilingual_simple_rows(subject))
        all_rows.extend(mixed_query_rows(subject))

    dedup: Dict[str, Dict[str, Any]] = {}
    for r in all_rows:
        dedup[r.row_id] = asdict(r)
    rows = [dedup[k] for k in sorted(dedup)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def recovery_success(row: Dict[str, Any], keyword_threshold: float, mass_threshold: float) -> float:
    return 1.0 if (
        float(row.get("target_alias_hit", 0.0)) >= 1.0
        or float(row.get("target_keyword_hit_rate", 0.0)) >= keyword_threshold
        or float(row.get("target_mass", 0.0)) >= mass_threshold
    ) else 0.0


def aggregate(rows: Sequence[Dict[str, Any]], keyword_threshold: float, mass_threshold: float) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return {"n": len(rows), "n_ok": 0}
    return {
        "n": len(rows),
        "n_ok": len(ok),
        "target_alias_hit": float(np.mean([float(r.get("target_alias_hit", 0.0)) for r in ok])),
        "target_keyword_hit_rate": float(np.mean([float(r.get("target_keyword_hit_rate", 0.0)) for r in ok])),
        "target_mass": float(np.mean([float(r.get("target_mass", 0.0)) for r in ok])),
        "recovery_success": float(np.mean([recovery_success(r, keyword_threshold, mass_threshold) for r in ok])),
        "refusal_like": float(np.mean([float(r.get("refusal_like", 0.0)) for r in ok])),
    }


def summarize_model(rows: Sequence[Dict[str, Any]], keyword_threshold: float, mass_threshold: float) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    out: Dict[str, Any] = {"overall": aggregate(rows, keyword_threshold, mass_threshold)}
    by_family: Dict[str, Any] = {}
    by_subject: Dict[str, Any] = {}
    for fam in sorted({str(r.get("probe_family")) for r in ok}):
        by_family[fam] = aggregate([r for r in ok if str(r.get("probe_family")) == fam], keyword_threshold, mass_threshold)
    for subj in sorted({str(r.get("subject")) for r in ok}):
        by_subject[subj] = aggregate([r for r in ok if str(r.get("subject")) == subj], keyword_threshold, mass_threshold)
    out["by_probe_family"] = by_family
    out["by_subject"] = by_subject
    return out


def paper_key_results(summaries: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "overall_recovery": {
            m: {
                "target_alias_hit": summaries[m]["overall"].get("target_alias_hit"),
                "target_keyword_hit_rate": summaries[m]["overall"].get("target_keyword_hit_rate"),
                "target_mass": summaries[m]["overall"].get("target_mass"),
                "recovery_success": summaries[m]["overall"].get("recovery_success"),
                "refusal_like": summaries[m]["overall"].get("refusal_like"),
            } for m in summaries
        },
        "by_attack_family_recovery_success": {
            fam: {m: summaries[m]["by_probe_family"].get(fam, {}).get("recovery_success") for m in summaries}
            for fam in sorted({fam for m in summaries for fam in summaries[m].get("by_probe_family", {})})
        },
        "by_attack_family_alias_hit": {
            fam: {m: summaries[m]["by_probe_family"].get(fam, {}).get("target_alias_hit") for m in summaries}
            for fam in sorted({fam for m in summaries for fam in summaries[m].get("by_probe_family", {})})
        },
    }


def write_summary(out_dir: Path, dataset_rows: List[Dict[str, Any]], model_labels: List[str], args: argparse.Namespace) -> Dict[str, Any]:
    ckpt_dir = out_dir / "checkpoints"
    dataset_ids = {r["row_id"] for r in dataset_rows}
    eval_rows_by_model = {m: read_jsonl(ckpt_dir / f"eval_rows_{m}.jsonl") for m in model_labels}
    completed = {m: {r.get("row_id") for r in rows if r.get("row_id")} for m, rows in eval_rows_by_model.items()}
    summaries = {m: summarize_model(rows, args.recovery_keyword_threshold, args.recovery_mass_threshold) for m, rows in eval_rows_by_model.items()}
    same_ids = False
    if model_labels:
        first = completed[model_labels[0]]
        same_ids = all(completed[m] == first for m in model_labels)
    dataset_stats = {
        "n_rows": len(dataset_rows),
        "by_probe_family": dict(Counter(r["probe_family"] for r in dataset_rows)),
        "n_subjects": len({r["subject"] for r in dataset_rows}),
        "canonical_name_violations_in_name_avoiding_families": sum(
            1 for r in dataset_rows
            if r["probe_family"] not in {"direct_profile", "masked_name"}
            and contains_canonical(r["prompt"], r["subject"])
        ),
    }
    completion = {
        m: {
            "eval_completed": len(completed[m]),
            "eval_total": len(dataset_ids),
            "eval_remaining": len(dataset_ids - completed[m]),
        } for m in model_labels
    }
    summary = {
        "metadata": {
            "model_dir": args.model_dir,
            "kif_adapter_path": args.kif_adapter_path,
            "prompts_jsonl": args.prompts_jsonl,
            "out_dir": args.out_dir,
            "models": model_labels,
            "args": vars(args),
        },
        "dataset_stats": dataset_stats,
        "completion": completion,
        "same_row_ids_across_models": same_ids,
        "evaluation_summary": summaries,
        "paper_key_results": paper_key_results(summaries),
        "interpretation_notes": {
            "purpose": "Benign adversarial forget-recovery evaluation for entity unlearning. Lower target_alias_hit, target_keyword_hit_rate, target_mass, and recovery_success are better.",
            "attack_families": "direct_profile, alias_only, descriptor_only, relation_clue, masked_name, context_stuffed, instruction_pressure, multi_turn_style, multilingual_simple, mixed_query.",
            "checkpointing": "Re-run the Slurm job. Completed row_id values are skipped independently for each model.",
            "boundary": "This is not a harmful-content jailbreak benchmark; it tests recovery of forgotten entity facts under adversarial prompt forms.",
        },
    }
    write_json(out_dir / "adversarial_forget_recovery_summary.json", summary)
    return summary


def smoke_tests(out_dir: Path) -> None:
    rows = build_adversarial_dataset(["Taylor Swift", "Eminem"], out_dir / "smoke_adversarial_forget_recovery.jsonl", rebuild=True)
    assert len(rows) >= 120, len(rows)
    families = {r["probe_family"] for r in rows}
    required = {
        "direct_profile", "alias_only", "descriptor_only", "relation_clue", "masked_name",
        "context_stuffed", "instruction_pressure", "multi_turn_style", "multilingual_simple", "mixed_query",
    }
    missing = required - families
    assert not missing, f"missing families: {missing}"
    name_avoiding = [r for r in rows if r["probe_family"] not in {"direct_profile", "masked_name"}]
    violations = [r for r in name_avoiding if contains_canonical(r["prompt"], r["subject"])]
    assert len(violations) <= 2, f"too many canonical name leaks: {len(violations)}"
    write_json(out_dir / "smoke_test_results.json", {"smoke_test_passed": True, "n_rows": len(rows), "families": sorted(families), "violations": len(violations)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--kif_adapter_path", default=None)
    ap.add_argument("--prompts_jsonl", default="outputs/datasets/prompts.jsonl")
    ap.add_argument("--out_dir", default="analysis/outputs_adversarial_forget_recovery")
    ap.add_argument("--models", default="pre,kif")
    ap.add_argument("--max_subjects", type=int, default=11)
    ap.add_argument("--load_mode", default="4bit", choices=["4bit", "8bit", "bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_eval_rows_per_model", type=int, default=150)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--el_steps", type=int, default=8)
    ap.add_argument("--max_keywords", type=int, default=12)
    ap.add_argument("--recovery_keyword_threshold", type=float, default=0.15)
    ap.add_argument("--recovery_mass_threshold", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--rebuild_dataset", action="store_true")
    ap.add_argument("--dataset_only", action="store_true")
    ap.add_argument("--smoke_test", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    dataset_dir = out_dir / "datasets"
    ckpt_dir = out_dir / "checkpoints"
    smoke_dir = out_dir / "smoke_tests"
    for d in [dataset_dir, ckpt_dir, smoke_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        smoke_tests(smoke_dir)
        log("Smoke tests passed")
        return

    subjects = parse_subjects(Path(args.prompts_jsonl), args.max_subjects)
    if not subjects:
        subjects = PREFERRED_FORGET_SUBJECTS[:args.max_subjects]
    dataset_path = dataset_dir / "adversarial_forget_recovery.jsonl"
    dataset_rows = build_adversarial_dataset(subjects, dataset_path, rebuild=args.rebuild_dataset)
    log(f"Dataset rows: {len(dataset_rows)} subjects={subjects}")
    log(f"Dataset stats: {dict(Counter(r['probe_family'] for r in dataset_rows))}")

    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    model_paths = {
        "pre": args.model_dir,
        "kif": args.kif_adapter_path,
    }
    model_labels = [m for m in requested if model_paths.get(m)]
    if "kif" in requested and not args.kif_adapter_path:
        raise ValueError("--kif_adapter_path is required when models include kif")

    if args.dataset_only:
        summary = write_summary(out_dir, dataset_rows, model_labels, args)
        log(json.dumps(summary["dataset_stats"], indent=2, ensure_ascii=False))
        return

    tok = load_tokenizer(args.model_dir)
    for label in model_labels:
        eval_path = ckpt_dir / f"eval_rows_{label}.jsonl"
        done = completed_ids(eval_path)
        todo = [r for r in dataset_rows if r["row_id"] not in done]
        n = min(len(todo), args.max_eval_rows_per_model)
        if n <= 0:
            log(f"Eval complete for {label}: {len(done)}/{len(dataset_rows)}")
            continue
        log(f"Evaluating {label}: {n}/{len(todo)} remaining rows this run")
        model = load_model(model_paths[label], args.model_dir, args.device, args.load_mode)
        try:
            for r in todo[:n]:
                rec = eval_one_row(model, tok, r, label, args.device, args.max_new_tokens, args.el_steps, args.max_keywords)
                append_jsonl(eval_path, rec)
        finally:
            free_model(model)

    summary = write_summary(out_dir, dataset_rows, model_labels, args)
    log("Summary written")
    log(json.dumps({
        "dataset_stats": summary["dataset_stats"],
        "completion": summary["completion"],
        "same_row_ids_across_models": summary["same_row_ids_across_models"],
        "paper_key_results": summary["paper_key_results"],
        "output": str(out_dir / "adversarial_forget_recovery_summary.json"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
