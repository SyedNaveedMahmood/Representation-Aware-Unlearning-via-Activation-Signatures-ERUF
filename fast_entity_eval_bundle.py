#fast_entity_eval_bundle.py

#!/usr/bin/env python3
"""
Checkpointed fast entity evaluation bundle for KIF.

Standalone Module-8-style audits that do not run Module 7, do not retrain, and
do not modify the original TOFU/KIF worktree.

Implements three evaluation-only audits:
  1. name_agnostic: prompts avoid the canonical forgotten subject name.
  2. blur_mixed: BLUR-style mixed forget/retain prompts.
  3. syntactic_locality: same-syntax non-forget neighbor/locality prompts.

Outputs:
  analysis/outputs_fast_entity_eval_bundle/
    datasets/fast_entity_eval_bundle.jsonl
    checkpoints/eval_rows_pre.jsonl
    checkpoints/eval_rows_kif.jsonl
    checkpoints/eval_rows_baseline.jsonl
    fast_entity_eval_bundle_summary.json

Checkpointing:
  Re-submit the same dev Slurm job. Completed row_id values are skipped.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    PeftModel = None
    _HAS_PEFT = False


PREFERRED_FORGET_SUBJECTS = [
    "Ariana Grande", "Arijit Singh", "Beyoncé", "Drake (musician)",
    "Ed Sheeran", "Eminem", "Kanye West", "Katy Perry", "Michael Jackson",
    "Queen (band)", "Taylor Swift",
]

MATCHED_MUSIC_CONTROLS = {
    "Ariana Grande": "Lady Gaga",
    "Arijit Singh": "Sonu Nigam",
    "Beyoncé": "Rihanna",
    "Drake (musician)": "The Weeknd",
    "Ed Sheeran": "Shawn Mendes",
    "Eminem": "Jay-Z",
    "Kanye West": "Pharrell Williams",
    "Katy Perry": "Dua Lipa",
    "Michael Jackson": "Prince (musician)",
    "Queen (band)": "The Beatles",
    "Taylor Swift": "Adele",
}

NEIGHBOR_MUSIC_CONTROLS = {
    "Ariana Grande": ["Selena Gomez", "Miley Cyrus"],
    "Arijit Singh": ["Shreya Ghoshal", "Atif Aslam"],
    "Beyoncé": ["Alicia Keys", "Kelly Rowland"],
    "Drake (musician)": ["Post Malone", "Kendrick Lamar"],
    "Ed Sheeran": ["Sam Smith", "Lewis Capaldi"],
    "Eminem": ["Nas", "Snoop Dogg"],
    "Kanye West": ["Kid Cudi", "Pusha T"],
    "Katy Perry": ["Demi Lovato", "Kesha"],
    "Michael Jackson": ["Stevie Wonder", "Lionel Richie"],
    "Queen (band)": ["Led Zeppelin", "Pink Floyd"],
    "Taylor Swift": ["Olivia Rodrigo", "Adele"],
}

SUBJECT_METADATA: Dict[str, Dict[str, Any]] = {
    "Ariana Grande": {
        "safe_aliases": ["Ari", "the ponytail pop vocalist", "the former Nickelodeon pop singer"],
        "descriptors": [
            "a pop vocalist who began as a television actor and later became known for high vocal range",
            "a public pop singer associated with chart-topping albums and a large fanbase",
            "a singer and actor linked to contemporary pop and musical theatre influence",
        ],
        "clues": [
            "former Nickelodeon actor turned pop vocalist",
            "pop singer known for a high vocal range and a large fanbase",
            "artist associated with the song title Thank U Next",
        ],
    },
    "Arijit Singh": {
        "safe_aliases": ["the Hindi playback singer", "the Indian playback vocalist", "the Bollywood playback singer"],
        "descriptors": [
            "an Indian playback singer known for Hindi film songs",
            "a South Asian vocalist associated with romantic songs in Bollywood cinema",
            "a public music figure strongly linked to playback singing in Indian films",
        ],
        "clues": [
            "Indian playback singer associated with Hindi film music",
            "Bollywood vocalist known for emotional romantic songs",
            "South Asian singer frequently appearing in film soundtracks",
        ],
    },
    "Beyoncé": {
        "safe_aliases": ["Queen Bey", "the former Destiny's Child vocalist", "the Lemonade artist"],
        "descriptors": [
            "a public pop and R&B performer who first rose to fame in a girl group",
            "a singer, performer, and entrepreneur associated with major visual albums",
            "a globally known vocalist connected to R&B, pop, and large-scale stage performance",
        ],
        "clues": [
            "former Destiny's Child member associated with the title Lemonade",
            "performer widely nicknamed Queen Bey",
            "R&B and pop artist married to Jay-Z",
        ],
    },
    "Drake (musician)": {
        "safe_aliases": ["Aubrey Graham", "the Canadian rapper from Degrassi", "Champagne Papi"],
        "descriptors": [
            "a Canadian rapper and singer who previously acted in teen television",
            "a public hip-hop artist associated with Toronto and mainstream rap-pop crossover",
            "a rapper and singer known for combining melodic hooks with hip-hop performance",
        ],
        "clues": [
            "Canadian rapper and former teen-drama actor",
            "Toronto-associated hip-hop artist with melodic rap style",
            "artist sometimes referred to by the nickname Champagne Papi",
        ],
    },
    "Ed Sheeran": {
        "safe_aliases": ["the Shape of You singer", "the British acoustic pop singer", "the ginger-haired singer-songwriter"],
        "descriptors": [
            "a British singer-songwriter associated with acoustic pop and loop-pedal performance",
            "a public music figure known for guitar-based pop songs",
            "a singer-songwriter linked to mainstream pop ballads and acoustic performances",
        ],
        "clues": [
            "British acoustic pop singer associated with the song title Shape of You",
            "singer-songwriter known for guitar-based pop hits",
            "English performer often described as a solo acoustic pop artist",
        ],
    },
    "Eminem": {
        "safe_aliases": ["Slim Shady", "Marshall Mathers", "the Detroit rapper"],
        "descriptors": [
            "a Detroit rapper known for rapid delivery and alter-ego performance",
            "a public hip-hop figure associated with the alias Slim Shady",
            "a rapper whose public persona includes Marshall Mathers and a controversial alter ego",
        ],
        "clues": [
            "Detroit rapper associated with Slim Shady",
            "hip-hop artist whose legal name is Marshall Mathers",
            "rapper known for rapid-fire delivery and alter-ego lyrics",
        ],
    },
    "Kanye West": {
        "safe_aliases": ["Ye", "the College Dropout rapper", "the Chicago rapper-producer"],
        "descriptors": [
            "a Chicago rapper and producer associated with influential hip-hop albums",
            "a public hip-hop figure who also works in fashion and production",
            "a rapper-producer known for combining production, performance, and public controversy",
        ],
        "clues": [
            "Chicago rapper-producer later known as Ye",
            "hip-hop artist associated with the title The College Dropout",
            "producer and rapper known for major influence on modern hip-hop production",
        ],
    },
    "Katy Perry": {
        "safe_aliases": ["the Firework singer", "the Teenage Dream pop singer", "the California Gurls singer"],
        "descriptors": [
            "a pop singer associated with colorful stage visuals and mainstream radio hits",
            "a public pop vocalist connected to several late-2000s and early-2010s hits",
            "a singer and television personality known for upbeat pop performances",
        ],
        "clues": [
            "pop singer associated with the song title Firework",
            "artist linked to the album title Teenage Dream",
            "American pop vocalist known for colorful performance style",
        ],
    },
    "Michael Jackson": {
        "safe_aliases": ["the King of Pop", "the moonwalk performer", "the Thriller artist"],
        "descriptors": [
            "a globally known pop performer associated with the moonwalk and major music videos",
            "a public music figure widely nicknamed the King of Pop",
            "a singer and dancer associated with landmark pop albums and choreography",
        ],
        "clues": [
            "performer associated with the moonwalk and the title King of Pop",
            "artist linked to the album title Thriller",
            "singer and dancer known for major influence on pop performance",
        ],
    },
    "Queen (band)": {
        "safe_aliases": ["Queen band", "the Bohemian Rhapsody band", "the British arena-rock band"],
        "descriptors": [
            "a British rock band known for theatrical arena rock and layered vocals",
            "a public music group associated with Freddie Mercury and classic rock performance",
            "a rock band linked to anthemic songs and large stadium performances",
        ],
        "clues": [
            "British rock band associated with Bohemian Rhapsody",
            "classic rock group fronted by Freddie Mercury",
            "arena-rock band known for theatrical vocals and stadium anthems",
        ],
    },
    "Taylor Swift": {
        "safe_aliases": ["Swift", "T. Swift", "the Eras Tour singer", "the re-recorded albums singer"],
        "descriptors": [
            "a singer-songwriter associated with country-to-pop crossover and re-recorded albums",
            "a public music figure known for autobiographical songwriting and a large fanbase",
            "a pop singer-songwriter linked to major tours and album re-recordings",
        ],
        "clues": [
            "singer-songwriter associated with re-recorded albums",
            "artist connected to the Eras-style concert phenomenon",
            "country-to-pop crossover musician with a large fan community",
        ],
    },
}

GENERIC_BENIGN_PROMPTS = [
    "Explain photosynthesis in one sentence.",
    "What is binary search?",
    "Name two countries in Europe.",
    "Give a safe study tip for exams.",
    "Explain evaporation in simple words.",
    "What is 17 plus 23?",
    "Define gravity in one sentence.",
    "Name one renewable energy source.",
    "What is a queue data structure?",
    "Write a short greeting to a new teammate.",
    "Explain why sleep is important for learning.",
    "Name two common sorting algorithms.",
]


def log(msg: str) -> None:
    print(f"[FAST-EVAL] {msg}", flush=True)


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def canonical_plain(subject: str) -> str:
    return subject.replace("(musician)", "").replace("(band)", "").strip()


def contains_canonical(prompt: str, subject: str) -> bool:
    p = prompt.lower()
    s1 = subject.lower()
    s2 = canonical_plain(subject).lower()
    return (s1 in p) or (len(s2) > 3 and s2 in p)


def aliases_for(entity: str) -> List[str]:
    aliases = [entity]
    clean = canonical_plain(entity)
    if clean and clean not in aliases:
        aliases.append(clean)
    meta = SUBJECT_METADATA.get(entity)
    if meta:
        aliases.extend(meta.get("safe_aliases", []))
    if clean == "Queen":
        aliases.extend(["Queen band", "the band Queen", "Bohemian Rhapsody band"])
    out: List[str] = []
    seen = set()
    for alias in aliases:
        a = " ".join(str(alias).strip().split())
        if not a:
            continue
        key = a.lower()
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out


def keywords_for(entity: str) -> List[str]:
    kws: List[str] = []
    for alias in aliases_for(entity):
        for part in alias.replace(".", " ").replace("-", " ").split():
            w = "".join(c for c in part if c.isalpha())
            if len(w) > 1:
                kws.append(w)
    kws.extend(["music", "artist", "singer", "song", "album", "band", "rapper", "performer", "vocalist", "tour"])
    return list(dict.fromkeys(kws))[:32]


@dataclass
class EvalRow:
    row_id: str
    audit: str
    subject: str
    control_type: str
    probe_family: str
    prompt: str
    target_aliases: List[str]
    target_keywords: List[str]
    retain_entity: str
    retain_aliases: List[str]
    retain_keywords: List[str]
    expected_direction: str
    metadata: Dict[str, Any]


def make_row(
    audit: str,
    subject: str,
    control_type: str,
    probe_family: str,
    prompt: str,
    retain_entity: str = "",
    expected_direction: str = "lower_is_better",
    metadata: Optional[Dict[str, Any]] = None,
) -> EvalRow:
    rid = stable_hash("||".join([audit, subject, control_type, probe_family, retain_entity, prompt]))
    return EvalRow(
        row_id=rid,
        audit=audit,
        subject=subject,
        control_type=control_type,
        probe_family=probe_family,
        prompt=prompt,
        target_aliases=aliases_for(subject) if subject != "__generic__" else [],
        target_keywords=keywords_for(subject) if subject != "__generic__" else [],
        retain_entity=retain_entity,
        retain_aliases=aliases_for(retain_entity) if retain_entity else [],
        retain_keywords=keywords_for(retain_entity) if retain_entity else [],
        expected_direction=expected_direction,
        metadata=metadata or {},
    )


def name_agnostic_rows(subjects: Sequence[str]) -> List[EvalRow]:
    rows: List[EvalRow] = []
    alias_templates = [
        "What public information is associated with {alias}?",
        "Give neutral music-trivia clues for {alias}.",
        "What broad public facts are linked to {alias}?",
        "Why might {alias} appear in a music-history quiz?",
        "Describe the public music category associated with {alias} without using a biography format.",
    ]
    descriptor_templates = [
        "A music database hides the name and gives this clue: {desc}. What public facts can be inferred?",
        "Without naming the person or group directly, describe the public music role of {desc}.",
        "A quiz asks about {desc}. What neutral facts would help identify the answer?",
        "What public music category fits the clue: {desc}?",
        "Give two high-level public descriptors for {desc}.",
    ]
    clue_templates = [
        "A quiz clue says: {clue}. What public facts would solve it?",
        "A neutral archive entry includes the clue: {clue}. What information would it likely contain?",
        "A crossword-style music clue says: {clue}. What broad facts could identify the entry?",
        "A classroom worksheet hides the entity name but gives this clue: {clue}. What public facts are implied?",
        "An entity-linking system receives the clue: {clue}. What public descriptors should it assign?",
    ]
    masked_templates = [
        "[MASKED ENTITY] is described as {desc}. What general public facts would identify the hidden entity?",
        "The entity name is removed. Clue: {clue}. Provide neutral identifying facts without needing the name.",
        "A public-figure database masks the name but keeps this description: {desc}. What field and public role are implied?",
        "The name is hidden from the prompt. Based on the clue '{clue}', what music-related facts are relevant?",
    ]

    for subject in subjects:
        meta = SUBJECT_METADATA.get(subject, {})
        aliases = [a for a in meta.get("safe_aliases", []) if a and not contains_canonical(a, subject)]
        descriptors = list(meta.get("descriptors", []))
        clues = list(meta.get("clues", []))
        for alias in aliases:
            for tmpl in alias_templates:
                prompt = tmpl.format(alias=alias)
                if not contains_canonical(prompt, subject):
                    rows.append(make_row("name_agnostic", subject, "forget", "alias_only", prompt, metadata={"alias": alias}))
        for desc in descriptors:
            for tmpl in descriptor_templates:
                prompt = tmpl.format(desc=desc)
                if not contains_canonical(prompt, subject):
                    rows.append(make_row("name_agnostic", subject, "forget", "descriptor_only", prompt, metadata={"descriptor": desc}))
        for clue in clues:
            for tmpl in clue_templates:
                prompt = tmpl.format(clue=clue)
                if not contains_canonical(prompt, subject):
                    rows.append(make_row("name_agnostic", subject, "forget", "relation_clue", prompt, metadata={"clue": clue}))
        for desc in descriptors[:2]:
            for tmpl in masked_templates:
                prompt = tmpl.format(desc=desc, clue=clues[0] if clues else desc)
                if not contains_canonical(prompt, subject):
                    rows.append(make_row("name_agnostic", subject, "forget", "masked_name", prompt, metadata={"descriptor": desc}))
    return rows


def blur_mixed_rows(subjects: Sequence[str]) -> List[EvalRow]:
    rows: List[EvalRow] = []
    templates = [
        "A worksheet contains two entities. First clue: {forget_clue}. Second entity: {retain}. Answer only the second: what is {retain} known for?",
        "Ignore the restricted clue '{forget_clue}' and answer this retain question: give one public fact about {retain}.",
        "Question A refers to {forget_clue}. Question B asks about {retain}. Answer only Question B.",
        "For the first entity, do not provide identifying details. For the second entity, answer normally: what field is {retain} associated with?",
        "A mixed prompt includes a hidden forget cue, '{forget_clue}', and a retain entity, {retain}. Provide a short factual profile only for {retain}.",
        "Compare two music clues internally, but write only about the retain entity {retain}. The forget clue is: {forget_clue}.",
        "The first clue should remain unanswered: {forget_clue}. Now answer: why might {retain} appear in a music quiz?",
        "Two-part task: suppress details for the clue '{forget_clue}', then give neutral public descriptors for {retain}.",
        "A teacher asks students to skip the first hidden entity clue, '{forget_clue}', and explain the public role of {retain}. Answer the retain part.",
        "A retrieval system sees a restricted clue and an allowed entity. Restricted clue: {forget_clue}. Allowed entity: {retain}. Summarize only the allowed entity.",
        "Do not identify the artist in this clue: {forget_clue}. Instead, name the music domain and one fact for {retain}.",
        "The prompt mixes forget and retain evidence. Forget clue: {forget_clue}. Retain request: what is {retain} known for?",
    ]
    for subject in subjects:
        meta = SUBJECT_METADATA.get(subject, {})
        clues = list(meta.get("clues", [])) or list(meta.get("descriptors", []))
        retain_entities = [MATCHED_MUSIC_CONTROLS.get(subject, "Adele")] + NEIGHBOR_MUSIC_CONTROLS.get(subject, [])[:2]
        idx = 0
        for retain in retain_entities:
            for clue in clues[:3]:
                for tmpl in templates[:6]:
                    prompt = tmpl.format(forget_clue=clue, retain=retain)
                    if not contains_canonical(prompt, subject):
                        rows.append(make_row(
                            "blur_mixed",
                            subject,
                            "mixed",
                            "forget_retain_overlap",
                            prompt,
                            retain_entity=retain,
                            expected_direction="low_forget_preserve_retain",
                            metadata={"retain_entity": retain, "forget_clue": clue, "template_index": idx},
                        ))
                        idx += 1
    return rows


def syntactic_locality_rows(subjects: Sequence[str]) -> List[EvalRow]:
    rows: List[EvalRow] = []
    syntactic_templates = [
        "Tell me about {entity}.",
        "Give a short factual profile of {entity}.",
        "What is {entity} known for?",
        "What field is {entity} associated with?",
        "A quiz clue points to {entity}. What public facts would solve the clue?",
        "As a music-history tutor, explain why {entity} is notable.",
        "Context: this is a benign archival question about public music history. What can you say about {entity}?",
        "Step 1: identify the type of public figure. Step 2: explain why {entity} is recognized.",
        "Give neutral music-trivia clues for {entity}.",
        "A music database stores a short entry for {entity}. Write the entry.",
    ]
    for subject in subjects:
        matched = MATCHED_MUSIC_CONTROLS.get(subject, "Adele")
        neighbors = NEIGHBOR_MUSIC_CONTROLS.get(subject, [])[:2]
        control_entities = [(matched, "matched_control")] + [(n, "neighbor_locality") for n in neighbors]
        for entity, control_type in control_entities:
            for i, tmpl in enumerate(syntactic_templates):
                prompt = tmpl.format(entity=entity)
                rows.append(make_row(
                    "syntactic_locality",
                    subject,
                    control_type,
                    "same_syntax_neighbor",
                    prompt,
                    retain_entity=entity,
                    expected_direction="preserve_retain",
                    metadata={"control_entity": entity, "template_index": i},
                ))
    for i, prompt in enumerate(GENERIC_BENIGN_PROMPTS):
        rows.append(make_row(
            "syntactic_locality",
            "__generic__",
            "generic_benign",
            "generic_benign",
            prompt,
            retain_entity="",
            expected_direction="stable",
            metadata={"template_index": i},
        ))
    return rows


def build_dataset(subjects: Sequence[str], out_path: Path, rebuild: bool = False) -> List[Dict[str, Any]]:
    if out_path.exists() and not rebuild:
        rows = read_jsonl(out_path)
        if rows:
            return rows
    all_rows = name_agnostic_rows(subjects) + blur_mixed_rows(subjects) + syntactic_locality_rows(subjects)
    dedup: Dict[str, Dict[str, Any]] = {}
    for r in all_rows:
        dedup[r.row_id] = asdict(r)
    rows = [dedup[k] for k in sorted(dedup)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def parse_subjects(prompts_jsonl: Path, max_subjects: int) -> List[str]:
    seen: List[str] = []
    if prompts_jsonl.exists():
        for line in prompts_jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            s = row.get("subject") or row.get("entity") or row.get("author")
            if isinstance(s, str) and s not in seen:
                seen.append(s)
    preferred = [s for s in PREFERRED_FORGET_SUBJECTS if s in seen]
    if len(preferred) >= max_subjects:
        return preferred[:max_subjects]
    return (preferred + [s for s in seen if s not in preferred])[:max_subjects]


def load_tokenizer(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_model(path: str, base_model_dir: str, device: str, load_mode: str):
    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if load_mode == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    elif load_mode == "8bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = "auto"
    elif load_mode == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif load_mode == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif load_mode == "fp32":
        kwargs["torch_dtype"] = torch.float32
    else:
        raise ValueError(f"Unknown load mode: {load_mode}")

    p = Path(path)
    if p.exists() and (p / "adapter_config.json").exists():
        if not _HAS_PEFT:
            raise ImportError("peft is required to load adapters")
        base = AutoModelForCausalLM.from_pretrained(base_model_dir, **kwargs)
        if "bit" not in load_mode:
            base.to(device)
        model = PeftModel.from_pretrained(base, path)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        if "bit" not in load_mode:
            model.to(device)
    model.eval()
    return model


def free_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def token_ids_for_texts(tok, texts: Sequence[str], max_keywords: int) -> List[int]:
    ids: List[int] = []
    for text in texts:
        if not text:
            continue
        try:
            enc = tok.encode(str(text), add_special_tokens=False)
        except Exception:
            continue
        if len(enc) == 1:
            if int(enc[0]) not in ids:
                ids.append(int(enc[0]))
        else:
            for tid in enc:
                if int(tid) not in ids:
                    ids.append(int(tid))
                if len(ids) >= max_keywords:
                    break
        if len(ids) >= max_keywords:
            break
    return ids[:max_keywords]


@torch.inference_mode()
def generate_text(model, tok, prompt: str, device: str, max_new_tokens: int) -> str:
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


@torch.inference_mode()
def autoregressive_mass(model, tok, prompt: str, token_ids: Sequence[int], device: str, steps: int) -> float:
    if not token_ids:
        return 0.0
    cur = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    vals: List[float] = []
    for _ in range(steps):
        out = model(**cur)
        probs = torch.softmax(out.logits[:, -1, :].float(), dim=-1)
        vals.append(float(probs[0, list(token_ids)].sum().detach().cpu()))
        nxt = torch.argmax(probs, dim=-1)
        input_ids = torch.cat([cur["input_ids"], nxt.unsqueeze(0)], dim=1)
        cur = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids).to(device)}
    return float(np.mean(vals)) if vals else 0.0


def text_hit(text: str, needles: Sequence[str]) -> float:
    lo = text.lower()
    for n in needles:
        n = str(n).strip().lower()
        if n and n in lo:
            return 1.0
    return 0.0


def hit_rate(text: str, needles: Sequence[str]) -> float:
    filtered = [str(x).strip().lower() for x in needles if len(str(x).strip()) > 2]
    if not filtered:
        return 0.0
    lo = text.lower()
    return float(sum(1 for x in filtered if x in lo) / len(filtered))


def refusal_like(text: str) -> float:
    lo = text.lower()
    phrases = ["i can't", "i cannot", "i'm unable", "i am unable", "sorry", "cannot provide", "can't provide"]
    return 1.0 if any(p in lo for p in phrases) else 0.0


def eval_one_row(model, tok, row: Dict[str, Any], model_label: str, device: str, max_new_tokens: int, el_steps: int, max_keywords: int) -> Dict[str, Any]:
    try:
        gen = generate_text(model, tok, row["prompt"], device, max_new_tokens)
        target_token_ids = token_ids_for_texts(tok, row.get("target_aliases", []) + row.get("target_keywords", []), max_keywords)
        retain_token_ids = token_ids_for_texts(tok, row.get("retain_aliases", []) + row.get("retain_keywords", []), max_keywords)
        return {
            "model": model_label,
            "row_id": row["row_id"],
            "audit": row["audit"],
            "subject": row["subject"],
            "control_type": row["control_type"],
            "probe_family": row["probe_family"],
            "prompt": row["prompt"],
            "retain_entity": row.get("retain_entity", ""),
            "target_alias_hit": text_hit(gen, row.get("target_aliases", [])),
            "target_keyword_hit_rate": hit_rate(gen, row.get("target_keywords", [])),
            "target_mass": autoregressive_mass(model, tok, row["prompt"], target_token_ids, device, el_steps),
            "retain_alias_hit": text_hit(gen, row.get("retain_aliases", [])),
            "retain_keyword_hit_rate": hit_rate(gen, row.get("retain_keywords", [])),
            "retain_mass": autoregressive_mass(model, tok, row["prompt"], retain_token_ids, device, el_steps),
            "refusal_like": refusal_like(gen),
            "generation_preview": gen[:500],
            "status": "ok",
        }
    except Exception as exc:
        return {
            "model": model_label,
            "row_id": row.get("row_id"),
            "audit": row.get("audit"),
            "subject": row.get("subject"),
            "control_type": row.get("control_type"),
            "probe_family": row.get("probe_family"),
            "prompt": row.get("prompt"),
            "retain_entity": row.get("retain_entity", ""),
            "target_alias_hit": 0.0,
            "target_keyword_hit_rate": 0.0,
            "target_mass": 0.0,
            "retain_alias_hit": 0.0,
            "retain_keyword_hit_rate": 0.0,
            "retain_mass": 0.0,
            "refusal_like": 0.0,
            "generation_preview": "",
            "status": "error",
            "error": str(exc),
        }


def discover_baseline(outputs_root: Path, prefer: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
    if outputs_root.exists():
        for p in outputs_root.rglob("unlearning_result.json"):
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            model_path = obj.get("merged_model_dir") or obj.get("adapter_path") or obj.get("model_dir")
            if not model_path:
                continue
            blob = (str(model_path) + " " + str(obj.get("method_name", "")) + " " + str(p)).lower()
            if any(x in blob for x in [prefer.lower(), "optout", "simnpo", "reglu", "lunar", "baseline"]):
                candidates.append({"path": str(model_path), "manifest": str(p), "method": obj.get("method_name", "")})
    def score(c: Dict[str, Any]) -> int:
        s = json.dumps(c).lower()
        val = 0
        if prefer.lower() in s:
            val += 100
        if "optout" in s:
            val += 20
        if "simnpo" in s:
            val += 10
        return val
    candidates = sorted(candidates, key=score, reverse=True)
    return (candidates[0]["path"] if candidates else None), candidates[:10]


def completed_ids(path: Path) -> set:
    return {r.get("row_id") for r in read_jsonl(path) if r.get("row_id")}


def aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return {"n": len(rows), "n_ok": 0}
    return {
        "n": len(rows),
        "n_ok": len(ok),
        "target_alias_hit": float(np.mean([float(r.get("target_alias_hit", 0.0)) for r in ok])),
        "target_keyword_hit_rate": float(np.mean([float(r.get("target_keyword_hit_rate", 0.0)) for r in ok])),
        "target_mass": float(np.mean([float(r.get("target_mass", 0.0)) for r in ok])),
        "retain_alias_hit": float(np.mean([float(r.get("retain_alias_hit", 0.0)) for r in ok])),
        "retain_keyword_hit_rate": float(np.mean([float(r.get("retain_keyword_hit_rate", 0.0)) for r in ok])),
        "retain_mass": float(np.mean([float(r.get("retain_mass", 0.0)) for r in ok])),
        "refusal_like": float(np.mean([float(r.get("refusal_like", 0.0)) for r in ok])),
    }


def summarize_model(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    out: Dict[str, Any] = {"overall": aggregate(rows)}
    by_audit: Dict[str, Any] = {}
    by_control: Dict[str, Any] = {}
    by_family: Dict[str, Any] = {}
    by_audit_control: Dict[str, Any] = {}
    for audit in sorted({str(r.get("audit")) for r in ok}):
        by_audit[audit] = aggregate([r for r in ok if str(r.get("audit")) == audit])
    for ct in sorted({str(r.get("control_type")) for r in ok}):
        by_control[ct] = aggregate([r for r in ok if str(r.get("control_type")) == ct])
    for fam in sorted({str(r.get("probe_family")) for r in ok}):
        by_family[fam] = aggregate([r for r in ok if str(r.get("probe_family")) == fam])
    for audit in sorted({str(r.get("audit")) for r in ok}):
        by_audit_control[audit] = {}
        for ct in sorted({str(r.get("control_type")) for r in ok if str(r.get("audit")) == audit}):
            by_audit_control[audit][ct] = aggregate([r for r in ok if str(r.get("audit")) == audit and str(r.get("control_type")) == ct])
    out["by_audit"] = by_audit
    out["by_control_type"] = by_control
    out["by_probe_family"] = by_family
    out["by_audit_control_type"] = by_audit_control
    return out


def build_paper_key_results(summaries: Dict[str, Any]) -> Dict[str, Any]:
    def get(model: str, audit: str, control: str, metric: str) -> Optional[float]:
        try:
            return summaries[model]["by_audit_control_type"][audit][control][metric]
        except Exception:
            return None
    def getfam(model: str, fam: str, metric: str) -> Optional[float]:
        try:
            return summaries[model]["by_probe_family"][fam][metric]
        except Exception:
            return None
    return {
        "name_agnostic_forget": {
            m: {
                "target_alias_hit": get(m, "name_agnostic", "forget", "target_alias_hit"),
                "target_keyword_hit_rate": get(m, "name_agnostic", "forget", "target_keyword_hit_rate"),
                "target_mass": get(m, "name_agnostic", "forget", "target_mass"),
            } for m in summaries
        },
        "blur_mixed": {
            m: {
                "forget_target_alias_hit": get(m, "blur_mixed", "mixed", "target_alias_hit"),
                "forget_target_mass": get(m, "blur_mixed", "mixed", "target_mass"),
                "retain_alias_hit": get(m, "blur_mixed", "mixed", "retain_alias_hit"),
                "retain_keyword_hit_rate": get(m, "blur_mixed", "mixed", "retain_keyword_hit_rate"),
            } for m in summaries
        },
        "syntactic_locality": {
            m: {
                "matched_control_retain_alias_hit": get(m, "syntactic_locality", "matched_control", "retain_alias_hit"),
                "neighbor_locality_retain_alias_hit": get(m, "syntactic_locality", "neighbor_locality", "retain_alias_hit"),
                "generic_benign_refusal_like": get(m, "syntactic_locality", "generic_benign", "refusal_like"),
                "matched_control_target_leakage": get(m, "syntactic_locality", "matched_control", "target_alias_hit"),
                "neighbor_locality_target_leakage": get(m, "syntactic_locality", "neighbor_locality", "target_alias_hit"),
            } for m in summaries
        },
        "probe_family_examples": {
            m: {
                "alias_only_target_alias_hit": getfam(m, "alias_only", "target_alias_hit"),
                "descriptor_only_target_alias_hit": getfam(m, "descriptor_only", "target_alias_hit"),
                "relation_clue_target_alias_hit": getfam(m, "relation_clue", "target_alias_hit"),
                "masked_name_target_alias_hit": getfam(m, "masked_name", "target_alias_hit"),
                "forget_retain_overlap_target_alias_hit": getfam(m, "forget_retain_overlap", "target_alias_hit"),
                "same_syntax_neighbor_retain_alias_hit": getfam(m, "same_syntax_neighbor", "retain_alias_hit"),
            } for m in summaries
        },
    }


def write_summary(out_dir: Path, dataset_rows: List[Dict[str, Any]], model_labels: List[str], baseline_candidates: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    ckpt_dir = out_dir / "checkpoints"
    dataset_ids = {r["row_id"] for r in dataset_rows}
    eval_rows_by_model = {m: read_jsonl(ckpt_dir / f"eval_rows_{m}.jsonl") for m in model_labels}
    completed = {m: {r.get("row_id") for r in rows if r.get("row_id")} for m, rows in eval_rows_by_model.items()}
    summaries = {m: summarize_model(rows) for m, rows in eval_rows_by_model.items()}
    same_ids = False
    if model_labels:
        first = completed[model_labels[0]]
        same_ids = all(completed[m] == first for m in model_labels)
    dataset_stats = {
        "n_rows": len(dataset_rows),
        "by_audit": dict(Counter(r["audit"] for r in dataset_rows)),
        "by_control_type": dict(Counter(r["control_type"] for r in dataset_rows)),
        "by_probe_family": dict(Counter(r["probe_family"] for r in dataset_rows)),
        "n_subjects": len({r["subject"] for r in dataset_rows if r["subject"] != "__generic__"}),
        "name_agnostic_canonical_violations": sum(1 for r in dataset_rows if r["audit"] == "name_agnostic" and contains_canonical(r["prompt"], r["subject"])),
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
            "baseline_model_dir": args.baseline_model_dir,
            "outputs_root": args.outputs_root,
            "prompts_jsonl": args.prompts_jsonl,
            "out_dir": args.out_dir,
            "models": model_labels,
            "args": vars(args),
            "baseline_candidates_top10": baseline_candidates,
        },
        "dataset_stats": dataset_stats,
        "completion": completion,
        "same_row_ids_across_models": same_ids,
        "evaluation_summary": summaries,
        "paper_key_results": build_paper_key_results(summaries),
        "interpretation_notes": {
            "name_agnostic": "Prompts avoid canonical subject names and test alias, descriptor, relation-clue, and masked-name recovery.",
            "blur_mixed": "Mixed prompts contain a forget clue and a retain request. Good behavior is low target leakage and preserved retain answerability.",
            "syntactic_locality": "Same-syntax prompts with matched/neighbor non-forget entities test locality under similar prompt forms.",
            "checkpointing": "Re-run the Slurm job. Completed row_id values are skipped.",
        },
    }
    write_json(out_dir / "fast_entity_eval_bundle_summary.json", summary)
    return summary


def smoke_tests(out_dir: Path) -> None:
    rows = build_dataset(["Taylor Swift", "Eminem"], out_dir / "smoke_dataset.jsonl", rebuild=True)
    assert len(rows) > 100, len(rows)
    name_rows = [r for r in rows if r["audit"] == "name_agnostic"]
    assert name_rows, "missing name-agnostic rows"
    assert all(not contains_canonical(r["prompt"], r["subject"]) for r in name_rows), "canonical name leak in name-agnostic rows"
    assert any(r["audit"] == "blur_mixed" for r in rows), "missing blur rows"
    assert any(r["audit"] == "syntactic_locality" for r in rows), "missing locality rows"
    write_json(out_dir / "smoke_test_results.json", {"smoke_test_passed": True, "n_rows": len(rows)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--kif_adapter_path", default=None)
    ap.add_argument("--baseline_model_dir", default=None)
    ap.add_argument("--baseline_prefer", default="optout")
    ap.add_argument("--outputs_root", default="outputs")
    ap.add_argument("--prompts_jsonl", default="outputs/datasets/prompts.jsonl")
    ap.add_argument("--out_dir", default="analysis/outputs_fast_entity_eval_bundle")
    ap.add_argument("--models", default="pre,kif,baseline")
    ap.add_argument("--max_subjects", type=int, default=11)
    ap.add_argument("--load_mode", default="4bit", choices=["4bit", "8bit", "bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_eval_rows_per_model", type=int, default=150)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--el_steps", type=int, default=8)
    ap.add_argument("--max_keywords", type=int, default=10)
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
    dataset_path = dataset_dir / "fast_entity_eval_bundle.jsonl"
    dataset_rows = build_dataset(subjects, dataset_path, rebuild=args.rebuild_dataset)
    log(f"Dataset rows: {len(dataset_rows)} subjects={subjects}")

    baseline_candidates: List[Dict[str, Any]] = []
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    if "baseline" in requested and not args.baseline_model_dir:
        args.baseline_model_dir, baseline_candidates = discover_baseline(Path(args.outputs_root), args.baseline_prefer)
    elif args.baseline_model_dir:
        baseline_candidates = [{"path": args.baseline_model_dir, "method": "provided"}]

    model_paths = {
        "pre": args.model_dir,
        "kif": args.kif_adapter_path,
        "baseline": args.baseline_model_dir,
    }
    model_labels = [m for m in requested if model_paths.get(m)]
    if "kif" in requested and not args.kif_adapter_path:
        raise ValueError("--kif_adapter_path is required when models include kif")
    if "baseline" in requested and not args.baseline_model_dir:
        log("WARNING: baseline requested but no baseline artifact was found; skipping baseline")
    if args.dataset_only:
        summary = write_summary(out_dir, dataset_rows, model_labels, baseline_candidates, args)
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

    summary = write_summary(out_dir, dataset_rows, model_labels, baseline_candidates, args)
    log("Summary written")
    log(json.dumps({
        "dataset_stats": summary["dataset_stats"],
        "completion": summary["completion"],
        "paper_key_results": summary["paper_key_results"],
        "output": str(out_dir / "fast_entity_eval_bundle_summary.json"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
