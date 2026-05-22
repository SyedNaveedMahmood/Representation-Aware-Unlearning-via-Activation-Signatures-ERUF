#src/llama20/loop.py

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama20.modules import module7, module8


def _extract_smr_el10(summary: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    """Pull SMR and EL10(post) from a Module 8 summary."""
    smr = summary.get("robustness_post_lora", {}).get("avg_subject_mention_rate")
    el10 = summary.get("extraction_likelihood", {}).get("EL10_post")
    return (
        float(smr) if smr is not None else None,
        float(el10) if el10 is not None else None,
    )


def run_forget_loop(
    confirm_rounds: int | None = None,
    smr_threshold: float | None = None,
    el10_threshold: float | None = None,
    max_days: float | None = None,
    eval_root: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Automated loop: train (Module 7) -> evaluate (Module 8) until SMR + EL10
    meet paper-style thresholds for a sustained window.

    Defaults (paper-style):
      - SMR <= 0.05 (5% tolerance)
      - EL10 < 1.0
      - 10 consecutive rounds meeting both thresholds
      - hard cap of 3 days wall-clock time
    """
    confirm_rounds = int(confirm_rounds or os.getenv("FORGET_CONFIRM_ROUNDS", 10))
    smr_threshold = float(smr_threshold or os.getenv("FORGET_SMR_THRESHOLD", 0.05))
    el10_threshold = float(el10_threshold or os.getenv("FORGET_EL10_THRESHOLD", 1.0))
    max_days = float(max_days or os.getenv("FORGET_MAX_DAYS", 3))
    eval_root = eval_root or os.getenv("FORGET_EVAL_ROOT", "outputs/eval_clean")

    history: List[Dict[str, Any]] = []
    consecutive_ok = 0
    start_time = time.monotonic()
    max_seconds = max_days * 24 * 60 * 60
    round_idx = 0

    while (time.monotonic() - start_time) < max_seconds:
        round_idx += 1
        elapsed_hours = (time.monotonic() - start_time) / 3600.0
        print(f"=== Round {round_idx} (elapsed {elapsed_hours:.2f}h) ===")

        # Train adapter (Module 7)
        adapter_path = module7.run_module7_qwen()
        if not adapter_path:
            history.append(
                {
                    "round": round_idx,
                    "status": "failed",
                    "reason": "module7 returned no adapter",
                }
            )
            break

        # Evaluate (Module 8) into a per-round folder
        eval_dir = Path(eval_root) / f"round_{round_idx}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        summary = module8.run_module8_clean(adapter_path=adapter_path, out_dir=str(eval_dir))

        # Metric extraction
        smr, el10 = _extract_smr_el10(summary or {})
        ok = (
            smr is not None
            and el10 is not None
            and smr <= smr_threshold
            and el10 < el10_threshold
        )
        consecutive_ok = consecutive_ok + 1 if ok else 0

        entry = {
            "round": round_idx,
            "adapter_path": adapter_path,
            "eval_dir": str(eval_dir),
            "smr": smr,
            "el10_post": el10,
            "ok": ok,
            "consecutive_ok": consecutive_ok,
            "elapsed_hours": elapsed_hours,
        }
        history.append(entry)
        print(
            "[Round {round}] SMR={smr} EL10={el10} ok={ok} streak={streak}/{target}".format(
                round=round_idx,
                smr=smr,
                el10=el10,
                ok=ok,
                streak=consecutive_ok,
                target=confirm_rounds,
            )
        )

        if consecutive_ok >= confirm_rounds:
            entry["stopped"] = True
            entry["reason"] = "thresholds_met"
            print("Stopping: SMR/EL10 thresholds met for required rounds.")
            break

    if (time.monotonic() - start_time) >= max_seconds:
        history.append(
            {
                "round": round_idx,
                "status": "stopped",
                "reason": "time_limit",
                "elapsed_hours": (time.monotonic() - start_time) / 3600.0,
            }
        )

    return history


if __name__ == "__main__":
    run_forget_loop()
