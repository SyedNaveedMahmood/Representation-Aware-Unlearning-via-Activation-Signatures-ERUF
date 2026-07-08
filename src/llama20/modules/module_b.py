# Module B: Activation Probing (Hooks-in-Parallel + Batching)
# -----------------------------------------------------------
# One forward pass per batch captures ALL target layers via hooks, then saves
# per-prompt, per-layer activations to disk. This removes the O(num_layers)
# forward-pass bottleneck and fully utilizes the GPU with batching.

import os
import re
import gc
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

# Optional advanced libs with safe fallbacks
try:
    import compress_pickle  # for fast compressed dumps
    ADVANCED_ANALYSIS_AVAILABLE = True
except Exception:
    ADVANCED_ANALYSIS_AVAILABLE = False
    import pickle
    import gzip

    def _cp_dump(data, filename, compression="gzip", compresslevel=3, **kwargs):
        if compression != "gzip":
            raise ValueError("Only gzip compression supported in fallback.")
        with gzip.open(filename, "wb", compresslevel=compresslevel) as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _cp_load(filename):
        with gzip.open(filename, "rb") as f:
            return pickle.load(f)

    compress_pickle = type("compress_pickle_fallback", (), {"dump": staticmethod(_cp_dump), "load": staticmethod(_cp_load)})

# -----------------------------------------------------------
# Logging
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("kif_probe.log")]
)
logger = logging.getLogger("KIF-ModuleB")

# -----------------------------------------------------------
# Config
# -----------------------------------------------------------
@dataclass
class ProbeConfig:
    """Configuration for activation probing via parallel hooks."""
    # IO
    model_dir: str = "outputs/model"
    prompts_file: str = "outputs/datasets/prompts.jsonl"
    output_dir: Path = Path("outputs/activations")

    # What to capture
    layers: List[int] = field(default_factory=lambda: list(range(32)))  # adapt to your model depth
    targets: List[str] = field(default_factory=lambda: ["mlp"])       # capture MLPs by default

    # Performance
    batch_size: int = 32
    max_length: int = 128
    use_half_precision: bool = True              # fp16 for model weights (if supported)
    save_dtype_fp16: bool = True                 # store activations as float16 on disk

    # Device & memory
    device_map: str = "auto"                    # let HF shard model
    cleanup_every_batches: int = 10              # empty CUDA cache periodically

    # Storage
    compression_level: int = 3                   # gzip compress level for dumps

    # Optional: capture only last token to reduce storage ("full" | "last_token")
    capture_scope: str = "full"

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        (self.output_dir / "mlp").mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------
# Utilities
# -----------------------------------------------------------


def _parse_layers(value: str | None, default: List[int]) -> List[int]:
    """Parse comma/range layer specs such as "20-28" or "16,20-28,31"."""
    if not value or not value.strip():
        return list(default)
    layers = set()
    for part in value.split(","):
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
        raise ValueError(f"No valid layers parsed from KIF_MODULE_B_LAYERS={value!r}")
    return sorted(layers)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default)
    return [x.strip() for x in raw.split(",") if x.strip()]


def get_primary_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def to_numpy_for_saving(t: torch.Tensor, fp16: bool = True) -> np.ndarray:
    if t.is_cuda:
        t = t.detach().to("cpu")
    if fp16 and t.dtype in (torch.float32, torch.float64):
        t = t.half()
    # ensure contiguous for pickle speed
    t = t.contiguous()
    return t.numpy().astype(np.float16 if fp16 else np.float32)


# -----------------------------------------------------------
# Parallel Hook Collector
# -----------------------------------------------------------
class ParallelActivationCollector:
    """Attach hooks to all target modules once and capture outputs in a single pass."""

    def __init__(self, model: nn.Module, tokenizer: AutoTokenizer, config: ProbeConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = get_primary_device(model)

        # module name -> layer_idx
        self.module_to_layer: Dict[str, int] = {}
        # ordered list of target module names (deterministic iteration)
        self.target_module_names: List[str] = []

        # populated on each forward pass: module_name -> tensor(batch, seq, hidden)
        self.captured_activations: Dict[str, torch.Tensor] = {}
        self.hooks: List[Any] = []

        self._discover_targets()
        self._register_all_hooks()

    # ---- discovery ----
    def _discover_targets(self) -> None:
        names = []
        for name, module in self.model.named_modules():
            # target MLP blocks only
            if any(t in name.lower() for t in self.config.targets):
                # detect layer index like "layers.12" or "model.layers.12"
                m = re.search(r"layers\.(\d+)", name)
                if not m:
                    continue
                layer_idx = int(m.group(1))
                if layer_idx in self.config.layers:
                    names.append((layer_idx, name))
        # stable sort by layer index then name
        names.sort(key=lambda x: (x[0], x[1]))
        self.target_module_names = [n for _, n in names]
        self.module_to_layer = {n: i for i, n in names}

        if not self.target_module_names:
            raise RuntimeError("No target modules found. Check `layers` and `targets` in ProbeConfig.")

        logger.info(f"Target modules: {len(self.target_module_names)} across {len(set(self.module_to_layer.values()))} layers")

    # ---- hooks ----
    def _make_hook(self, module_name: str):
        def hook_fn(module, inputs, output):
            # For LlamaMLP and similar, output is a tensor
            out = output[0] if isinstance(output, tuple) else output
            # Detach immediately; keep on device to avoid D2H thrash
            self.captured_activations[module_name] = out.detach()
        return hook_fn

    def _register_all_hooks(self) -> None:
        named = dict(self.model.named_modules())
        for module_name in self.target_module_names:
            mod = named.get(module_name, None)
            if mod is None:
                logger.warning(f"Module not found during hook registration: {module_name}")
                continue
            self.hooks.append(mod.register_forward_hook(self._make_hook(module_name)))
        logger.info(f"Registered {len(self.hooks)} forward hooks")

    def remove_hooks(self) -> None:
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks.clear()
        logger.info("Removed all hooks")

    # ---- capture ----
    def _tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )
        # With device_map="auto", place inputs on the device of the embeddings (first param device)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        return enc

    def _maybe_slice_last_token(self, t: torch.Tensor) -> torch.Tensor:
        if self.config.capture_scope == "last_token":
            # keep shape (batch, 1, hidden) for consistency
            return t[:, -1:, :]
        return t

    @torch.inference_mode()
    def collect_batch(self, prompts: List[str], prompt_ids: List[str]) -> Dict[str, List[Path]]:
        """Run one forward pass and save per-prompt, per-layer activations.
        Returns mapping prompt_id -> list[Path] of saved files.
        """
        assert len(prompts) == len(prompt_ids)
        self.captured_activations.clear()

        inputs = self._tokenize_batch(prompts)
        # disable caches/extra outputs for speed if supported
        try:
            _ = self.model(**inputs, use_cache=False, output_hidden_states=False, output_attentions=False)
        except TypeError:
            _ = self.model(**inputs)

        # Save everything then drop references
        saved: Dict[str, List[Path]] = defaultdict(list)

        for module_name, tensor in self.captured_activations.items():
            layer_idx = self.module_to_layer.get(module_name)
            if layer_idx is None:
                continue

            tensor = self._maybe_slice_last_token(tensor)
            # tensor shape: (B, S, H) or (B, 1, H)
            B = tensor.shape[0]
            for b in range(B):
                pid = prompt_ids[b]
                data_np = to_numpy_for_saving(tensor[b], fp16=self.config.save_dtype_fp16)
                filename = f"{pid}_layer{layer_idx}_mlp.pkl.gz"
                path = self.config.output_dir / "mlp" / filename
                compress_pickle.dump(data_np, path, compression="gzip", compresslevel=self.config.compression_level)
                saved[pid].append(path)
                del data_np

            # free activation ASAP
            del tensor

        # post-batch cleanup
        self.captured_activations.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return saved

# -----------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------
class OptimizedProbeRobot:
    def __init__(self, config: ProbeConfig):
        self.config = config
        self._load_prompts()
        self._load_model()
        self.collector = ParallelActivationCollector(self.model, self.tokenizer, self.config)

    # ---- IO ----
    def _load_prompts(self) -> None:
        with open(self.config.prompts_file, "r", encoding="utf-8") as f:
            self.prompts: List[Dict[str, Any]] = [json.loads(line) for line in f]
        if not self.prompts:
            raise RuntimeError("No prompts found in prompts_file.")
        uniq = len({p.get("triple_id") for p in self.prompts})
        logger.info(f"Loaded {len(self.prompts)} prompts across {uniq} triples")

    def _load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_dir, use_fast=True)
        if self.tokenizer.pad_token is None:
            # fallback padding
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info(f"Loading model from: {self.config.model_dir}")
        # free caches before big load
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_dir,
            device_map=self.config.device_map,
            torch_dtype=torch.float16 if self.config.use_half_precision else torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        if torch.cuda.is_available():
            dev = get_primary_device(self.model)
            logger.info(f"Model loaded on device: {dev}; CUDA device count: {torch.cuda.device_count()}")
        else:
            logger.warning("CUDA not available; running on CPU.")

    # ---- collection ----
    def collect_all(self) -> Dict[str, List[Path]]:
        saved_all: Dict[str, List[Path]] = defaultdict(list)

        B = self.config.batch_size
        total = len(self.prompts)
        for start in tqdm(range(0, total, B), desc="Collecting activations"):
            batch = self.prompts[start:start + B]
            texts = [p["prompt"] for p in batch]
            ids = [p["id"] for p in batch]

            try:
                saved = self.collector.collect_batch(texts, ids)
                for pid, paths in saved.items():
                    saved_all[pid].extend(paths)
            except Exception as e:
                logger.error(f"Batch {start}-{start+len(batch)} failed: {e}")
                continue

            # periodic cleanup
            if torch.cuda.is_available() and ((start // max(B, 1) + 1) % self.config.cleanup_every_batches == 0):
                torch.cuda.empty_cache()

        return saved_all

    # ---- reports ----
    def _write_index(self, saved_paths: Dict[str, List[Path]]) -> None:
        index = {
            "config": {
                "layers": self.config.layers,
                "targets": self.config.targets,
                "model": self.config.model_dir,
                "batch_size": self.config.batch_size,
                "capture_scope": self.config.capture_scope,
                "fp16_model": self.config.use_half_precision,
                "fp16_saves": self.config.save_dtype_fp16,
            },
            "prompts": {
                pid: {
                    "paths": [str(p) for p in paths],
                    "triple_id": next((x.get("triple_id") for x in self.prompts if x.get("id") == pid), None),
                    "activation_count": len(paths),
                }
                for pid, paths in saved_paths.items()
            },
            "summary": {
                "total_prompts": len(saved_paths),
                "total_files": sum(len(v) for v in saved_paths.values()),
            },
        }
        path = self.config.output_dir / "activation_index.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
        logger.info(f"Wrote activation index to {path}")

    def _write_report(self, saved_paths: Dict[str, List[Path]]) -> None:
        total_size = 0
        layer_counts: Dict[int, int] = defaultdict(int)
        for plist in saved_paths.values():
            for p in plist:
                try:
                    total_size += os.path.getsize(p)
                    m = re.search(r"layer(\d+)", str(p))
                    if m:
                        layer_counts[int(m.group(1))] += 1
                except OSError:
                    pass

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.config.model_dir,
            "results": {
                "prompts_processed": len(saved_paths),
                "total_files": sum(len(v) for v in saved_paths.values()),
                "storage_gb": total_size / (1024 ** 3),
                "files_per_layer": dict(layer_counts),
            },
        }
        path = self.config.output_dir / "collection_report.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Wrote collection report to {path}")

    def run(self) -> None:
        logger.info("=" * 64)
        logger.info("Starting Module B: Hooks-in-Parallel + Batching")
        logger.info("=" * 64)

        saved = self.collect_all()
        self._write_index(saved)
        self._write_report(saved)

        # tidy up hooks before exit
        self.collector.remove_hooks()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Module B completed successfully ✔")


# -----------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------

def run_module_b():
    # Default remains full-depth capture, but local/cluster runs can override
    # without editing source. Examples:
    #   KIF_MODULE_B_LAYERS=20-28 llama20 module_b
    #   KIF_MODULE_B_LAYERS=20,22,24-28 KIF_MODULE_B_BATCH_SIZE=8 llama20 module_b
    default_layers = list(range(32))
    cfg = ProbeConfig(
        model_dir=os.environ.get("KIF_MODULE_B_MODEL_DIR", "outputs/model"),
        prompts_file=os.environ.get("KIF_MODULE_B_PROMPTS_FILE", "outputs/datasets/prompts.jsonl"),
        output_dir=Path(os.environ.get("KIF_MODULE_B_OUTPUT_DIR", "outputs/activations")),
        layers=_parse_layers(os.environ.get("KIF_MODULE_B_LAYERS"), default_layers),
        targets=_env_list("KIF_MODULE_B_TARGETS", ["mlp"]),
        batch_size=_env_int("KIF_MODULE_B_BATCH_SIZE", 32),
        max_length=_env_int("KIF_MODULE_B_MAX_LENGTH", 128),
        use_half_precision=_env_bool("KIF_MODULE_B_USE_HALF_PRECISION", True),
        save_dtype_fp16=_env_bool("KIF_MODULE_B_SAVE_DTYPE_FP16", True),
        device_map=os.environ.get("KIF_MODULE_B_DEVICE_MAP", "auto"),
        cleanup_every_batches=_env_int("KIF_MODULE_B_CLEANUP_EVERY_BATCHES", 10),
        compression_level=_env_int("KIF_MODULE_B_COMPRESSION_LEVEL", 3),
        capture_scope=os.environ.get("KIF_MODULE_B_CAPTURE_SCOPE", "full"),
    )
    logger.info(
        "Module B config: layers=%s targets=%s batch_size=%s max_length=%s capture_scope=%s output_dir=%s",
        cfg.layers, cfg.targets, cfg.batch_size, cfg.max_length, cfg.capture_scope, cfg.output_dir,
    )

    robot = OptimizedProbeRobot(cfg)
    robot.run()


if __name__ == "__main__":
    run_module_b()
