# Module C: Signature Mining with ROME Integration (CUDA-Accelerated — Operational Version with R3 Fixes)

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time
import logging
import torch
import numpy as np
import gc
import re
import pickle
import gzip
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")

try:
    import compress_pickle
except ImportError:
    class CompressPickleFallback:
        def load(self, filename):
            with gzip.open(filename, 'rb') as f:
                return pickle.load(f)
        def dump(self, data, filename, compression="gzip", compresslevel=3, **kwargs):
            with gzip.open(filename, 'wb', compresslevel=compresslevel) as f:
                pickle.dump(data, f)
    compress_pickle = CompressPickleFallback()
    logging.warning("compress_pickle not available, using fallback.")


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("kif_signature_cuda_balanced.log")
    ]
)
logger = logging.getLogger('KIF-ModuleC-CUDA-Balanced')


# ============================================================
# SECTION 1 — CUDA-Accelerated Utilities
# ============================================================

class StandardScaler:
    """PyTorch StandardScaler with GPU support."""
    def __init__(self, device='cpu'):
        self.mean_  = None
        self.scale_ = None
        self.device = device

    def fit(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)
        self.mean_  = torch.mean(X, dim=0)
        self.scale_ = torch.std(X, dim=0, unbiased=True)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class PCA:
    """PyTorch PCA via full SVD with GPU support."""
    def __init__(self, n_components=None, random_state=None, device='cpu'):
        self.n_components              = n_components
        self.random_state              = random_state
        self.device                    = device
        self.components_               = None
        self.explained_variance_ratio_ = None
        self.mean_                     = None

    def fit(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        self.mean_ = torch.mean(X, dim=0)
        X_centered = X - self.mean_
        _, s, Vt   = torch.linalg.svd(X_centered, full_matrices=False)
        if self.n_components is None:
            self.n_components = min(X.shape[0], X.shape[1])
        self.components_ = Vt[:self.n_components]
        explained_variance = (s ** 2) / (X.shape[0] - 1)
        total_variance     = torch.sum(explained_variance)
        self.explained_variance_ratio_ = (
            explained_variance[:self.n_components] / total_variance
            if total_variance > 0
            else torch.zeros(self.n_components, device=self.device)
        )
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)
        return torch.matmul(X - self.mean_, self.components_.T)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def compute_silhouette_score(X, labels, device='cpu'):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    X      = X.to(device)
    labels = labels.to(device)
    unique_labels = torch.unique(labels)
    if len(unique_labels) == 1:
        return 0.0
    silhouette_scores = []
    for i in range(X.shape[0]):
        same_mask   = labels == labels[i]
        same_points = X[same_mask]
        if same_points.shape[0] <= 1:
            silhouette_scores.append(0.0)
            continue
        dists = torch.norm(X[i].unsqueeze(0) - same_points, dim=1)
        mask  = dists > 0
        a     = torch.mean(dists[mask]) if mask.sum() > 0 else torch.tensor(0.0)
        b     = float('inf')
        for lab in unique_labels:
            if lab != labels[i]:
                others = X[labels == lab]
                if others.shape[0] > 0:
                    avg_d = torch.mean(torch.norm(X[i].unsqueeze(0) - others, dim=1))
                    b     = min(b, avg_d.item())
        if b == float('inf'):
            silhouette_scores.append(0.0)
        else:
            silhouette_scores.append((b - a.item()) / max(a.item(), b))
    return float(np.mean(silhouette_scores))


def bootstrap_resample(data, random_state=None, device='cpu'):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    data = data.to(device)
    if random_state is not None:
        torch.manual_seed(random_state)
    n   = data.shape[0]
    idx = torch.randint(0, n, (n,), device=device)
    return data[idx]


# ============================================================
# SECTION 2 — Configs
# ============================================================

@dataclass
class ROMEHyperParams:
    layers:                 List[int] = field(default_factory=lambda: [9, 10, 11])
    layer_selection:        str       = "top_k"
    target_module:          str       = "mlp"
    edit_weight:            float     = 1.0
    significance_threshold: float     = 2.0
    fact_token_strategy:    str       = "last"
    v_num_grad_steps:       int       = 20
    v_lr:                   float     = 5e-1
    v_loss_layer:           int       = -1
    v_weight_decay:         float     = 0.5
    clamp_norm_factor:      float     = 0.01
    window_size:            int       = 5

    def __post_init__(self):
        self.layers = sorted(self.layers)


@dataclass
class SignatureMiningConfig:
    # Paths
    activations_dir: Path = Path("outputs/activations")
    output_dir:      Path = Path("outputs/signatures")
    model_dir:       str  = "meta-llama/Meta-Llama-3.1-8B"
    prompts_file:    str  = "outputs/datasets/prompts.jsonl"

    rome_hparams: ROMEHyperParams = field(default_factory=ROMEHyperParams)

    top_k_directions:         int  = 3
    min_prompts_per_subject:  int  = 3
    use_semantic_negatives:   bool = True
    min_controls_per_subject: int  = 1
    allow_synthetic_fallback: bool = True

    positive_keys: List[str] = field(default_factory=lambda: ["direct", "contextual", "implicit", "reasoning", "misleading"])
    control_keys: List[str] = field(default_factory=lambda: ["control"])

    # Oversampling fields kept for API compatibility but disabled for mining
    # (FIX R3-1: oversampling corrupts direction learning).
    enable_oversampling:     bool = False   # kept False; field preserved for compat
    oversample_strategy:     str  = "max"
    oversample_separately:   bool = True
    preserve_original_ratio: bool = False

    # FIX R3-2: negative-pool policy fields
    # "match_positives" → target_negatives = len(pos_features) per layer (mining)
    # "fixed"           → target_negatives = fixed_negative_pool_size     (eval)
    negative_pool_mode:       str   = "match_positives"
    fixed_negative_pool_size: int   = 100
    synthetic_fraction:       float = 0.10  # max fraction of target that may be synthetic

    batch_size:         int  = 4
    device:             str  = "cuda" if torch.cuda.is_available() else "cpu"
    use_half_precision: bool = False

    activation_strategy: str           = "mean_token"
    token_pos:           int           = -1
    standardize_dims:    bool          = True
    target_dim:          Optional[int] = None

    enable_memory_cleanup: bool = True
    cleanup_frequency:     int  = 5

    n_bootstrap_samples: int = 100
    random_state:        int = 42

    def __post_init__(self):
        self.output_dir      = Path(self.output_dir)
        self.activations_dir = Path(self.activations_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.activations_dir.exists():
            raise FileNotFoundError(
                f"Activation directory not found: {self.activations_dir}"
            )
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "subject_data").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        if self.device == "cuda" and torch.cuda.is_available():
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"CUDA memory available: "
                f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )
        else:
            logger.info("Using CPU for computations")
        logger.info(
            f"Oversampling: {self.enable_oversampling} | "
            f"neg_pool_mode: {self.negative_pool_mode} | "
            f"synthetic_fraction: {self.synthetic_fraction}"
        )
        logger.info(f"ROME hyperparameters: {self.rome_hparams.__dict__}")


# ============================================================
# SECTION 3 — Memory / Activation
# ============================================================

class MemoryManager:
    def __init__(self, config: SignatureMiningConfig):
        self.config = config

    def get_gpu_memory_mb(self) -> float:
        return (torch.cuda.memory_allocated() / (1024 * 1024)
                if torch.cuda.is_available() else 0.0)

    def cleanup(self) -> None:
        if not self.config.enable_memory_cleanup:
            return
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.debug(f"Memory cleanup — GPU: {self.get_gpu_memory_mb():.0f} MB")


class ActivationManager:
    """Load and process activations; group prompts by subject."""

    def __init__(self, config: SignatureMiningConfig):
        self.config           = config
        self.memory_manager   = MemoryManager(config)
        self.activation_index = None
        self.prompts_data     = None
        self.target_dim       = None
        self.load_activation_index()
        self.load_prompts()
        self._detect_target_dimension()

    def load_activation_index(self) -> None:
        index_path = self.config.activations_dir / "activation_index.json"
        with open(index_path, 'r') as f:
            self.activation_index = json.load(f)
        logger.info(
            f"Loaded activation index: "
            f"{len(self.activation_index['prompts'])} prompts"
        )

    def load_prompts(self) -> None:
        prompts = []
        with open(self.config.prompts_file, 'r', encoding='utf-8') as f:
            for line in f:
                prompts.append(json.loads(line))
        self.prompts_data = {p["id"]: p for p in prompts}
        logger.info(
            f"Loaded {len(self.prompts_data)} prompts from "
            f"{self.config.prompts_file}"
        )

    @staticmethod
    def _iter_possible_fields(d: Dict[str, Any]) -> List[str]:
        vals = []
        for v in d.values():
            if v is None:
                continue
            if isinstance(v, str):
                vals.append(v.lower())
            elif isinstance(v, (int, float, bool)):
                vals.append(str(v).lower())
            elif isinstance(v, (list, tuple, set)):
                for x in v:
                    if isinstance(x, str):
                        vals.append(x.lower())
            elif isinstance(v, dict):
                for x in v.values():
                    if isinstance(x, str):
                        vals.append(x.lower())
        return vals

    def classify_prompt(self, prompt_data: Dict[str, Any]) -> str:
        vals = self._iter_possible_fields(prompt_data)
        for key in self.config.control_keys:
            if any(key in v for v in vals):
                return "control"
        for key in self.config.positive_keys:
            if any(key in v for v in vals):
                return "positive"
        text_hints = " ".join([
            prompt_data.get("prompt", ""),
            prompt_data.get("expected", "")
        ]).lower()
        if any(k in text_hints for k in self.config.control_keys):
            return "control"
        if any(k in text_hints for k in self.config.positive_keys):
            return "positive"
        return "unknown"

    def load_activation_file(self, path: str) -> Optional[np.ndarray]:
        try:
            act = compress_pickle.load(path)
            if isinstance(act, np.ndarray) and act.dtype != np.float32:
                act = act.astype(np.float32)
            return act
        except Exception as e:
            logger.error(f"Failed to load activation from {path}: {e}")
            return None

    def _standardize_dimension(
        self, activation: np.ndarray, target_dim: int
    ) -> np.ndarray:
        cur = activation.shape[0]
        if cur == target_dim:
            return activation
        if cur > target_dim:
            return activation[:target_dim]
        padded = np.zeros(target_dim, dtype=activation.dtype)
        padded[:cur] = activation
        return padded

    def _process_single_activation(
        self, activation: np.ndarray
    ) -> Optional[np.ndarray]:
        if activation is None:
            return None
        try:
            if activation.ndim == 1:
                processed = activation
            elif activation.ndim == 2:
                strat = self.config.activation_strategy
                if strat == "mean_token":
                    processed = np.mean(activation, axis=0)
                elif strat == "specific_token":
                    pos = self.config.token_pos
                    if pos < 0:
                        pos = activation.shape[0] + pos
                    processed = activation[
                        np.clip(pos, 0, activation.shape[0] - 1)
                    ]
                elif strat == "flatten_mean":
                    processed = np.mean(activation, axis=0)
                else:
                    processed = activation[-1]
            elif activation.ndim == 3:
                processed = self._process_single_activation(activation[0])
            else:
                processed = self._process_single_activation(
                    activation.reshape(-1, activation.shape[-1])
                )
            if processed.ndim > 1:
                processed = processed.flatten()
            if self.config.standardize_dims and self.target_dim is not None:
                processed = self._standardize_dimension(processed, self.target_dim)
            return processed.astype(np.float32)
        except Exception as e:
            logger.warning(f"Error in _process_single_activation: {e}, returning None")
            return None

    def _detect_target_dimension(self) -> None:
        if not self.config.standardize_dims:
            return
        logger.info("Auto-detecting target activation dimension...")
        sample_dims, count = [], 0
        for pinfo in self.activation_index["prompts"].values():
            if count >= 10:
                break
            for path in pinfo["paths"][:1]:
                try:
                    act = self.load_activation_file(path)
                    if act is not None:
                        proc = self._process_single_activation(act)
                        if proc is not None:
                            sample_dims.append(proc.shape[-1])
                            count += 1
                            break
                except Exception:
                    continue
        if sample_dims:
            from collections import Counter
            self.target_dim = Counter(sample_dims).most_common(1)[0][0]
            logger.info(f"Auto-detected target dimension: {self.target_dim}")
        else:
            logger.warning("Could not auto-detect target dimension")

    def group_by_subject(self) -> Dict[str, List[Dict]]:
        """
        Group prompts by subject.
        'category' stored normalised (strip+lower) for C_Analysis compatibility.
        Oversampling applied only when config.enable_oversampling=True
        (kept False for mining per FIX R3-1).
        """
        groups = defaultdict(list)
        for prompt_id, pinfo in self.activation_index["prompts"].items():
            if prompt_id not in self.prompts_data:
                continue
            pdata   = self.prompts_data[prompt_id]
            subject = pdata.get("subject", "")
            if not subject:
                continue
            label    = self.classify_prompt(pdata)
            raw_cat  = pdata.get("category", "")
            category = raw_cat.strip().lower() if raw_cat else ""
            groups[subject].append({
                "prompt_id": prompt_id,
                "paths":     pinfo["paths"],
                "triple_id": pdata.get("triple_id", ""),
                "prompt":    pdata.get("prompt", ""),
                "expected":  pdata.get("expected", ""),
                "class":     label,
                "category":  category,
            })

        filtered = {
            subj: plist for subj, plist in groups.items()
            if len([x for x in plist if x["class"] in ("positive", "unknown")])
               >= self.config.min_prompts_per_subject
        }
        logger.info(
            f"Grouped prompts into {len(filtered)} subject groups (after filtering)"
        )

        if self.config.enable_oversampling:
            filtered = self._apply_oversampling(filtered)

        return filtered

    def _apply_oversampling(
        self, subject_groups: Dict[str, List[Dict]]
    ) -> Dict[str, List[Dict]]:
        """
        Kept for API compatibility (e.g. if enable_oversampling is True for
        experiments), but not called during normal mining.
        """
        if not subject_groups:
            return subject_groups
        all_sizes = [len(p) for p in subject_groups.values()]
        if self.config.oversample_strategy == "max":
            target_size = max(all_sizes)
        elif self.config.oversample_strategy == "median":
            target_size = int(np.median(all_sizes))
        else:
            try:
                target_size = int(self.config.oversample_strategy)
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid oversample_strategy "
                    f"'{self.config.oversample_strategy}', using 'max'"
                )
                target_size = max(all_sizes)

        logger.info(
            f"Oversampling subjects to target size: {target_size}  "
            f"(original: min={min(all_sizes)}, max={max(all_sizes)}, "
            f"median={np.median(all_sizes):.1f})"
        )
        balanced_groups = {}
        np.random.seed(self.config.random_state)

        for subject, prompts in subject_groups.items():
            if len(prompts) >= target_size:
                balanced_groups[subject] = prompts
                continue

            if self.config.oversample_separately:
                positives = [p for p in prompts if p["class"] in ("positive", "unknown")]
                controls  = [p for p in prompts if p["class"] == "control"]
                if self.config.preserve_original_ratio and positives and controls:
                    ratio            = len(positives) / len(prompts)
                    target_positives = int(target_size * ratio)
                    target_controls  = target_size - target_positives
                else:
                    if positives and controls:
                        target_positives = target_size // 2
                        target_controls  = target_size - target_positives
                    elif positives:
                        target_positives, target_controls = target_size, 0
                    else:
                        target_positives, target_controls = 0, target_size

                oversampled_positives = []
                if positives and target_positives > 0:
                    idx = np.random.choice(
                        len(positives), size=target_positives, replace=True
                    )
                    oversampled_positives = [positives[i].copy() for i in idx]
                    for i, item in enumerate(oversampled_positives):
                        if i >= len(positives):
                            item["oversampled"] = True

                oversampled_controls = []
                if controls and target_controls > 0:
                    idx = np.random.choice(
                        len(controls), size=target_controls, replace=True
                    )
                    oversampled_controls = [controls[i].copy() for i in idx]
                    for i, item in enumerate(oversampled_controls):
                        if i >= len(controls):
                            item["oversampled"] = True

                balanced_prompts = oversampled_positives + oversampled_controls
            else:
                idx = np.random.choice(len(prompts), size=target_size, replace=True)
                balanced_prompts = [prompts[i].copy() for i in idx]
                for i, item in enumerate(balanced_prompts):
                    if i >= len(prompts):
                        item["oversampled"] = True

            np.random.shuffle(balanced_prompts)
            balanced_groups[subject] = balanced_prompts

        logger.info("Oversampling complete")
        return balanced_groups


# ============================================================
# SECTION 4 — Negative-pool helpers  (FIX R3-4, R3-5, P4)
# ============================================================

def _resolve_negative_target(n_pos: int, mode: str, fixed_size: int) -> int:
    """
    FIX R3-4: decide how many negatives to target for one layer.

    mode="match_positives" → target = n_pos   (mining: subject-proportional)
    mode="fixed"           → target = fixed_size (evaluation: comparable rows)
    """
    if mode == "match_positives":
        return n_pos
    return fixed_size


def _get_cross_subject_negatives(
    exclude_subject: str,
    all_groups:      dict,
    layer:           int,
    tracer,
    n:               Optional[int] = None,
    seed:            int = 42,
) -> List[np.ndarray]:
    """
    FIX R3-3 + P4: pool ALL non-control prompts from every other subject,
    shuffle subject order first (removes order bias), subsample n prompt
    dicts, then load activations only for those sampled dicts.
    """
    rng = np.random.default_rng(seed)
    candidates: List[Dict] = []
    other_subjects = [s for s in all_groups if s != exclude_subject]
    rng.shuffle(other_subjects)  # FIX P4: remove sequential order bias

    for subj in other_subjects:
        candidates.extend([
            p for p in all_groups[subj]
            if p["class"] in ("positive", "unknown")
        ])

    if not candidates:
        return []

    if n is not None and len(candidates) > n:
        idx             = rng.choice(len(candidates), n, replace=False)
        sampled_prompts = [candidates[i] for i in idx]
    else:
        sampled_prompts = candidates

    feats, _ = tracer.load_and_process_activations(sampled_prompts, layer)
    return feats


def _build_negative_pool(
    subject:            str,
    initial_negs:       List[np.ndarray],
    pos_features:       List[np.ndarray],
    all_groups:         dict,
    layer:              int,
    tracer,
    target_negatives:   int,
    synthetic_fraction: float,
    seed:               int,
) -> Tuple[List[np.ndarray], Dict[str, int]]:
    """
    FIX R3-5: unified negative-pool builder with explicit synthetic budget.

    Priority:
      1. initial_negs — real same-subject controls (always kept)
      2. cross-subject — real non-control prompts from other subjects
      3. synthetic — generated, capped at synthetic_fraction * target

    Returns (neg_features, composition) where composition tracks how many
    negatives came from each source for logging.
    """
    neg_features = list(initial_negs)   # always preserve real controls
    n_control    = len(neg_features)

    desired_synth = int(round(target_negatives * synthetic_fraction))
    desired_synth = min(desired_synth, max(0, target_negatives - n_control))
    desired_cross = max(0, target_negatives - n_control - desired_synth)

    cross_neg: List[np.ndarray] = []
    if desired_cross > 0:
        cross_neg = _get_cross_subject_negatives(
            exclude_subject=subject,
            all_groups=all_groups,
            layer=layer,
            tracer=tracer,
            n=desired_cross,
            seed=seed,
        )
        neg_features.extend(cross_neg)

    synth: List[np.ndarray] = []
    synth_needed = max(desired_synth, target_negatives - len(neg_features))
    if synth_needed > 0:
        synth = tracer.generate_synthetic_negatives(
            pos_features, num_negatives=synth_needed
        )
        neg_features.extend(synth)

    # Safety backfill for edge cases
    if len(neg_features) < target_negatives:
        extra = tracer.generate_synthetic_negatives(
            pos_features,
            num_negatives=target_negatives - len(neg_features)
        )
        neg_features.extend(extra)
        synth.extend(extra)

    composition = {
        "control":       n_control,
        "cross_subject": len(cross_neg),
        "synthetic":     len(synth),
    }
    return neg_features, composition


# ============================================================
# SECTION 5 — CausalTracer
# ============================================================

class CausalTracer:
    """Find signature directions contrasting positive vs negative activations."""

    def __init__(
        self,
        config: SignatureMiningConfig,
        activation_manager: ActivationManager
    ):
        self.config             = config
        self.activation_manager = activation_manager
        self.memory_manager     = MemoryManager(config)
        self.device             = torch.device(config.device)

    def _select_paths_for_layer(self, prompt: Dict, layer: int) -> Optional[str]:
        """
        FIX P5: flexible path selection — handles layer_, layer-, layer.
        separators and zero-padded layer numbers.
        Two-pass: strict (with module_tag) then relaxed.
        """
        module_tag = self.config.rome_hparams.target_module
        layer_pat  = re.compile(rf"layer[_\-]?0*{layer}[_\-\.]", re.IGNORECASE)
        for strict in (True, False):
            for path in prompt["paths"]:
                p = path.replace("\\", "/")
                if layer_pat.search(p) and (not strict or module_tag in p):
                    return path
        return None

    def load_and_process_activations(
        self, prompt_group: List[Dict], layer: int
    ) -> Tuple[List[np.ndarray], List[str]]:
        processed, failures = [], []
        for prompt in prompt_group:
            layer_path = self._select_paths_for_layer(prompt, layer)
            if not layer_path:
                failures.append(f"No path found for layer {layer}")
                continue
            raw = self.activation_manager.load_activation_file(layer_path)
            if raw is None:
                failures.append(f"Failed to load {layer_path}")
                continue
            proc = self.activation_manager._process_single_activation(raw)
            if proc is not None:
                processed.append(proc)
            else:
                failures.append(f"Failed to process {layer_path}")
        if failures:
            logger.warning(
                f"Failed to process {len(failures)} activations for layer {layer}"
            )
        return processed, failures

    def generate_synthetic_negatives(
        self,
        positive_features: List[np.ndarray],
        num_negatives: int = None
    ) -> List[np.ndarray]:
        """Fallback when real negatives are insufficient — uses GPU."""
        if not positive_features:
            return []
        if num_negatives is None:
            num_negatives = len(positive_features)
        try:
            pos_stack     = torch.from_numpy(
                np.vstack(positive_features)
            ).float().to(self.device)
            feature_std   = torch.std(pos_stack, dim=0)
            mean_features = torch.mean(pos_stack, dim=0)
            neg_features  = []
            half = max(1, num_negatives // 2)
            for _ in range(half):
                noise = torch.randn_like(mean_features) * feature_std
                neg_features.append(
                    (mean_features - 2 * noise).cpu().numpy().astype(np.float32)
                )
            for i in range(num_negatives - len(neg_features)):
                base = positive_features[i % len(positive_features)].copy()
                np.random.shuffle(base)
                neg_features.append(base.astype(np.float32))
            return neg_features
        except Exception as e:
            logger.warning(f"Synthetic negatives generation failed: {e}")
            neg = []
            for i in range(num_negatives):
                base  = positive_features[i % len(positive_features)]
                noise = np.random.normal(0, 0.1, base.shape)
                neg.append((base + noise).astype(np.float32))
            return neg

    def compute_signature_directions(
        self,
        positive_features: List[np.ndarray],
        negative_features: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Learn mean-difference direction in standardised space.

        FIX 1:  scaler mean/scale stored in 'preproc' so downstream
                modules (Module D projection, Module E hooks) can
                re-apply the same standardisation before projecting.
        FIX R3-9: direction sign-orientation — flip if
                pos_proj.mean() < neg_proj.mean() so stored direction
                always points toward the positive (subject) class.
        """
        if not positive_features or not negative_features:
            return {"directions": [], "scores": [], "stats": {}, "preproc": {}}
        try:
            # Dimension alignment
            for feat_list in (positive_features, negative_features):
                dims = [f.shape[0] for f in feat_list]
                if len(set(dims)) > 1:
                    min_d = min(dims)
                    feat_list[:] = [f[:min_d] for f in feat_list]
            min_d = min(
                positive_features[0].shape[0], negative_features[0].shape[0]
            )
            positive_features = [f[:min_d] for f in positive_features]
            negative_features = [f[:min_d] for f in negative_features]

            pos_stack = torch.from_numpy(
                np.vstack(positive_features)
            ).float().to(self.device)
            neg_stack = torch.from_numpy(
                np.vstack(negative_features)
            ).float().to(self.device)

            scaler   = StandardScaler(device=self.device)
            combined = torch.cat([pos_stack, neg_stack], dim=0)
            scaler.fit(combined)
            pos_scaled = scaler.transform(pos_stack)
            neg_scaled = scaler.transform(neg_stack)

            # FIX 1: persist scaler params for downstream projection
            scaler_mean_np  = scaler.mean_.detach().cpu().numpy().astype(np.float32)
            scaler_scale_np = scaler.scale_.detach().cpu().numpy().astype(np.float32)

            pos_mean = torch.mean(pos_scaled, dim=0)
            neg_mean = torch.mean(neg_scaled, dim=0)
            diff_vec = pos_mean - neg_mean
            norm     = torch.norm(diff_vec)
            primary  = diff_vec / norm if norm > 0 else diff_vec

            pos_proj = torch.matmul(pos_scaled, primary)
            neg_proj = torch.matmul(neg_scaled, primary)

            # FIX R3-9: ensure direction points toward positive class
            if float(torch.mean(pos_proj).item()) < float(torch.mean(neg_proj).item()):
                primary  = -primary
                pos_proj = -pos_proj
                neg_proj = -neg_proj

            pos_mean_proj = float(torch.mean(pos_proj).item())
            neg_mean_proj = float(torch.mean(neg_proj).item())
            pooled_std    = float(torch.sqrt(
                (torch.var(pos_proj, unbiased=True)
                 + torch.var(neg_proj, unbiased=True)) / 2
            ).item())
            effect_size = abs(pos_mean_proj - neg_mean_proj) / (pooled_std + 1e-6)

            effect_samples = []
            for i in range(min(self.config.n_bootstrap_samples, 50)):
                ps  = bootstrap_resample(
                    pos_proj, self.config.random_state + i, self.device
                )
                ns  = bootstrap_resample(
                    neg_proj, self.config.random_state + i + 1000, self.device
                )
                pm  = float(torch.mean(ps).item())
                nm  = float(torch.mean(ns).item())
                psd = float(torch.sqrt(
                    (torch.var(ps, unbiased=True)
                     + torch.var(ns, unbiased=True)) / 2
                ).item())
                effect_samples.append(abs(pm - nm) / (psd + 1e-6))
            effect_samples = np.asarray(effect_samples)

            directions = [primary.cpu().numpy()]
            scores     = [float(effect_size)]

            if self.config.top_k_directions > 1:
                try:
                    all_scaled = torch.cat([pos_scaled, neg_scaled], dim=0)
                    proj_vals  = torch.matmul(all_scaled, primary)
                    residuals  = all_scaled - torch.outer(proj_vals, primary)
                    n_extra    = min(
                        self.config.top_k_directions - 1,
                        max(1, min(pos_scaled.shape[0], neg_scaled.shape[0]) - 1)
                    )
                    pca = PCA(
                        n_components=n_extra,
                        random_state=self.config.random_state,
                        device=self.device
                    )
                    pca.fit(residuals)
                    for comp in pca.components_:
                        comp   = comp / (torch.norm(comp) + 1e-12)
                        pos_c  = torch.matmul(pos_scaled, comp)
                        neg_c  = torch.matmul(neg_scaled, comp)
                        pooled = float(torch.sqrt(
                            (torch.var(pos_c, unbiased=True)
                             + torch.var(neg_c, unbiased=True)) / 2
                        ).item())
                        eff = (
                            abs(torch.mean(pos_c).item() - torch.mean(neg_c).item())
                            / (pooled + 1e-6)
                        )
                        if eff >= self.config.rome_hparams.significance_threshold:
                            directions.append(comp.cpu().numpy())
                            scores.append(float(eff))
                except Exception as e:
                    logger.warning(f"Secondary PCA directions failed: {e}")

            stats = {
                "pos_mean":       pos_mean_proj,
                "neg_mean":       neg_mean_proj,
                "pos_std":        float(torch.std(pos_proj).item()),
                "neg_std":        float(torch.std(neg_proj).item()),
                "effect_size":    float(effect_size),
                "effect_ci_low":  float(np.percentile(effect_samples, 2.5)),
                "effect_ci_high": float(np.percentile(effect_samples, 97.5)),
                "pos_count":      len(positive_features),
                "neg_count":      len(negative_features),
                "feature_dim":    int(diff_vec.shape[0])
            }
            return {
                "directions": directions[:self.config.top_k_directions],
                "scores":     scores[:self.config.top_k_directions],
                "stats":      stats,
                "preproc": {                      # FIX 1
                    "scaler_mean":  scaler_mean_np.tolist(),
                    "scaler_scale": scaler_scale_np.tolist(),
                }
            }
        except Exception as e:
            logger.error(f"Error in compute_signature_directions: {e}")
            return {"directions": [], "scores": [], "stats": {}, "preproc": {}}

    def generate_pca_visualization(
        self,
        pos_features: List[np.ndarray],
        neg_features: List[np.ndarray],
        subject: str,
        layer: int
    ) -> Optional[str]:
        if len(pos_features) < 3 or len(neg_features) < 3:
            return None
        try:
            all_dims = [f.shape[0] for f in pos_features + neg_features]
            if len(set(all_dims)) > 1:
                min_dim      = min(all_dims)
                pos_features = [f[:min_dim] for f in pos_features]
                neg_features = [f[:min_dim] for f in neg_features]

            pos_stack = torch.from_numpy(
                np.vstack(pos_features)
            ).float().to(self.device)
            neg_stack = torch.from_numpy(
                np.vstack(neg_features)
            ).float().to(self.device)
            all_data  = torch.cat([pos_stack, neg_stack], dim=0)
            labels    = np.array([1] * len(pos_features) + [0] * len(neg_features))

            scaler     = StandardScaler(device=self.device)
            all_scaled = scaler.fit_transform(all_data)
            emb        = PCA(
                n_components=2,
                random_state=self.config.random_state,
                device=self.device
            ).fit_transform(all_scaled).cpu().numpy()

            plt.figure(figsize=(10, 8))
            sc = plt.scatter(
                emb[:, 0], emb[:, 1],
                c=labels, cmap='coolwarm', alpha=0.8, s=100
            )
            plt.colorbar(sc, label='Class (1=Subject, 0=Negative pool)')
            plt.title(f'PCA Visualization: {subject} (Layer {layer})')
            plt.xlabel('PC1'); plt.ylabel('PC2')
            try:
                sil = compute_silhouette_score(
                    all_scaled.cpu().numpy(), labels, device=self.device
                )
                plt.annotate(
                    f'Silhouette: {sil:.3f}',
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="white", ec="gray", alpha=0.8)
                )
            except Exception:
                pass
            viz_path = (
                self.config.output_dir / "visualizations"
                / f"{subject.replace(' ', '_')}_layer{layer}_pca.png"
            )
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            return str(viz_path)
        except Exception as e:
            logger.error(f"Failed to generate PCA visualization: {e}")
            return None

    def analyze_subject(
        self,
        subject:              str,
        prompt_group:         List[Dict],
        cross_subject_groups: Optional[Dict] = None  # FIX P3
    ) -> Dict[str, Any]:
        """
        Mine signature directions across all available layers.

        Negative pool (FIX R3-6: matched-size, bounded synthetic):
          target_negatives = len(pos_features)  [mode="match_positives"]
          pool via _build_negative_pool():
            1. Same-subject controls (real; auxiliary falsification)
            2. Cross-subject positives from all other subjects (entity-discriminative)
            3. Synthetic fill capped at synthetic_fraction * target

        FIX P5:  layer discovery uses flexible regex.
        FIX 1:   preproc stored in every layer result.
        FIX R3-1: no oversampling — positive pool is all real non-control prompts.
        """
        # FIX P5: flexible layer-discovery regex
        layer_disc_pat = re.compile(r"layer[_\-]?0*(\d+)", re.IGNORECASE)

        original_count   = len([p for p in prompt_group if not p.get("oversampled", False)])
        oversampled_count = len([p for p in prompt_group if p.get("oversampled", False)])

        logger.info(
            f"Analyzing subject: {subject} with {len(prompt_group)} prompts "
            f"({original_count} original, {oversampled_count} oversampled)"
        )

        # Discover available layers
        available_layers: set = set()
        for p in prompt_group:
            for path in p["paths"]:
                m = layer_disc_pat.search(path.replace("\\", "/"))  # FIX P5
                if m:
                    available_layers.add(int(m.group(1)))

        layer_selection = self.config.rome_hparams.layer_selection
        layers_cfg      = self.config.rome_hparams.layers

        if layer_selection == "all":
            layers = sorted(available_layers)
        elif layers_cfg:
            layers = [l for l in layers_cfg if l in available_layers]
        else:
            layers = sorted(available_layers)

        if not layers:
            logger.warning(f"No valid layers found for subject {subject}")
            return {
                "subject": subject, "layers": {},
                "summary": {"status": "no_layers"}
            }

        positives_all = [
            p for p in prompt_group if p["class"] in ("positive", "unknown")
        ]
        controls_all  = [p for p in prompt_group if p["class"] == "control"]

        if len(positives_all) < self.config.min_prompts_per_subject:
            logger.warning(
                f"Subject {subject}: insufficient positives "
                f"({len(positives_all)})"
            )
            return {
                "subject": subject, "layers": {},
                "summary": {"status": "too_few_positives"}
            }

        results = {
            "subject":          subject,
            "layers":           {},
            "summary":          {},
            "visualizations":   {},
            "oversampling_info": {
                "original":   original_count,
                "oversampled": oversampled_count
            }
        }

        for layer in layers:
            try:
                pos_features, _ = self.load_and_process_activations(
                    positives_all, layer
                )
                if len(pos_features) < max(3, self.config.min_prompts_per_subject):
                    logger.warning(
                        f"Not enough positive examples for layer {layer} "
                        f"(got {len(pos_features)})"
                    )
                    continue

                # --- Negative pool (FIX R3-6) ---
                raw_neg_features: List[np.ndarray] = []
                if (self.config.use_semantic_negatives
                        and len(controls_all) >= self.config.min_controls_per_subject):
                    raw_neg_features, _ = self.load_and_process_activations(
                        controls_all, layer
                    )

                target_negatives = _resolve_negative_target(
                    n_pos=len(pos_features),
                    mode=self.config.negative_pool_mode,
                    fixed_size=self.config.fixed_negative_pool_size,
                )

                neg_features, neg_composition = _build_negative_pool(
                    subject=subject,
                    initial_negs=raw_neg_features,
                    pos_features=pos_features,
                    all_groups=(
                        cross_subject_groups
                        if cross_subject_groups is not None else {}
                    ),
                    layer=layer,
                    tracer=self,
                    target_negatives=target_negatives,
                    synthetic_fraction=self.config.synthetic_fraction,
                    seed=self.config.random_state,
                )

                logger.info(
                    f"Layer {layer}: n_pos={len(pos_features)} | "
                    f"n_neg={len(neg_features)} "
                    f"(control={neg_composition['control']}, "
                    f"cross_subject={neg_composition['cross_subject']}, "
                    f"synthetic={neg_composition['synthetic']})"
                )

                if len(neg_features) < 2:
                    logger.warning(
                        f"Layer {layer}: insufficient negatives "
                        f"({len(neg_features)})"
                    )
                    continue

                viz_path = self.generate_pca_visualization(
                    pos_features, neg_features, subject, layer
                )
                if viz_path:
                    results["visualizations"][str(layer)] = viz_path

                sig = self.compute_signature_directions(pos_features, neg_features)

                # Store per-layer result.
                # 'preproc' is new (FIX 1/P6); older Module D ignores unknown keys.
                results["layers"][str(layer)] = {
                    "directions": [
                        v.tolist() if isinstance(v, np.ndarray) else v
                        for v in sig["directions"]
                    ],
                    "scores":           sig["scores"],
                    "stats":            sig["stats"],
                    "preproc":          sig.get("preproc", {}),   # FIX 1
                    "positive_count":   len(pos_features),
                    "negative_count":   len(neg_features),
                    "neg_counts": {                                # FIX C
                        "control":       neg_composition["control"],
                        "cross_subject": neg_composition["cross_subject"],
                        "synthetic":     neg_composition["synthetic"],
                    },
                    "failed_activations": 0
                }
                del pos_features, neg_features

            except Exception as e:
                logger.error(
                    f"Error processing layer {layer} for subject {subject}: {e}"
                )

        if results["layers"]:
            best_layer, best_data = max(
                results["layers"].items(),
                key=lambda kv: kv[1]["scores"][0] if kv[1]["scores"] else 0
            )
            results["summary"] = {
                # --- Module D / E compatible key names ---
                "best_layer":  best_layer,
                "best_score":  best_data["scores"][0] if best_data["scores"] else 0.0,
                "status":      "success",
                "layer_count": len(results["layers"]),
                "visualizations": len(results.get("visualizations", {}))
            }
        else:
            results["summary"] = {"status": "no_data"}

        return results

    def plot_signature_distributions(
        self, subject: str, subject_results: Dict
    ) -> Optional[str]:
        if subject_results.get("summary", {}).get("status") != "success":
            return None
        try:
            best_layer = subject_results["summary"]["best_layer"]
            layer_data = subject_results["layers"][best_layer]

            pm  = layer_data["stats"]["pos_mean"]
            nm  = layer_data["stats"]["neg_mean"]
            ps  = max(1e-6, layer_data["stats"]["pos_std"])
            ns  = max(1e-6, layer_data["stats"]["neg_std"])
            x   = np.linspace(
                min(pm - 3*ps, nm - 3*ns),
                max(pm + 3*ps, nm + 3*ns), 1000
            )
            pd_ = (1 / (ps * np.sqrt(2*np.pi))
                   * np.exp(-(x - pm)**2 / (2*ps**2)))
            nd_ = (1 / (ns * np.sqrt(2*np.pi))
                   * np.exp(-(x - nm)**2 / (2*ns**2)))

            plt.figure(figsize=(10, 6))
            plt.plot(x, pd_, label='Subject (Leak) Distribution')
            plt.plot(x, nd_, label='Negative Distribution')
            plt.axvline(x=pm, linestyle='--', alpha=0.5)
            plt.axvline(x=nm, linestyle='--', alpha=0.5)

            eff   = layer_data["stats"]["effect_size"]
            lo    = layer_data["stats"].get("effect_ci_low",  eff * 0.9)
            hi    = layer_data["stats"].get("effect_ci_high", eff * 1.1)
            title = f'Signature Distribution for "{subject}" (Layer {best_layer})'
            if "oversampling_info" in subject_results:
                info  = subject_results["oversampling_info"]
                title += (
                    f'\n({info["original"]} original, '
                    f'{info["oversampled"]} oversampled)'
                )
            plt.title(title)
            plt.xlabel('Projection Value')
            plt.ylabel('Density')
            plt.annotate(
                f'Effect Size: {eff:.2f} [{lo:.2f}, {hi:.2f}]',
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="white", ec="gray", alpha=0.8)
            )
            plt.legend()
            plt.tight_layout()

            plot_path = (
                self.config.output_dir / "plots"
                / f"{subject.replace(' ', '_')}_layer{best_layer}.png"
            )
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        except Exception as e:
            logger.error(f"Failed to create plot for {subject}: {e}")
            return None


# ============================================================
# SECTION 6 — SignatureExtractor (Orchestration)
# ============================================================

class SignatureExtractor:
    def __init__(self, config: SignatureMiningConfig):
        self.config             = config
        self.memory_manager     = MemoryManager(config)
        self.activation_manager = ActivationManager(config)
        self.causal_tracer      = CausalTracer(config, self.activation_manager)
        self.processing_stats   = {
            "successful_subjects":    0,
            "failed_subjects":        0,
            "total_signatures":       0,
            "visualizations_created": 0,
            "start_time":             time.time()
        }
        self.subjects_processed = 0

    def extract_all_signatures(self) -> Dict[str, Any]:
        subject_groups = self.activation_manager.group_by_subject()
        all_signatures = {}

        for subject, prompt_group in tqdm(
            subject_groups.items(), desc="Extracting signatures"
        ):
            try:
                # FIX P3: pass all groups for cross-subject negatives during mining
                subject_results = self.causal_tracer.analyze_subject(
                    subject, prompt_group,
                    cross_subject_groups=subject_groups
                )

                plot_path = None
                if subject_results["summary"].get("status") == "success":
                    plot_path = self.causal_tracer.plot_signature_distributions(
                        subject, subject_results
                    )
                    self.processing_stats["successful_subjects"] += 1
                    for ld in subject_results["layers"].values():
                        self.processing_stats["total_signatures"] += len(
                            ld.get("directions", [])
                        )
                    self.processing_stats["visualizations_created"] += len(
                        subject_results.get("visualizations", {})
                    )
                else:
                    self.processing_stats["failed_subjects"] += 1

                if plot_path:
                    subject_results["summary"]["plot_path"] = plot_path

                # Write per-subject JSON (same schema as before)
                subject_file = (
                    self.config.output_dir / "subject_data"
                    / f"{subject.replace(' ', '_')}.json"
                )
                with open(subject_file, 'w', encoding='utf-8') as f:
                    json.dump(subject_results, f, indent=2)

                all_signatures[subject] = subject_results

                self.subjects_processed += 1
                if (self.config.enable_memory_cleanup
                        and self.subjects_processed
                        % self.config.cleanup_frequency == 0):
                    self.memory_manager.cleanup()
                    logger.info(
                        f"Processed {self.subjects_processed} subjects, "
                        f"GPU memory: {self.memory_manager.get_gpu_memory_mb():.0f} MB"
                    )

            except Exception as e:
                logger.error(f"Failed to process subject {subject}: {e}")
                self.processing_stats["failed_subjects"] += 1

        return all_signatures

    def save_signature_index(self, signatures: Dict[str, Any]) -> None:
        """
        Write signature_index.json and top_signatures.pkl.gz.

        Index schema: unchanged from original (Module D/E read 'best_layer',
        'best_score', 'status').

        top_signatures.pkl.gz per-subject keys:
          'best_layer'   — layer index (Module D uses this)
          'effect_size'  — mining Cohen's d proxy (Module D uses this)
          'signatures'   — list of direction vectors (Module D uses this)
          'preproc'      — scaler params (FIX P6; ignored by older Module D)
        """
        index = {
            "config": {
                "rome_layers":           self.config.rome_hparams.layers,
                "layer_selection":       self.config.rome_hparams.layer_selection,
                "top_k_directions":      self.config.top_k_directions,
                "model_dir":             self.config.model_dir,
                "random_state":          self.config.random_state,
                "activation_strategy":   self.config.activation_strategy,
                "standardize_dims":      self.config.standardize_dims,
                "target_dim":            self.activation_manager.target_dim,
                "device":                self.config.device,
                "oversampling_enabled":  self.config.enable_oversampling,
                "oversample_strategy":   self.config.oversample_strategy,
                "negative_pool_mode":    self.config.negative_pool_mode,
                "synthetic_fraction":    self.config.synthetic_fraction,
            },
            "subjects": {},
            "stats": {
                "successful_subjects":    self.processing_stats["successful_subjects"],
                "failed_subjects":        self.processing_stats["failed_subjects"],
                "total_signatures":       self.processing_stats["total_signatures"],
                "visualizations_created": self.processing_stats["visualizations_created"],
                "processing_time":        time.time() - self.processing_stats["start_time"]
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        for subject, subject_data in signatures.items():
            if "summary" in subject_data:
                summary = subject_data["summary"].copy()
                if "oversampling_info" in subject_data:
                    summary["oversampling_info"] = subject_data["oversampling_info"]
                # Record which layers were actually mined
                actual_layers = sorted(
                    int(k) for k in subject_data.get("layers", {}).keys()
                )
                summary["layer_selection"]     = self.config.rome_hparams.layer_selection
                summary["actual_layers_mined"] = actual_layers
                index["subjects"][subject]     = summary

        index_path = self.config.output_dir / "signature_index.json"
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)
        logger.info(f"Saved signature index to {index_path}")

        # Build top_signatures with Module D / E compatible key names
        top_signatures = {}
        for subject, data in signatures.items():
            if data.get("summary", {}).get("status") == "success":
                best_layer = data["summary"].get("best_layer")   # compat key
                if best_layer is not None:
                    layer_entry = data["layers"].get(str(best_layer), {})
                    dirs        = layer_entry.get("directions", [])
                    if dirs and len(dirs[0]) > 0:
                        top_signatures[subject] = {
                            # --- Keys read by Module D ---
                            "best_layer":  best_layer,
                            "effect_size": data["summary"].get("best_score"),
                            "signatures":  dirs[:1],
                            # --- FIX P6: preproc for correct projection space ---
                            "preproc":     layer_entry.get("preproc", {}),
                        }
                    else:
                        logger.warning(
                            f"Skipping {subject}: no valid signature directions "
                            f"for layer {best_layer}"
                        )
                else:
                    logger.warning(f"Skipping {subject}: best_layer is None")

        if not top_signatures:
            logger.error("No valid signatures generated!")
            raise ValueError("No valid signatures generated for any subject")

        top_path = self.config.output_dir / "top_signatures.pkl.gz"
        compress_pickle.dump(top_signatures, top_path, compression="gzip")
        logger.info(
            f"Saved {len(top_signatures)} top signatures to {top_path}"
        )

    def create_summary_report(self) -> None:
        elapsed = time.time() - self.processing_stats["start_time"]
        report  = {
            "title":     "KIF Module C: Signature Mining Summary",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model":     self.config.model_dir,
            "hardware": {
                "device":         self.config.device,
                "cuda_available": torch.cuda.is_available(),
                "device_name": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else "CPU"
                ),
                "cuda_version": (
                    torch.version.cuda
                    if torch.cuda.is_available() else "N/A"
                )
            },
            "config": {
                "rome_layers":           self.config.rome_hparams.layers,
                "layer_selection":       self.config.rome_hparams.layer_selection,
                "top_k_directions":      self.config.top_k_directions,
                "significance_threshold":
                    self.config.rome_hparams.significance_threshold,
                "activation_strategy":   self.config.activation_strategy,
                "standardize_dims":      self.config.standardize_dims,
                "target_dim":            self.activation_manager.target_dim,
                "use_half_precision":    self.config.use_half_precision,
                "oversampling": {
                    "enabled":          self.config.enable_oversampling,
                    "strategy":         self.config.oversample_strategy,
                    "separate_classes": self.config.oversample_separately,
                    "preserve_ratio":   self.config.preserve_original_ratio
                },
                "negative_pool": {
                    "mode":              self.config.negative_pool_mode,
                    "fixed_size":        self.config.fixed_negative_pool_size,
                    "synthetic_fraction":self.config.synthetic_fraction
                },
                "note": (
                    "layer_selection='all' mines ALL available activation layers. "
                    "Per-subject actual_layers_mined in signature_index.json."
                )
            },
            "results": {
                "total_subjects_analyzed": (
                    self.processing_stats["successful_subjects"]
                    + self.processing_stats["failed_subjects"]
                ),
                "successful_subjects":     self.processing_stats["successful_subjects"],
                "failed_subjects":         self.processing_stats["failed_subjects"],
                "total_signatures":        self.processing_stats["total_signatures"],
                "visualizations_created":  self.processing_stats["visualizations_created"],
                "processing_time_seconds": elapsed,
                "processing_time_formatted": (
                    f"{int(elapsed//3600):02d}:"
                    f"{int((elapsed%3600)//60):02d}:"
                    f"{int(elapsed%60):02d}"
                )
            }
        }

        report_path = self.config.output_dir / "summary_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        md_report = f"""# KIF Module C: Signature Mining Summary

## Overview
- **Timestamp:** {report['timestamp']}
- **Model:** {report['model']}
- **Device:** {report['hardware']['device_name']}
- **CUDA Version:** {report['hardware']['cuda_version']}

## Configuration
- **ROME Layers:** {report['config']['rome_layers']}
- **Layer Selection:** {report['config']['layer_selection']}
- **Top-K Directions:** {report['config']['top_k_directions']}
- **Significance Threshold:** {report['config']['significance_threshold']}
- **Activation Strategy:** {report['config']['activation_strategy']}
- **Dimension Standardization:** {report['config']['standardize_dims']}
- **Target Dimension:** {report['config']['target_dim']}

## Negative Pool
- **Mode:** {report['config']['negative_pool']['mode']}
- **Synthetic fraction cap:** {report['config']['negative_pool']['synthetic_fraction']}

## Results
- **Total Subjects Analyzed:** {report['results']['total_subjects_analyzed']}
- **Successful Subjects:** {report['results']['successful_subjects']}
- **Failed Subjects:** {report['results']['failed_subjects']}
- **Total Signatures Extracted:** {report['results']['total_signatures']}
- **Visualizations Created:** {report['results']['visualizations_created']}
- **Processing Time:** {report['results']['processing_time_formatted']}

## Next Steps
Extracted signatures → Module D (capsule forger) → Module E (hyper-sentinel).
"""
        md_path = self.config.output_dir / "summary_report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_report)
        logger.info(f"Saved summary report to {report_path} and {md_path}")


# ============================================================
# SECTION 7 — Entrypoint
# ============================================================

def run_module_c():
    """
    Run Module C: Signature Mining (operational).

    Produces:
      outputs/signatures/top_signatures.pkl.gz   ← read by Module D
      outputs/signatures/signature_index.json    ← read by Module D/E
      outputs/signatures/subject_data/<s>.json   ← per-subject details
      outputs/signatures/plots/*.png             ← distribution plots
      outputs/signatures/visualizations/*.png    ← PCA plots
      outputs/signatures/summary_report.{json,md}
    """
    logger.info("=" * 60)
    logger.info("Starting KIF Module C: Signature Mining")
    logger.info("=" * 60)

    config = SignatureMiningConfig(
        rome_hparams=ROMEHyperParams(
            layers=[11, 12, 13, 14],  # used only when layer_selection != "all"
            layer_selection="top_k",
            target_module="mlp",
            significance_threshold=1.5
        ),
        top_k_directions=3,
        min_prompts_per_subject=2,
        use_semantic_negatives=True,
        min_controls_per_subject=1,
        allow_synthetic_fallback=True,

        # FIX R3-1: oversampling OFF for direction learning
        enable_oversampling=False,
        oversample_strategy="max",
        oversample_separately=True,
        preserve_original_ratio=False,

        # FIX R3-7: matched-size neg pool for mining; 10% synthetic cap
        negative_pool_mode="match_positives",
        fixed_negative_pool_size=100,   # ignored in match_positives mode
        synthetic_fraction=0.10,

        activation_strategy="mean_token",
        standardize_dims=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_half_precision=False,
        enable_memory_cleanup=True,
        cleanup_frequency=5
    )

    extractor = SignatureExtractor(config)
    try:
        logger.info("Extracting signatures for all subjects...")
        signatures = extractor.extract_all_signatures()

        logger.info("Saving signature results...")
        extractor.save_signature_index(signatures)

        logger.info("Creating summary report...")
        extractor.create_summary_report()

        logger.info("=" * 60)
        logger.info("Module C completed successfully!")
        logger.info(
            f"Extracted signatures for "
            f"{extractor.processing_stats['successful_subjects']} subjects"
        )
        logger.info(
            f"Final GPU memory: "
            f"{extractor.memory_manager.get_gpu_memory_mb():.0f} MB"
        )
        logger.info("=" * 60)

        return signatures
    except Exception as e:
        logger.error(f"Module C failed: {e}", exc_info=True)
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    signatures = run_module_c()
