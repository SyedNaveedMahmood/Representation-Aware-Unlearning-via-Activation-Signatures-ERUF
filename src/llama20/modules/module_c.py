#src/llama20/modules/module_c.py

 # Module C: Signature Mining with ROME Integration (CUDA-Accelerated + Balanced Dataset)

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import time
import logging
import torch
import numpy as np
import gc
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm.auto import tqdm
import pickle
import gzip
import matplotlib.pyplot as plt

# Ensure pickle compression is available
try:
    import compress_pickle
except ImportError:
    # Create fallback implementation
    class CompressPickleFallback:
        def load(self, filename):
            with gzip.open(filename, 'rb') as f:
                return pickle.load(f)

        def dump(self, data, filename, compression="gzip", compresslevel=3, **kwargs):
            with gzip.open(filename, 'wb', compresslevel=compresslevel) as f:
                pickle.dump(data, f)

    compress_pickle = CompressPickleFallback()
    logging.warning("compress_pickle not available, using fallback implementation.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("kif_signature_cuda_balanced.log")
    ]
)
logger = logging.getLogger('KIF-ModuleC-CUDA-Balanced')

# -------------------------
# CUDA-Accelerated Utilities
# -------------------------

class StandardScaler:
    """PyTorch implementation of StandardScaler with GPU support"""
    def __init__(self, device='cpu'):
        self.mean_ = None
        self.scale_ = None
        self.device = device

    def fit(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)
        
        self.mean_ = torch.mean(X, dim=0)
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
    """PyTorch implementation of PCA with GPU support (via SVD)"""
    def __init__(self, n_components=None, random_state=None, device='cpu'):
        self.n_components = n_components
        self.random_state = random_state
        self.device = device
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)
        
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        self.mean_ = torch.mean(X, dim=0)
        X_centered = X - self.mean_
        
        U, s, Vt = torch.linalg.svd(X_centered, full_matrices=False)

        if self.n_components is None:
            self.n_components = min(X.shape[0], X.shape[1])

        self.components_ = Vt[:self.n_components]
        explained_variance = (s ** 2) / (X.shape[0] - 1)
        total_variance = torch.sum(explained_variance)
        
        if total_variance == 0:
            self.explained_variance_ratio_ = torch.zeros(self.n_components, device=self.device)
        else:
            self.explained_variance_ratio_ = explained_variance[:self.n_components] / total_variance
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)
        X_centered = X - self.mean_
        return torch.matmul(X_centered, self.components_.T)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

def compute_silhouette_score(X, labels, device='cpu'):
    """Compute silhouette score using PyTorch"""
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    
    X = X.to(device)
    labels = labels.to(device)
    
    n_samples = X.shape[0]
    unique_labels = torch.unique(labels)
    
    if len(unique_labels) == 1:
        return 0.0

    silhouette_scores = []
    for i in range(n_samples):
        same_mask = labels == labels[i]
        same_points = X[same_mask]
        
        if same_points.shape[0] <= 1:
            silhouette_scores.append(0.0)
            continue

        # Compute a (mean intra-cluster distance)
        dists = torch.norm(X[i].unsqueeze(0) - same_points, dim=1)
        mask = dists > 0  # Exclude the point itself
        if mask.sum() > 0:
            a = torch.mean(dists[mask])
        else:
            a = 0.0

        # Compute b (mean nearest-cluster distance)
        b = float('inf')
        for lab in unique_labels:
            if lab != labels[i]:
                other_points = X[labels == lab]
                if other_points.shape[0] > 0:
                    avg_dist = torch.mean(torch.norm(X[i].unsqueeze(0) - other_points, dim=1))
                    b = min(b, avg_dist.item())
        
        if b == float('inf'):
            silhouette_scores.append(0.0)
        else:
            silhouette_scores.append((b - a.item()) / max(a.item(), b))
    
    return float(np.mean(silhouette_scores))

def bootstrap_resample(data, random_state=None, device='cpu'):
    """Bootstrap resampling using PyTorch"""
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    data = data.to(device)
    
    if random_state is not None:
        torch.manual_seed(random_state)
    
    n = data.shape[0]
    idx = torch.randint(0, n, (n,), device=device)
    return data[idx]

# -------------------------
# Configs
# -------------------------

@dataclass
class ROMEHyperParams:
    """Hyperparameters for ROME-based signature mining"""
    layers: List[int] = field(default_factory=lambda: [9, 10, 11])
    layer_selection: str = "all"  # "all" or "top_k"
    target_module: str = "mlp"
    edit_weight: float = 1.0
    significance_threshold: float = 2.0
    fact_token_strategy: str = "last"
    v_num_grad_steps: int = 20
    v_lr: float = 5e-1
    v_loss_layer: int = -1
    v_weight_decay: float = 0.5
    clamp_norm_factor: float = 0.01
    window_size: int = 5

    def __post_init__(self):
        self.layers = sorted(self.layers)

@dataclass
class SignatureMiningConfig:
    # Paths
    activations_dir: Path = Path("outputs/activations_subject_span_mlpblock")
    output_dir: Path = Path("outputs/signatures_subject_span_mlpblock")
    model_dir: str = "outputs/model"
    prompts_file: str = "outputs/datasets/prompts.jsonl"

    rome_hparams: ROMEHyperParams = field(default_factory=ROMEHyperParams)

    top_k_directions: int = 3
    min_prompts_per_subject: int = 3

    use_semantic_negatives: bool = True
    min_controls_per_subject: int = 1
    allow_synthetic_fallback: bool = True

    positive_keys: List[str] = field(default_factory=lambda: [
        "direct", "contextual", "context", "implicit",
        "reasoning", "reason", "misleading", "leak"
    ])
    control_keys: List[str] = field(default_factory=lambda: ["control"])

    # v1: no duplicate-based balancing during mining
    enable_oversampling: bool = False
    oversample_strategy: str = "max"
    oversample_separately: bool = False
    preserve_original_ratio: bool = True

    # Hybrid negative pool
    negative_pool_mode: str = "match_positives"   # "match_positives" | "fixed"
    fixed_negative_pool_size: int = 100
    synthetic_fraction: float = 0.10              # max 10% synthetic in mining

    batch_size: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_half_precision: bool = False

    # v1 activation processing
    activation_strategy: str = "subject_span_mean"
    token_pos: int = -1

    # v1: do not pad/truncate dimensions silently
    standardize_dims: bool = False
    target_dim: Optional[int] = None

    enable_memory_cleanup: bool = True
    cleanup_frequency: int = 5

    n_bootstrap_samples: int = 100
    random_state: int = 42

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.activations_dir = Path(self.activations_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.activations_dir.exists():
            raise FileNotFoundError(f"Activation directory not found: {self.activations_dir}")
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "subject_data").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        
        # Log device info
        if self.device == "cuda" and torch.cuda.is_available():
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.info("Using CPU for computations")
            
        logger.info(f"Signature mining config: device={self.device}, half_precision={self.use_half_precision}")
        logger.info(f"Oversampling enabled: {self.enable_oversampling}, strategy={self.oversample_strategy}")
        logger.info(f"ROME hyperparameters: {self.rome_hparams.__dict__}")

# -------------------------
# Memory / Activation
# -------------------------

class MemoryManager:
    def __init__(self, config: SignatureMiningConfig):
        self.config = config

    def get_gpu_memory_mb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0

    def cleanup(self) -> None:
        """Manual cleanup - only called when explicitly needed"""
        if not self.config.enable_memory_cleanup:
            return
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.debug(f"Memory cleanup - GPU: {self.get_gpu_memory_mb():.0f}MB")

class ActivationManager:
    """Manager for loading and processing activations & prompt metadata"""
    def __init__(self, config: SignatureMiningConfig):
        self.config = config
        self.memory_manager = MemoryManager(config)
        self.activation_index = None
        self.prompts_data = None
        self.target_dim = None
        self.load_activation_index()
        self.load_prompts()
        self._detect_target_dimension()

    def load_activation_index(self) -> None:
        index_path = self.config.activations_dir / "activation_index.json"
        with open(index_path, 'r') as f:
            self.activation_index = json.load(f)
        logger.info(f"Loaded activation index with {len(self.activation_index['prompts'])} prompts")

    def load_prompts(self) -> None:
        prompts = []
        with open(self.config.prompts_file, 'r', encoding='utf-8') as f:
            for line in f:
                prompts.append(json.loads(line))
        self.prompts_data = {p["id"]: p for p in prompts}
        logger.info(f"Loaded {len(self.prompts_data)} prompts from {self.config.prompts_file}")

    # ---- prompt classification helpers ----

    @staticmethod
    def _iter_possible_fields(d: Dict[str, Any]) -> List[str]:
        """Flatten and lower-case all string-like metadata for robust matching."""
        vals = []
        for k, v in d.items():
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
                # one level deep
                for x in v.values():
                    if isinstance(x, str):
                        vals.append(x.lower())
        return vals

    def classify_prompt(self, prompt_data: Dict[str, Any]) -> str:
        """Return 'positive' (leak) or 'control' if detectable; else 'unknown'."""
        vals = self._iter_possible_fields(prompt_data)
        # strong signals
        for key in self.config.control_keys:
            if any(key in v for v in vals):
                return "control"
        for key in self.config.positive_keys:
            if any(key in v for v in vals):
                return "positive"
        # weak heuristics (fallback)
        text_hints = " ".join([prompt_data.get("prompt", ""), prompt_data.get("expected", "")]).lower()
        if any(k in text_hints for k in self.config.control_keys):
            return "control"
        if any(k in text_hints for k in self.config.positive_keys):
            return "positive"
        return "unknown"

    # ---- activation IO/processing ----

    def load_activation_file(self, path: str) -> Optional[np.ndarray]:
        try:
            activation = compress_pickle.load(path)
            if isinstance(activation, np.ndarray) and activation.dtype != np.float32:
                activation = activation.astype(np.float32)
            return activation
        except Exception as e:
            logger.error(f"Failed to load activation from {path}: {e}")
            return None

    def _standardize_dimension(self, activation: np.ndarray, target_dim: int) -> np.ndarray:
        current_dim = activation.shape[0]
        if current_dim == target_dim:
            return activation
        if current_dim > target_dim:
            return activation[:target_dim]
        padded = np.zeros(target_dim, dtype=activation.dtype)
        padded[:current_dim] = activation
        return padded

    def _process_single_activation(self, activation: np.ndarray) -> Optional[np.ndarray]:
        """
        v1 activation contract:
        Module B saves subject-span MLP-block activations as [1, H].
        Return a strict 1D [H] vector.
        """
        if activation is None:
            return None

        try:
            strat = self.config.activation_strategy

            if strat != "subject_span_mean":
                raise ValueError(
                    "KIF-SubjectSpan-MLPBlock-v1 requires "
                    "activation_strategy='subject_span_mean'."
                )

            if activation.ndim == 1:
                processed = activation

            elif activation.ndim == 2:
                # Expected Module B v1 shape: [1, H].
                # mean(axis=0) also tolerates accidental [subject_tokens, H].
                processed = np.mean(activation, axis=0)

            elif activation.ndim == 3 and activation.shape[0] == 1:
                processed = np.mean(activation[0], axis=0)

            else:
                raise ValueError(f"Unexpected activation shape for v1: {activation.shape}")

            if processed.ndim != 1:
                processed = processed.reshape(-1)

            return processed.astype(np.float32)

        except Exception as e:
            logger.warning(f"Error in _process_single_activation: {e}")
            return None

    def _detect_target_dimension(self) -> None:
        if not self.config.standardize_dims:
            return
        logger.info("Auto-detecting target activation dimension...")
        sample_dims = []
        max_samples = 10
        count = 0
        for prompt_id, prompt_info in self.activation_index["prompts"].items():
            if count >= max_samples:
                break
            for path in prompt_info["paths"][:1]:
                try:
                    activation = self.load_activation_file(path)
                    if activation is not None:
                        processed = self._process_single_activation(activation)
                        if processed is not None:
                            sample_dims.append(processed.shape[-1])
                            count += 1
                            break
                except Exception as e:
                    logger.warning(f"Failed to sample activation from {path}: {e}")
                    continue
        if sample_dims:
            from collections import Counter
            self.target_dim = Counter(sample_dims).most_common(1)[0][0]
            logger.info(f"Auto-detected target dimension: {self.target_dim}")
        else:
            logger.warning("Could not auto-detect target dimension, using raw activations")

    # ---- grouping ----

    def group_by_subject(self) -> Dict[str, List[Dict]]:
        """Group prompts by subject and annotate each with its class ('positive' or 'control')."""
        groups = defaultdict(list)
        for prompt_id, pinfo in self.activation_index["prompts"].items():
            if prompt_id not in self.prompts_data:
                continue
            pdata = self.prompts_data[prompt_id]
            subject = pdata.get("subject", "")
            if not subject:
                continue
            label = self.classify_prompt(pdata)
            groups[subject].append({
                "prompt_id": prompt_id,
                "paths": pinfo["paths"],
                "records": pinfo.get("records", []),
                "triple_id": pdata.get("triple_id", ""),
                "prompt": pdata.get("prompt", ""),
                "expected": pdata.get("expected", ""),
                "class": label
            })

        # optional: limit subjects by size; here we only filter if no positives at all
        filtered = {}
        for subj, plist in groups.items():
            pos = [x for x in plist if x["class"] == "positive" or x["class"] == "unknown"]
            if len(pos) >= self.config.min_prompts_per_subject:
                filtered[subj] = plist
        logger.info(f"Grouped prompts into {len(filtered)} subject groups (after filtering)")
        
        # Apply oversampling if enabled
        if self.config.enable_oversampling:
            filtered = self._apply_oversampling(filtered)
        
        return filtered

    def _apply_oversampling(self, subject_groups: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        Oversample all subjects to have the same number of prompts as the subject with the most prompts.
        Can optionally preserve positive/control ratios and oversample classes separately.
        """
        if not subject_groups:
            return subject_groups
        
        # Calculate target size based on strategy
        all_sizes = [len(prompts) for prompts in subject_groups.values()]
        
        if self.config.oversample_strategy == "max":
            target_size = max(all_sizes)
        elif self.config.oversample_strategy == "median":
            target_size = int(np.median(all_sizes))
        else:
            try:
                target_size = int(self.config.oversample_strategy)
            except (ValueError, TypeError):
                logger.warning(f"Invalid oversample_strategy '{self.config.oversample_strategy}', using 'max'")
                target_size = max(all_sizes)
        
        logger.info(f"Oversampling subjects to target size: {target_size}")
        logger.info(f"Original subject sizes: min={min(all_sizes)}, max={max(all_sizes)}, median={np.median(all_sizes):.1f}")
        
        balanced_groups = {}
        oversample_stats = {"subjects_oversampled": 0, "total_prompts_added": 0}
        
        np.random.seed(self.config.random_state)
        
        for subject, prompts in subject_groups.items():
            original_size = len(prompts)
            
            if original_size >= target_size:
                # No oversampling needed
                balanced_groups[subject] = prompts
                continue
            
            if self.config.oversample_separately:
                # Separate by class and oversample each independently
                positives = [p for p in prompts if p["class"] in ("positive", "unknown")]
                controls = [p for p in prompts if p["class"] == "control"]
                
                if self.config.preserve_original_ratio and len(positives) > 0 and len(controls) > 0:
                    # Maintain the original ratio
                    original_ratio = len(positives) / len(prompts)
                    target_positives = int(target_size * original_ratio)
                    target_controls = target_size - target_positives
                else:
                    # Equal distribution or handle edge cases
                    if len(positives) > 0 and len(controls) > 0:
                        target_positives = target_size // 2
                        target_controls = target_size - target_positives
                    elif len(positives) > 0:
                        target_positives = target_size
                        target_controls = 0
                    else:
                        target_positives = 0
                        target_controls = target_size
                
                # Oversample positives
                oversampled_positives = []
                if len(positives) > 0 and target_positives > 0:
                    indices = np.random.choice(len(positives), size=target_positives, replace=True)
                    oversampled_positives = [prompts[i].copy() for i in indices]
                    # Mark oversampled items
                    for i, item in enumerate(oversampled_positives):
                        if i >= len(positives):
                            item["oversampled"] = True
                
                # Oversample controls
                oversampled_controls = []
                if len(controls) > 0 and target_controls > 0:
                    control_indices = [i for i, p in enumerate(prompts) if p["class"] == "control"]
                    indices = np.random.choice(len(control_indices), size=target_controls, replace=True)
                    oversampled_controls = [prompts[control_indices[i]].copy() for i in indices]
                    # Mark oversampled items
                    for i, item in enumerate(oversampled_controls):
                        if i >= len(controls):
                            item["oversampled"] = True
                
                balanced_prompts = oversampled_positives + oversampled_controls
            else:
                # Simple oversampling: sample with replacement from entire prompt list
                indices = np.random.choice(len(prompts), size=target_size, replace=True)
                balanced_prompts = [prompts[i].copy() for i in indices]
                # Mark oversampled items
                for i, item in enumerate(balanced_prompts):
                    if i >= original_size:
                        item["oversampled"] = True
            
            # Shuffle to avoid any ordering bias
            np.random.shuffle(balanced_prompts)
            
            balanced_groups[subject] = balanced_prompts
            oversample_stats["subjects_oversampled"] += 1
            oversample_stats["total_prompts_added"] += len(balanced_prompts) - original_size
            
            logger.debug(f"Subject '{subject}': {original_size} -> {len(balanced_prompts)} prompts "
                        f"(+{len(balanced_prompts) - original_size} oversampled)")
        
        logger.info(f"Oversampling complete: {oversample_stats['subjects_oversampled']} subjects balanced, "
                   f"{oversample_stats['total_prompts_added']} total prompts added")
        
        # Log final statistics
        final_sizes = [len(prompts) for prompts in balanced_groups.values()]
        logger.info(f"Final subject sizes: min={min(final_sizes)}, max={max(final_sizes)}, mean={np.mean(final_sizes):.1f}")
        
        return balanced_groups
    

def _resolve_negative_target(n_pos: int, mode: str, fixed_size: int) -> int:
    if mode == "match_positives":
        return n_pos
    if mode == "fixed":
        return fixed_size
    raise ValueError(f"Unknown negative_pool_mode={mode}")


def _get_cross_subject_negatives(
    exclude_subject: str,
    all_groups: Dict[str, List[Dict]],
    layer: int,
    tracer,
    n: Optional[int] = None,
    seed: int = 42,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)

    candidates: List[Dict] = []
    other_subjects = [s for s in all_groups if s != exclude_subject]
    rng.shuffle(other_subjects)

    for subj in other_subjects:
        candidates.extend([
            p for p in all_groups[subj]
            if p["class"] in ("positive", "unknown")
        ])

    if not candidates:
        return []

    if n is not None and len(candidates) > n:
        idx = rng.choice(len(candidates), n, replace=False)
        candidates = [candidates[i] for i in idx]

    feats, _ = tracer.load_and_process_activations(candidates, layer)
    return feats


def _build_negative_pool(
    subject: str,
    initial_negs: List[np.ndarray],
    pos_features: List[np.ndarray],
    all_groups: Dict[str, List[Dict]],
    layer: int,
    tracer,
    target_negatives: int,
    synthetic_fraction: float,
    allow_synthetic_fallback: bool,
    seed: int,
) -> Tuple[List[np.ndarray], Dict[str, int]]:
    """
    Negative priority:
      1. same-subject controls
      2. cross-subject real positives
      3. bounded synthetic fill
    """
    neg_features = list(initial_negs)
    n_control = len(neg_features)

    target_negatives = max(target_negatives, n_control)

    effective_synth_fraction = synthetic_fraction if allow_synthetic_fallback else 0.0
    synth_budget = int(round(target_negatives * effective_synth_fraction))
    synth_budget = min(synth_budget, max(0, target_negatives - n_control))

    desired_cross = max(0, target_negatives - n_control - synth_budget)

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
    synth_needed = max(0, target_negatives - len(neg_features))
    synth_needed = min(synth_needed, synth_budget)

    if synth_needed > 0:
        synth = tracer.generate_synthetic_negatives(
            pos_features,
            num_negatives=synth_needed,
        )
        neg_features.extend(synth)

    composition = {
        "control": n_control,
        "cross_subject": len(cross_neg),
        "synthetic": len(synth),
    }

    return neg_features, composition

# -------------------------
# Causal Tracer (CUDA-Accelerated)
# -------------------------

class CausalTracer:
    """Find signature directions contrasting leak (positive) vs control (negative) activations"""
    def __init__(self, config: SignatureMiningConfig, activation_manager: ActivationManager):
        self.config = config
        self.activation_manager = activation_manager
        self.memory_manager = MemoryManager(config)
        self.device = torch.device(config.device)

    def _select_record_for_layer(self, prompt: Dict, layer: int) -> Optional[Dict[str, Any]]:
        for record in prompt.get("records", []):
            if int(record.get("layer", -1)) != int(layer):
                continue

            if record.get("activation_source") != "mlp_block":
                continue

            if record.get("token_scope") != "subject_span_mean":
                continue

            target = record.get("target_module_name", "")
            if target and not target.endswith(".mlp"):
                continue

            return record

        return None


    def _select_paths_for_layer(self, prompt: Dict, layer: int) -> Optional[str]:
        record = self._select_record_for_layer(prompt, layer)
        if record is not None:
            return record["path"]

        # Fallback for older indexes, but still only subject-span MLP-block files.
        layer_pat = re.compile(rf"layer[_\-]?0*{layer}[_\-\.]", re.IGNORECASE)
        for path in prompt.get("paths", []):
            p = path.replace("\\", "/")
            if layer_pat.search(p) and ("mlpblock" in p or "mlp_block" in p):
                return path

        return None


    def _layer_metadata(self, prompt_group: List[Dict], layer: int) -> Dict[str, Any]:
        for prompt in prompt_group:
            record = self._select_record_for_layer(prompt, layer)
            if record is not None:
                return {
                    "target_layer": int(record.get("layer", layer)),
                    "target_module_name": record.get(
                        "target_module_name",
                        f"model.layers.{layer}.mlp"
                    ),
                    "activation_source": record.get("activation_source", "mlp_block"),
                    "token_scope": record.get("token_scope", "subject_span_mean"),
                    "feature_dim": int(record.get("feature_dim", 0) or 0),
                }

        return {
            "target_layer": int(layer),
            "target_module_name": f"model.layers.{layer}.mlp",
            "activation_source": "mlp_block",
            "token_scope": "subject_span_mean",
            "feature_dim": 0,
        }
        
    def load_and_process_activations(self, prompt_group: List[Dict], layer: int) -> Tuple[List[np.ndarray], List[str]]:
        processed_activations, failures = [], []
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
                processed_activations.append(proc)
            else:
                failures.append(f"Failed to process {layer_path}")
        if failures:
            logger.warning(f"Failed to process {len(failures)} activations for layer {layer}")
        return processed_activations, failures

    def generate_synthetic_negatives(self, positive_features: List[np.ndarray], num_negatives: int = None) -> List[np.ndarray]:
        """Fallback when no control prompts exist - using GPU"""
        if not positive_features:
            return []
        if num_negatives is None:
            num_negatives = len(positive_features)

        try:
            # Convert to torch tensor on device
            pos_stack = torch.from_numpy(np.vstack(positive_features)).float().to(self.device)
            neg_features = []
            
            feature_std = torch.std(pos_stack, dim=0)
            mean_features = torch.mean(pos_stack, dim=0)

            # move away from centroid
            half = max(1, num_negatives // 2)
            for _ in range(half):
                noise = torch.randn_like(mean_features) * feature_std
                neg = (mean_features - 2 * noise).cpu().numpy().astype(np.float32)
                neg_features.append(neg)

            # shuffled positives
            for i in range(num_negatives - len(neg_features)):
                base = positive_features[i % len(positive_features)].copy()
                np.random.shuffle(base)
                neg_features.append(base.astype(np.float32))
            
            return neg_features
        except Exception as e:
            logger.warning(f"Synth negatives generation failed: {e}")
            # ultra-simple fallback
            neg = []
            for i in range(num_negatives):
                base = positive_features[i % len(positive_features)]
                noise = np.random.normal(0, 0.1, base.shape)
                neg.append((base + noise).astype(np.float32))
            return neg
    def _assert_same_feature_dim(
        self,
        positive_features: List[np.ndarray],
        negative_features: List[np.ndarray],
    ) -> int:
        dims = [f.shape[0] for f in positive_features + negative_features]
        unique = sorted(set(dims))
        if len(unique) != 1:
            raise ValueError(
                f"Feature dimension mismatch in Module C v1: {unique}. "
                "Do not pad/truncate in the subject-span MLP-block experiment."
            )
        return unique[0]

    def compute_signature_directions(
        self,
        positive_features: List[np.ndarray],
        negative_features: List[np.ndarray],
    ) -> Dict[str, Any]:
        if not positive_features or not negative_features:
            return {
                "directions": [],
                "directions_raw": [],
                "directions_scaled": [],
                "scores": [],
                "stats": {},
                "preproc": {},
            }

        try:
            feature_dim = self._assert_same_feature_dim(
                positive_features,
                negative_features,
            )

            pos_stack = torch.from_numpy(np.vstack(positive_features)).float().to(self.device)
            neg_stack = torch.from_numpy(np.vstack(negative_features)).float().to(self.device)

            scaler = StandardScaler(device=self.device)
            combined = torch.cat([pos_stack, neg_stack], dim=0)
            scaler.fit(combined)

            pos_scaled = scaler.transform(pos_stack)
            neg_scaled = scaler.transform(neg_stack)

            scaler_mean = scaler.mean_.detach().cpu().numpy().astype(np.float32)
            scaler_scale = scaler.scale_.detach().cpu().numpy().astype(np.float32)
            scaler_scale[scaler_scale == 0] = 1.0

            pos_mean = torch.mean(pos_scaled, dim=0)
            neg_mean = torch.mean(neg_scaled, dim=0)

            diff_vec = pos_mean - neg_mean
            norm = torch.norm(diff_vec)
            primary_scaled = diff_vec / norm if norm > 0 else diff_vec

            # Orient direction toward positive examples.
            pos_proj = torch.matmul(pos_scaled, primary_scaled)
            neg_proj = torch.matmul(neg_scaled, primary_scaled)
            if torch.mean(pos_proj) < torch.mean(neg_proj):
                primary_scaled = -primary_scaled
                pos_proj = -pos_proj
                neg_proj = -neg_proj

            primary_scaled_np = primary_scaled.detach().cpu().numpy().astype(np.float32)

            # Critical v1 conversion:
            # standardized direction -> raw activation-space direction.
            primary_raw_np = primary_scaled_np / scaler_scale
            primary_raw_np = primary_raw_np / (np.linalg.norm(primary_raw_np) + 1e-12)
            primary_raw_np = primary_raw_np.astype(np.float32)

            pos_mean_proj = float(torch.mean(pos_proj).cpu().item())
            neg_mean_proj = float(torch.mean(neg_proj).cpu().item())
            pooled_std = float(torch.sqrt(
                (torch.var(pos_proj, unbiased=True) + torch.var(neg_proj, unbiased=True)) / 2
            ).cpu().item())
            effect_size = abs(pos_mean_proj - neg_mean_proj) / (pooled_std + 1e-6)

            effect_samples = []
            for i in range(min(self.config.n_bootstrap_samples, 50)):
                ps = bootstrap_resample(pos_proj, self.config.random_state + i, self.device)
                ns = bootstrap_resample(neg_proj, self.config.random_state + i + 1000, self.device)
                p_mean = float(torch.mean(ps).cpu().item())
                n_mean = float(torch.mean(ns).cpu().item())
                p_std = float(torch.sqrt(
                    (torch.var(ps, unbiased=True) + torch.var(ns, unbiased=True)) / 2
                ).cpu().item())
                effect_samples.append(abs(p_mean - n_mean) / (p_std + 1e-6))

            effect_samples = np.asarray(effect_samples, dtype=np.float32)

            directions_scaled = [primary_scaled_np]
            directions_raw = [primary_raw_np]
            scores = [float(effect_size)]

            # Keep secondary PCA if desired, but export raw conversions too.
            if self.config.top_k_directions > 1:
                try:
                    all_scaled = torch.cat([pos_scaled, neg_scaled], dim=0)
                    proj_vals = torch.matmul(all_scaled, primary_scaled)
                    residuals = all_scaled - torch.outer(proj_vals, primary_scaled)

                    n_extra = min(
                        self.config.top_k_directions - 1,
                        max(1, min(pos_scaled.shape[0], neg_scaled.shape[0]) - 1),
                    )

                    pca = PCA(
                        n_components=n_extra,
                        random_state=self.config.random_state,
                        device=self.device,
                    )
                    pca.fit(residuals)

                    for comp in pca.components_:
                        comp = comp / (torch.norm(comp) + 1e-12)

                        pos_c = torch.matmul(pos_scaled, comp)
                        neg_c = torch.matmul(neg_scaled, comp)

                        if torch.mean(pos_c) < torch.mean(neg_c):
                            comp = -comp
                            pos_c = -pos_c
                            neg_c = -neg_c

                        pooled = float(torch.sqrt(
                            (torch.var(pos_c, unbiased=True) + torch.var(neg_c, unbiased=True)) / 2
                        ).cpu().item())

                        eff = abs(
                            float(torch.mean(pos_c).cpu().item()) -
                            float(torch.mean(neg_c).cpu().item())
                        ) / (pooled + 1e-6)

                        if eff >= self.config.rome_hparams.significance_threshold:
                            comp_scaled_np = comp.detach().cpu().numpy().astype(np.float32)
                            comp_raw_np = comp_scaled_np / scaler_scale
                            comp_raw_np = comp_raw_np / (np.linalg.norm(comp_raw_np) + 1e-12)
                            comp_raw_np = comp_raw_np.astype(np.float32)

                            directions_scaled.append(comp_scaled_np)
                            directions_raw.append(comp_raw_np)
                            scores.append(float(eff))

                except Exception as e:
                    logger.warning(f"Secondary directions PCA failed: {e}")

            stats = {
                "pos_mean": pos_mean_proj,
                "neg_mean": neg_mean_proj,
                "pos_std": float(torch.std(pos_proj).cpu().item()),
                "neg_std": float(torch.std(neg_proj).cpu().item()),
                "effect_size": float(effect_size),
                "effect_ci_low": float(np.percentile(effect_samples, 2.5)),
                "effect_ci_high": float(np.percentile(effect_samples, 97.5)),
                "pos_count": len(positive_features),
                "neg_count": len(negative_features),
                "feature_dim": int(feature_dim),
                "projection_space": "scaled_for_stats_raw_for_export",
            }

            return {
                # Compatibility: downstream code that reads "directions" gets raw vectors.
                "directions": directions_raw[:self.config.top_k_directions],
                "directions_raw": directions_raw[:self.config.top_k_directions],
                "directions_scaled": directions_scaled[:self.config.top_k_directions],
                "scores": scores[:self.config.top_k_directions],
                "stats": stats,
                "preproc": {
                    "mean": scaler_mean.tolist(),
                    "scale": scaler_scale.tolist(),
                },
            }

        except Exception as e:
            logger.error(f"Error in compute_signature_directions: {e}")
            return {
                "directions": [],
                "directions_raw": [],
                "directions_scaled": [],
                "scores": [],
                "stats": {},
                "preproc": {},
            }
    def generate_pca_visualization(self, pos_features: List[np.ndarray], neg_features: List[np.ndarray], subject: str, layer: int) -> Optional[str]:
        if len(pos_features) < 3 or len(neg_features) < 3:
            return None
        try:
            # Dimension alignment
            dims = [f.shape[0] for f in pos_features + neg_features]
            if len(set(dims)) != 1:
                logger.warning(
                    f"Skipping PCA visualization for subject={subject}, layer={layer}: "
                    f"feature dimension mismatch: {sorted(set(dims))}"
                )
                return None

            # Convert to tensors on GPU
            pos_stack = torch.from_numpy(np.vstack(pos_features)).float().to(self.device)
            neg_stack = torch.from_numpy(np.vstack(neg_features)).float().to(self.device)
            all_data = torch.cat([pos_stack, neg_stack], dim=0)
            labels = np.array([1] * len(pos_features) + [0] * len(neg_features))

            # Scaling and PCA on GPU
            scaler = StandardScaler(device=self.device)
            all_scaled = scaler.fit_transform(all_data)

            pca = PCA(n_components=2, random_state=self.config.random_state, device=self.device)
            emb = pca.fit_transform(all_scaled)
            
            # Move back to CPU for plotting
            emb_np = emb.cpu().numpy()
            all_scaled_np = all_scaled.cpu().numpy()

            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(emb_np[:, 0], emb_np[:, 1], c=labels, cmap='coolwarm', alpha=0.8, s=100)
            plt.colorbar(scatter, label='Class (1=Subject, 0=Control)')
            plt.title(f'PCA Visualization: {subject} (Layer {layer})')
            plt.xlabel('PC1'); plt.ylabel('PC2')

            if len(np.unique(labels)) > 1 and all_scaled_np.shape[0] > 2:
                try:
                    sil = compute_silhouette_score(all_scaled_np, labels, device=self.device)
                    plt.annotate(f'Silhouette: {sil:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                except Exception as e:
                    logger.warning(f"Silhouette failed: {e}")

            viz_path = self.config.output_dir / "visualizations" / f"{subject.replace(' ', '_')}_layer{layer}_pca.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            return str(viz_path)
        except Exception as e:
            logger.error(f"Failed to generate PCA visualization: {e}")
            return None

    def analyze_subject(
        self,
        subject: str,
        prompt_group: List[Dict],
        cross_subject_groups: Optional[Dict[str, List[Dict]]] = None,
    ) -> Dict[str, Any]:
        """Core: use leak (positive) vs control (negative) activations to compute signatures."""
        # Count oversampled vs original prompts
        original_count = len([p for p in prompt_group if not p.get("oversampled", False)])
        oversampled_count = len([p for p in prompt_group if p.get("oversampled", False)])
        
        logger.info(f"Analyzing subject: {subject} with {len(prompt_group)} prompts "
                   f"({original_count} original, {oversampled_count} oversampled)")

        # Determine available layers from filenames
        available_layers = set()
        for p in prompt_group:
            for path in p["paths"]:
                m = re.search(r"layer(\d+)", path)
                if m: available_layers.add(int(m.group(1)))

        # Choose layers (respect config if present; else all available)
        layers_cfg = self.config.rome_hparams.layers
        if not layers_cfg:
            layers = sorted(available_layers)
        else:
            layers = [l for l in layers_cfg if l in available_layers]
        if not layers:
            logger.warning(f"No valid layers found for subject {subject}")
            return {"subject": subject, "layers": {}, "summary": {"status": "no_layers"}}

        # Separate prompt subsets
        positives_all = [p for p in prompt_group if p["class"] == "positive" or p["class"] == "unknown"]
        controls_all = [p for p in prompt_group if p["class"] == "control"]

        if len(positives_all) < self.config.min_prompts_per_subject:
            logger.warning(f"Subject {subject}: insufficient positives ({len(positives_all)})")
            return {"subject": subject, "layers": {}, "summary": {"status": "too_few_positives"}}

        results = {"subject": subject, "layers": {}, "summary": {}, "visualizations": {}, 
                  "oversampling_info": {"original": original_count, "oversampled": oversampled_count}}

        for layer in layers:
            try:
                pos_features, pos_fail = self.load_and_process_activations(positives_all, layer)

                if len(pos_features) < max(3, self.config.min_prompts_per_subject):
                    logger.warning(f"Not enough positive examples for layer {layer} (got {len(pos_features)})")
                    continue

                # Preferred: semantic controls
                raw_control_features: List[np.ndarray] = []

                if (
                    self.config.use_semantic_negatives
                    and len(controls_all) >= self.config.min_controls_per_subject
                ):
                    raw_control_features, _ = self.load_and_process_activations(controls_all, layer)

                target_negatives = _resolve_negative_target(
                    n_pos=len(pos_features),
                    mode=self.config.negative_pool_mode,
                    fixed_size=self.config.fixed_negative_pool_size,
                )

                neg_features, neg_counts = _build_negative_pool(
                    subject=subject,
                    initial_negs=raw_control_features,
                    pos_features=pos_features,
                    all_groups=cross_subject_groups or {},
                    layer=layer,
                    tracer=self,
                    target_negatives=target_negatives,
                    synthetic_fraction=self.config.synthetic_fraction,
                    allow_synthetic_fallback=self.config.allow_synthetic_fallback,
                    seed=self.config.random_state,
                )

                logger.info(
                    f"Layer {layer}: n_pos={len(pos_features)} | "
                    f"n_neg={len(neg_features)} "
                    f"(control={neg_counts['control']}, "
                    f"cross_subject={neg_counts['cross_subject']}, "
                    f"synthetic={neg_counts['synthetic']})"
                )


                if len(neg_features) < 2:
                    logger.warning(f"Layer {layer}: insufficient negatives ({len(neg_features)})")
                    continue

                viz_path = self.generate_pca_visualization(pos_features, neg_features, subject, layer)
                if viz_path:
                    results["visualizations"][str(layer)] = viz_path

                sig = self.compute_signature_directions(pos_features, neg_features)

                meta = self._layer_metadata(prompt_group, layer)

                directions_raw = sig.get("directions_raw", [])
                directions_scaled = sig.get("directions_scaled", [])

                if not directions_raw:
                    logger.warning(f"No directions produced for subject={subject}, layer={layer}")
                    continue

                results["layers"][str(layer)] = {
                    # Compatibility: raw directions are downstream suppression vectors
                    "directions": [
                        v.tolist() if isinstance(v, np.ndarray) else v
                        for v in sig["directions"]
                    ],
                    "directions_raw": [
                        v.tolist() if isinstance(v, np.ndarray) else v
                        for v in directions_raw
                    ],
                    "directions_scaled": [
                        v.tolist() if isinstance(v, np.ndarray) else v
                        for v in directions_scaled
                    ],

                    # Explicit primary vectors
                    "signature_vector": (
                        directions_raw[0].tolist()
                        if isinstance(directions_raw[0], np.ndarray)
                        else directions_raw[0]
                    ),
                    "signature_vector_raw": (
                        directions_raw[0].tolist()
                        if isinstance(directions_raw[0], np.ndarray)
                        else directions_raw[0]
                    ),
                    "signature_vector_scaled": (
                        directions_scaled[0].tolist()
                        if directions_scaled and isinstance(directions_scaled[0], np.ndarray)
                        else (directions_scaled[0] if directions_scaled else None)
                    ),

                    "scores": sig["scores"],
                    "stats": sig["stats"],
                    "preproc": sig.get("preproc", {}),

                    # v1 representation metadata
                    "target_layer": meta["target_layer"],
                    "target_module_name": meta["target_module_name"],
                    "activation_source": meta["activation_source"],
                    "token_scope": meta["token_scope"],
                    "feature_dim": meta["feature_dim"] or sig["stats"].get("feature_dim"),

                    # Negative-pool diagnostics
                    "negative_pool": neg_counts,
                    "positive_count": len(pos_features),
                    "negative_count": len(neg_features),
                    "failed_activations": len(pos_fail),
                }

                # Clean up large arrays
                del pos_features, neg_features

            except Exception as e:
                logger.error(f"Error processing layer {layer} for subject {subject}: {e}")

        # Summarize
        if results["layers"]:
            best_layer, best_data = max(
                results["layers"].items(),
                key=lambda kv: kv[1]["scores"][0] if kv[1]["scores"] else 0
            )
            results["summary"] = {
                "best_layer": best_layer,
                "best_score": best_data["scores"][0] if best_data["scores"] else 0.0,
                "status": "success",
                "layer_count": len(results["layers"]),
                "visualizations": len(results.get("visualizations", {}))
            }
        else:
            results["summary"] = {"status": "no_data"}

        return results

    def plot_signature_distributions(self, subject: str, subject_results: Dict) -> Optional[str]:
        if subject_results.get("summary", {}).get("status") != "success":
            return None
        try:
            best_layer = subject_results["summary"]["best_layer"]
            layer_data = subject_results["layers"][best_layer]

            pos_mean = layer_data["stats"]["pos_mean"]
            neg_mean = layer_data["stats"]["neg_mean"]
            pos_std = max(1e-6, layer_data["stats"]["pos_std"])
            neg_std = max(1e-6, layer_data["stats"]["neg_std"])

            x = np.linspace(
                min(pos_mean - 3*pos_std, neg_mean - 3*neg_std),
                max(pos_mean + 3*pos_std, neg_mean + 3*neg_std),
                1000
            )
            pos_dist = 1/(pos_std * np.sqrt(2 * np.pi)) * np.exp(-(x - pos_mean)**2 / (2 * pos_std**2))
            neg_dist = 1/(neg_std * np.sqrt(2 * np.pi)) * np.exp(-(x - neg_mean)**2 / (2 * neg_std**2))

            plt.figure(figsize=(10, 6))
            plt.plot(x, pos_dist, label='Subject (Leak) Distribution')
            plt.plot(x, neg_dist, label='Control Distribution')
            plt.axvline(x=pos_mean, linestyle='--', alpha=0.5)
            plt.axvline(x=neg_mean, linestyle='--', alpha=0.5)

            eff = layer_data["stats"]["effect_size"]
            lo = layer_data["stats"].get("effect_ci_low", eff * 0.9)
            hi = layer_data["stats"].get("effect_ci_high", eff * 1.1)
            
            # Add oversampling info if available
            title = f'Signature Distribution for "{subject}" (Layer {best_layer})'
            if "oversampling_info" in subject_results:
                info = subject_results["oversampling_info"]
                title += f'\n({info["original"]} original, {info["oversampled"]} oversampled)'
            
            plt.title(title)
            plt.xlabel('Projection Value'); plt.ylabel('Density')
            plt.annotate(f'Effect Size: {eff:.2f} [{lo:.2f}, {hi:.2f}]',
                         xy=(0.05, 0.95), xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            plt.legend(); plt.tight_layout()

            plot_path = self.config.output_dir / "plots" / f"{subject.replace(' ', '_')}_layer{best_layer}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        except Exception as e:
            logger.error(f"Failed to create plot for {subject}: {e}")
            return None

# -------------------------
# Orchestration
# -------------------------

class SignatureExtractor:
    def __init__(self, config: SignatureMiningConfig):
        self.config = config
        self.memory_manager = MemoryManager(config)
        self.activation_manager = ActivationManager(config)
        self.causal_tracer = CausalTracer(config, self.activation_manager)
        self.processing_stats = {
            "successful_subjects": 0,
            "failed_subjects": 0,
            "total_signatures": 0,
            "start_time": time.time(),
            "visualizations_created": 0
        }
        self.subjects_processed = 0

    def extract_all_signatures(self) -> Dict[str, Any]:
        subject_groups = self.activation_manager.group_by_subject()
        all_signatures = {}

        for subject, prompt_group in tqdm(subject_groups.items(), desc="Extracting signatures"):
            try:
                subject_results = self.causal_tracer.analyze_subject(
                    subject,
                    prompt_group,
                    cross_subject_groups=subject_groups,
                )

                plot_path = None
                if subject_results["summary"].get("status") == "success":
                    plot_path = self.causal_tracer.plot_signature_distributions(subject, subject_results)
                    self.processing_stats["successful_subjects"] += 1
                    for _, layer_data in subject_results["layers"].items():
                        self.processing_stats["total_signatures"] += len(layer_data.get("directions", []))
                    self.processing_stats["visualizations_created"] += len(subject_results.get("visualizations", {}))
                else:
                    self.processing_stats["failed_subjects"] += 1

                if plot_path:
                    subject_results["summary"]["plot_path"] = plot_path

                subject_file = self.config.output_dir / "subject_data" / f"{subject.replace(' ', '_')}.json"
                with open(subject_file, 'w', encoding='utf-8') as f:
                    json.dump(subject_results, f, indent=2)

                all_signatures[subject] = subject_results

                # Optional cleanup at intervals
                self.subjects_processed += 1
                if self.config.enable_memory_cleanup and self.subjects_processed % self.config.cleanup_frequency == 0:
                    self.memory_manager.cleanup()
                    logger.info(f"Processed {self.subjects_processed} subjects, GPU memory: {self.memory_manager.get_gpu_memory_mb():.0f}MB")

            except Exception as e:
                logger.error(f"Failed to process subject {subject}: {e}")
                self.processing_stats["failed_subjects"] += 1

        return all_signatures

    def save_signature_index(self, signatures: Dict[str, Any]) -> None:
        index = {
            "config": {
                "rome_layers": self.config.rome_hparams.layers,
                "top_k_directions": self.config.top_k_directions,
                "model_dir": self.config.model_dir,
                "random_state": self.config.random_state,
                "activation_strategy": self.config.activation_strategy,
                "standardize_dims": self.config.standardize_dims,
                "target_dim": self.activation_manager.target_dim,
                "device": self.config.device,
                "oversampling_enabled": self.config.enable_oversampling,
                "oversample_strategy": self.config.oversample_strategy
            },
            "subjects": {},
            "stats": {
                "successful_subjects": self.processing_stats["successful_subjects"],
                "failed_subjects": self.processing_stats["failed_subjects"],
                "total_signatures": self.processing_stats["total_signatures"],
                "visualizations_created": self.processing_stats["visualizations_created"],
                "processing_time": time.time() - self.processing_stats["start_time"]
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        for subject, subject_data in signatures.items():
            if "summary" in subject_data:
                summary = subject_data["summary"].copy()
                if "oversampling_info" in subject_data:
                    summary["oversampling_info"] = subject_data["oversampling_info"]
                index["subjects"][subject] = summary

        index_path = self.config.output_dir / "signature_index.json"
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)
        logger.info(f"Saved signature index to {index_path}")

        top_signatures = {}

        for subject, data in signatures.items():
            if data.get("summary", {}).get("status") != "success":
                continue

            best_layer = data["summary"].get("best_layer")
            if best_layer is None:
                best_layer = data["summary"].get("best_layer_mining")

            if best_layer is None:
                logger.warning(f"Skipping {subject}: no best layer")
                continue

            layer = data["layers"].get(str(best_layer), {})
            raw_vec = layer.get("signature_vector_raw") or layer.get("signature_vector")
            scaled_vec = layer.get("signature_vector_scaled")

            if raw_vec is None or len(raw_vec) == 0:
                logger.warning(f"Skipping {subject}: no raw signature vector")
                continue

            top_signatures[subject] = {
                # Old compatibility
                "best_layer": int(best_layer),
                "effect_size": data["summary"].get("best_score") or data["summary"].get("best_score_mining"),
                "signatures": [raw_vec],

                # New explicit fields
                "target_layer": int(best_layer),
                "target_module_name": layer.get(
                    "target_module_name",
                    f"model.layers.{int(best_layer)}.mlp",
                ),
                "activation_source": layer.get("activation_source", "mlp_block"),
                "token_scope": layer.get("token_scope", "subject_span_mean"),
                "feature_dim": int(layer.get("feature_dim") or len(raw_vec)),

                # Critical v1 compatibility for D/E/7
                "signature_vector": raw_vec,
                "signature_vector_raw": raw_vec,
                "signature_vector_scaled": scaled_vec,

                # Diagnostics
                "preproc": layer.get("preproc", {}),
                "negative_pool": layer.get("negative_pool", {}),
                "positive_count": layer.get("positive_count"),
                "negative_count": layer.get("negative_count"),
            }

        if not top_signatures:
            logger.error("No valid signatures generated! Check signature extraction logic.")
            raise ValueError("No valid signatures generated for any subject")

        top_path = self.config.output_dir / "top_signatures.pkl.gz"
        compress_pickle.dump(top_signatures, top_path, compression="gzip")
        logger.info(f"Saved {len(top_signatures)} top signatures to {top_path}")

    def create_summary_report(self) -> None:
        elapsed_time = time.time() - self.processing_stats["start_time"]
        report = {
            "title": "KIF Module C: Subject-Span MLP-Block Signature Mining Summary",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.config.model_dir,
            "hardware": {
                "device": self.config.device,
                "cuda_available": torch.cuda.is_available(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A"
            },
            "config": {
                "rome_layers": self.config.rome_hparams.layers,
                "top_k_directions": self.config.top_k_directions,
                "significance_threshold": self.config.rome_hparams.significance_threshold,
                "activation_strategy": self.config.activation_strategy,
                "standardize_dims": self.config.standardize_dims,
                "target_dim": self.activation_manager.target_dim,
                "use_half_precision": self.config.use_half_precision,
                "oversampling": {
                    "enabled": self.config.enable_oversampling,
                    "strategy": self.config.oversample_strategy,
                    "separate_classes": self.config.oversample_separately,
                    "preserve_ratio": self.config.preserve_original_ratio
                }
            },
            "results": {
                "total_subjects_analyzed": self.processing_stats["successful_subjects"] + self.processing_stats["failed_subjects"],
                "successful_subjects": self.processing_stats["successful_subjects"],
                "failed_subjects": self.processing_stats["failed_subjects"],
                "total_signatures": self.processing_stats["total_signatures"],
                "visualizations_created": self.processing_stats["visualizations_created"],
                "processing_time_seconds": elapsed_time,
                "processing_time_formatted": f"{int(elapsed_time // 3600):02d}:{int((elapsed_time % 3600) // 60):02d}:{int(elapsed_time % 60):02d}"
            }
        }

        report_path = self.config.output_dir / "summary_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        md_report = f"""# KIF Module C: Signature Mining Summary (CUDA-Accelerated + Balanced Dataset)

## Overview
- **Timestamp:** {report['timestamp']}
- **Model:** {report['model']}
- **Device:** {report['hardware']['device_name']}
- **CUDA Version:** {report['hardware']['cuda_version']}

## Configuration
- **ROME Layers:** {report['config']['rome_layers']}
- **Top-K Directions:** {report['config']['top_k_directions']}
- **Significance Threshold:** {report['config']['significance_threshold']}
- **Activation Strategy:** {report['config']['activation_strategy']}
- **Dimension Standardization:** {report['config']['standardize_dims']}
- **Target Dimension:** {report['config']['target_dim']}
- **Half Precision:** {report['config']['use_half_precision']}

## Negative Pool Construction
- **Enabled:** {report['config']['oversampling']['enabled']}
- **Strategy:** {report['config']['oversampling']['strategy']}
- **Separate Classes:** {report['config']['oversampling']['separate_classes']}
- **Preserve Ratio:** {report['config']['oversampling']['preserve_ratio']}

## Results
- **Total Subjects Analyzed:** {report['results']['total_subjects_analyzed']}
- **Successful Subjects:** {report['results']['successful_subjects']}
- **Failed Subjects:** {report['results']['failed_subjects']}
- **Total Signatures Extracted:** {report['results']['total_signatures']}
- **Visualizations Created:** {report['results']['visualizations_created']}
- **Processing Time:** {report['results']['processing_time_formatted']}

## Performance
All computations performed on GPU using PyTorch CUDA acceleration for:
- Tensor standardization
- PCA decomposition
- Bootstrap resampling
- Statistical calculations

## Negative Pool Details
Oversampling was disabled for KIF-SubjectSpan-MLPBlock-v1.

Negative pools were constructed in priority order:
1. same-subject control prompts
2. cross-subject real activations
3. bounded synthetic negatives

Synthetic negative fraction: {self.config.synthetic_fraction}
Synthetic fallback enabled: {self.config.allow_synthetic_fallback}

## Next Steps
The extracted signatures can now be used in Module D to create antibody capsules.
"""
        md_path = self.config.output_dir / "summary_report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_report)
        logger.info(f"Saved summary report to {report_path} and {md_path}")

# -------------------------
# Entrypoint
# -------------------------

def run_module_c():
    """Run Module C: Signature Mining with CUDA acceleration and dataset balancing"""
    logger.info("=" * 60)
    logger.info("Starting KIF Module C: Signature Mining (CUDA-Accelerated + Balanced)")
    logger.info("=" * 60)

    # Configure
    config = SignatureMiningConfig(
        activations_dir=Path("outputs/activations_subject_span_mlpblock"),
        output_dir=Path("outputs/signatures_subject_span_mlpblock"),

        rome_hparams=ROMEHyperParams(
            layers=[11, 12, 13, 14],
            layer_selection="all",
            target_module="mlp",  # path/record fallback; exact target enforced by metadata
            significance_threshold=1.5,
        ),

        top_k_directions=3,
        min_prompts_per_subject=2,

        use_semantic_negatives=True,
        min_controls_per_subject=1,
        allow_synthetic_fallback=True,

        positive_keys=["direct", "contextual", "implicit", "reasoning", "misleading"],
        control_keys=["control"],

        # v1: no oversampling
        enable_oversampling=False,
        oversample_strategy="max",
        oversample_separately=False,
        preserve_original_ratio=True,

        # v1 hybrid negatives
        negative_pool_mode="match_positives",
        fixed_negative_pool_size=100,
        synthetic_fraction=0.10,

        # v1 activation source
        activation_strategy="subject_span_mean",
        standardize_dims=False,

        device="cuda" if torch.cuda.is_available() else "cpu",
        use_half_precision=False,
        enable_memory_cleanup=True,
        cleanup_frequency=5,
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
        logger.info(f"Extracted signatures for {extractor.processing_stats['successful_subjects']} subjects")
        logger.info(f"Final GPU memory: {extractor.memory_manager.get_gpu_memory_mb():.0f}MB")
        logger.info("=" * 60)

        return signatures
    except Exception as e:
        logger.error(f"Module C failed: {e}", exc_info=True)
        raise
    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    signatures = run_module_c()