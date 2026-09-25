"""Training loop: AdamW parameter groups, warmup + cosine, gradient accumulation, AMP,
per-epoch validation on a frozen reference, early stopping and resumable checkpoints.

Three modes:
  * `synthetic` - no real images at all, verifies the full train/validate/checkpoint path;
  * `overfit`   - 8-16 real studies, no augmentation, in-sample debugging only;
  * `fold`      - normal fold training.

Tiny-subset overfit numbers are never validation performance, and the run summary says so.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from .config import Config, save_config
from .constants import (
    CHECKPOINT_VERSION, DEFAULT_BACKBONE, LABELS_VERSION, PREPROCESS_VERSION, SPLITS_VERSION, STUDY_ID, TARGETS,
)
from .dataset import (
    EpochSampler,
    StudyBagDataset,
    SyntheticBagDataset,
    augment_batch,
    check_study_coverage,
    collate_studies,
    enforce_coverage_gate,
    normalization_stats,
    resolve_augment_device,
)
from .labels import LabelTable, build_label_table, check_training_readiness, load_reference_table
from .loss import (
    GlobalNormalizer,
    LossAccumulator,
    WindowNormalizer,
    global_normalized_bce,
    masked_class_normalized_bce,
    window_normalized_bce,
)
from .metrics import evaluate_predictions, selection_metric, soft_target_warning
from .model import build_model
from .preprocess import preprocess_hash
from .splits import assert_group_disjoint, fold_study_ids, load_splits
from .utils import (
    LOG,
    add_file_logging,
    atomic_write_dataframe,
    atomic_write_json,
    autocast_ctx,
    log_environment,
    peak_gpu_memory_gb,
    reset_peak_gpu_memory,
    seed_everything,
    select_device,
    worker_init_fn,
)


# --------------------------------------------------------------------------------------
# Scheduler
# --------------------------------------------------------------------------------------


def make_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay, stepped once per *optimizer* step."""
    warmup_steps = max(int(warmup_steps), 0)
    total_steps = max(int(total_steps), warmup_steps + 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        progress = (step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return float(0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def eval_loader_settings(cfg: Config, role: str = "eval") -> tuple[int, int | None, bool]:
    """(num_workers, prefetch_factor, persistent_workers) for a training or evaluation loader.

    `train.eval_num_workers` / `train.eval_prefetch_factor` fall back to the training values
    when unset. Evaluation workers are never persistent: they are respawned per validation,
    which costs seconds and keeps their memory out of the training epoch.
    """
    num_workers = int(cfg.train.num_workers)
    prefetch = cfg.train.get("prefetch_factor")
    if role != "eval":
        return num_workers, int(prefetch) if prefetch else None, True

    eval_workers = cfg.train.get("eval_num_workers")
    eval_prefetch = cfg.train.get("eval_prefetch_factor")
    num_workers = num_workers if eval_workers is None else int(eval_workers)
    prefetch = prefetch if eval_prefetch is None else eval_prefetch
    return num_workers, int(prefetch) if prefetch else None, False


def window_sizes(n_microbatches: int, accumulation_steps: int) -> list[int]:
    """Size of the accumulation window each micro-batch belongs to (last one may be short)."""
    sizes: list[int] = []
    remaining = n_microbatches
    while remaining > 0:
        size = min(accumulation_steps, remaining)
        sizes.extend([size] * size)
        remaining -= size
    return sizes


# --------------------------------------------------------------------------------------
# Frozen evaluation reference
# --------------------------------------------------------------------------------------


@dataclass
class EvaluationReference:
    """Targets, masks and study order frozen *before* any training-label experiment."""

    study_ids: list[str]
    values: np.ndarray  # [N, 12] in [0, 1]: 0/1 for hard labels, the supplied value for soft ones
    valid: np.ndarray  # [N, 12] bool
    note: str | None = None

    @classmethod
    def from_table(cls, table: LabelTable, study_ids: Sequence[str]) -> "EvaluationReference":
        subset = table.subset(list(study_ids))
        values, valid = subset.continuous_reference()
        return cls(list(study_ids), values, valid, soft_target_warning(subset.kinds))

    @classmethod
    def from_csv(cls, path: str | Path, study_ids: Sequence[str]) -> "EvaluationReference":
        """Load a reference frozen by `freeze-reference`, independent of the training-label policy."""
        frame = pd.read_csv(path, dtype={STUDY_ID: "string"}).set_index(STUDY_ID)
        missing = [s for s in study_ids if s not in frame.index]
        if missing:
            raise KeyError(f"{len(missing)} validation studies absent from the frozen reference {path}, e.g. {missing[:3]}")
        rows = frame.loc[list(study_ids)]
        values = rows[[f"ref::{t}" for t in TARGETS]].to_numpy(dtype=np.float32)
        valid = rows[[f"valid::{t}" for t in TARGETS]].astype(bool).to_numpy()
        return cls(list(study_ids), values, valid, f"frozen reference: {path}")

    @classmethod
    def for_run(cls, cfg: Config, table: LabelTable, study_ids: Sequence[str]) -> "EvaluationReference":
        """The frozen file when configured, otherwise the reference derived from the label table."""
        frozen = cfg.paths.get("frozen_reference_csv")
        if frozen:
            return cls.from_csv(frozen, study_ids)
        return cls.from_table(table, study_ids)

    def save(self, path: str | Path) -> Path:
        frame = {STUDY_ID: self.study_ids}
        for c, target in enumerate(TARGETS):
            frame[f"ref::{target}"] = self.values[:, c]
            frame[f"valid::{target}"] = self.valid[:, c]
        return atomic_write_dataframe(pd.DataFrame(frame), path)


# --------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_score: float = float("-inf")
    best_epoch: int = -1
    epochs_without_improvement: int = 0
    skipped_steps: int = 0  # optimizer steps the GradScaler skipped on inf/NaN gradients
    history: list[dict] = field(default_factory=list)


def _fingerprint(*parts: Any) -> str:
    """sha256 over strings and arrays; identifies label and reference content, not file paths."""
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            digest.update(str((part.dtype, part.shape)).encode())
            digest.update(np.ascontiguousarray(part).tobytes())
        else:
            digest.update(json.dumps(part, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def _train_label_fingerprint(dataset: Dataset) -> str:
    table = getattr(dataset, "label_table", None)
    if table is not None:
        return _fingerprint(list(dataset.study_ids), table.targets.astype(np.float32), table.weights.astype(np.float32))
    return _fingerprint(len(dataset), getattr(dataset, "seed", None), dataset.label_weight_matrix())


# Everything an exact resume must share with the checkpoint. File paths are left out on
# purpose: the same labels on another machine are still the same labels.
RESUME_CONFIG_KEYS = (
    "seed", "split.fold", "data", "model", "augment", "eval.selection_metric",
    "train.loss_normalization", "train.microbatch_studies", "train.accumulation_steps", "train.warmup_epochs",
    "train.max_epochs", "train.encoder_lr", "train.head_lr", "train.weight_decay", "train.grad_clip",
    "train.early_stopping_patience", "train.amp",
)
# Model keys that do not change what is computed: the pretrained source is overwritten by
# the checkpoint, and chunking and gradient checkpointing only bound memory.
RESUME_IGNORED_MODEL_KEYS = ("weights", "encoder_chunk_size", "grad_checkpointing")


def _dotted(tree: dict, key: str) -> Any:
    for part in key.split("."):
        tree = tree.get(part) if isinstance(tree, dict) else None
    return tree


class Trainer:
    def __init__(
        self,
        cfg: Config,
        model: torch.nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        reference: EvaluationReference,
        run_dir: Path,
        mode: str = "fold",
        provenance: dict | None = None,
    ) -> None:
        self.cfg = cfg
        self.mode = mode
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.reference = reference
        self.provenance = provenance or {}
        self.selection_name = str(cfg.eval.get("selection_metric", "macro_soft_auc"))

        self.device_spec = select_device(cfg.train.amp)
        LOG.info("Compute: %s", self.device_spec.describe())
        self.model = model.to(self.device_spec.device)

        # Augmentation placement. The datasets resolve this independently from the same
        # config key, so a batch carrying `augment_params` is raw and must be augmented here.
        self.augment_device = resolve_augment_device(cfg)
        mean, std = normalization_stats(cfg)
        self.norm_mean = torch.tensor(mean, dtype=torch.float32).view(1, 1, 1, 3, 1, 1).to(self.device_spec.device)
        self.norm_std = torch.tensor(std, dtype=torch.float32).view(1, 1, 1, 3, 1, 1).to(self.device_spec.device)
        self.augment_noise_std = float(cfg.augment.noise_std or 0.0)
        self.augment_generator = torch.Generator(device=self.device_spec.device)
        self.augment_generator.manual_seed(int(cfg.seed))
        LOG.info(
            "Augmentation: %s (augment.device=%s, enabled=%s)",
            "on the training device" if self.augment_device == "cuda" else "in the DataLoader workers",
            cfg.augment.get("device", "auto"),
            bool(cfg.augment.enabled),
        )

        self.sampler_generator: torch.Generator | None = None  # set by the shuffled loader
        self.train_loader = self._make_loader(train_dataset, shuffle=True, batch_size=int(cfg.train.microbatch_studies))
        self.val_loader = self._make_loader(
            val_dataset, shuffle=False, batch_size=int(cfg.train.eval_batch_studies), role="eval"
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameter_groups(
                encoder_lr=float(cfg.train.encoder_lr),
                head_lr=float(cfg.train.head_lr),
                weight_decay=float(cfg.train.weight_decay),
            )
        )
        self.accumulation_steps = int(cfg.train.accumulation_steps)
        self.loss_normalization = str(cfg.train.get("loss_normalization", "microbatch"))
        self.global_normalizer: GlobalNormalizer | None = None
        if self.loss_normalization == "global":
            self.global_normalizer = GlobalNormalizer.from_training_weights(
                train_dataset.label_weight_matrix(),
                window_studies=int(cfg.train.microbatch_studies) * self.accumulation_steps,
            )
            LOG.info("Global loss denominators D_c (expected weight per window): %s", self.global_normalizer.describe())
        self.steps_per_epoch = max(1, math.ceil(len(self.train_loader) / self.accumulation_steps))
        self.scheduler = make_scheduler(
            self.optimizer,
            warmup_steps=self.steps_per_epoch * int(cfg.train.warmup_epochs),
            total_steps=self.steps_per_epoch * int(cfg.train.max_epochs),
        )
        self.scaler = torch.amp.GradScaler(
            self.device_spec.device.type, enabled=bool(self.device_spec.use_grad_scaler)
        )
        self.state = TrainState()
        self.resume_signature = self._resume_signature(train_dataset)
        LOG.info(
            "Effective batch: %d studies/micro-batch x %d accumulation = %d studies per optimizer step; "
            "%d micro-batches -> %d optimizer steps per epoch",
            int(cfg.train.microbatch_studies),
            self.accumulation_steps,
            int(cfg.train.microbatch_studies) * self.accumulation_steps,
            len(self.train_loader),
            self.steps_per_epoch,
        )

    def _make_loader(self, dataset: Dataset, shuffle: bool, batch_size: int, role: str = "train") -> DataLoader:
        """Build a loader. `role="eval"` uses the evaluation worker budget (see eval_loader_settings).

        Validation workers would otherwise sit idle through the whole training epoch while
        holding a full worker's memory each, and their prefetch queue peaks at the same
        moment the training loader refills - the epoch boundary, where the process commit
        peaks. They are therefore neither persistent nor as numerous.
        """
        num_workers, prefetch_factor, persistent = eval_loader_settings(self.cfg, role)
        kwargs: dict[str, Any] = {
            "batch_size": max(1, batch_size),
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": collate_studies,
            "pin_memory": self.device_spec.device.type == "cuda",
            "drop_last": False,
        }
        if num_workers > 0:
            kwargs["worker_init_fn"] = worker_init_fn
            kwargs["persistent_workers"] = persistent
            if prefetch_factor:
                kwargs["prefetch_factor"] = int(prefetch_factor)
        if shuffle:
            # Same seeded shuffle as `shuffle=True, generator=...`; the EpochSampler only attaches
            # the epoch to every index, because persistent workers never see dataset.set_epoch().
            generator = torch.Generator()
            generator.manual_seed(int(self.cfg.seed))
            kwargs["shuffle"] = False
            kwargs["sampler"] = EpochSampler(RandomSampler(dataset, generator=generator))
            self.sampler_generator = generator  # its state is the next epoch's order; checkpointed
        return DataLoader(dataset, **kwargs)

    def _resume_signature(self, train_dataset: Dataset) -> dict:
        """What an exact resume must share with its checkpoint (see RESUME_CONFIG_KEYS)."""
        tree = self.cfg.to_dict()
        signature: dict[str, Any] = {key: _dotted(tree, key) for key in RESUME_CONFIG_KEYS}
        signature["model"] = {
            k: v for k, v in (signature["model"] or {}).items() if k not in RESUME_IGNORED_MODEL_KEYS
        }
        # Runs from before model.backbone existed were all B0 and stored no such key.
        if signature["model"].get("backbone") == DEFAULT_BACKBONE:
            del signature["model"]["backbone"]
        signature["mode"] = self.mode
        signature["device"] = self.device_spec.device.type
        signature["augment_device"] = self.augment_device
        signature["train_labels"] = _train_label_fingerprint(train_dataset)
        signature["validation_reference"] = _fingerprint(
            list(self.reference.study_ids), self.reference.values.astype(np.float32), self.reference.valid.astype(bool)
        )
        return json.loads(json.dumps(signature, sort_keys=True, default=str))  # tuples -> lists, as stored

    # -- one epoch -------------------------------------------------------------------

    def _forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move one micro-batch to the device, augment it if deferred, and run the model."""
        images = batch["images"].to(self.device_spec.device, non_blocking=True)
        slice_valid = batch["slice_valid_mask"].to(self.device_spec.device, non_blocking=True)
        present = batch["series_present_mask"].to(self.device_spec.device, non_blocking=True)
        targets = batch["targets"].to(self.device_spec.device, non_blocking=True)
        weights = batch["label_weights"].to(self.device_spec.device, non_blocking=True)

        if "augment_params" in batch:
            # Raw [0, 1] bags: augment, normalise and mask here, in float32 and outside
            # autocast, so the numbers match what the CPU path would have produced. The
            # workers may ship them as float16 (lossless for a float16 cache, half the
            # shared memory); widen before the transform.
            images = augment_batch(
                images.float(),
                batch["augment_params"].to(self.device_spec.device, non_blocking=True),
                slice_valid,
                present,
                self.norm_mean,
                self.norm_std,
                noise_std=self.augment_noise_std,
                generator=self.augment_generator,
            )

        with autocast_ctx(self.device_spec):
            logits = self.model(images, slice_valid, present)
        return logits, targets, weights

    def _optimizer_step(self) -> None:
        if float(self.cfg.train.grad_clip) > 0:
            self.scaler.unscale_(self.optimizer)  # clip on unscaled gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.train.grad_clip))
        scale_before = self.scaler.get_scale() if self.scaler.is_enabled() else None
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if scale_before is not None and self.scaler.get_scale() < scale_before:
            # Inf/NaN gradients: the scaler skipped optimizer.step() and lowered the scale. The
            # schedule follows the updates actually made, so it does not advance either.
            self.state.skipped_steps += 1
            return
        self.scheduler.step()
        self.state.global_step += 1

    def _log_progress(self, epoch: int, accumulator: LossAccumulator, n_studies: int, started: float) -> None:
        if self.state.global_step % max(1, int(self.cfg.train.log_every)) != 0:
            return
        elapsed = time.time() - started
        LOG.info(
            "epoch %d step %d/%d | loss %.4f | lr enc %.2e head %.2e | %.2f studies/s | peak GPU %.2f GB",
            epoch,
            self.state.global_step,
            self.steps_per_epoch * int(self.cfg.train.max_epochs),
            accumulator.macro,
            self.optimizer.param_groups[0]["lr"],
            self.optimizer.param_groups[1]["lr"],
            n_studies / max(elapsed, 1e-6),
            peak_gpu_memory_gb(),
        )

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()  # reapplies the frozen-BN policy
        dataset = self.train_loader.dataset
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)  # main-process copy (num_workers=0)
        sampler = self.train_loader.sampler
        if isinstance(sampler, EpochSampler):
            sampler.set_epoch(epoch)  # reaches persistent workers through the indices

        started = time.time()
        self.optimizer.zero_grad(set_to_none=True)
        reset_peak_gpu_memory()
        if self.loss_normalization in ("window", "global"):
            accumulator, n_studies, data_failures, objective = self._train_windows(epoch, started)
        else:
            accumulator, n_studies, data_failures = self._train_microbatches(epoch, started)
            objective = None

        summary = accumulator.summary()
        summary.update(
            {
                "epoch": epoch,
                "objective": objective,
                "seconds": round(time.time() - started, 2),
                "studies_per_second": round(n_studies / max(time.time() - started, 1e-6), 3),
                "optimizer_steps": self.state.global_step,
                "skipped_optimizer_steps": self.state.skipped_steps,
                "encoder_lr": self.optimizer.param_groups[0]["lr"],
                "head_lr": self.optimizer.param_groups[1]["lr"],
                "peak_gpu_gb": round(peak_gpu_memory_gb(), 3),
                "studies_without_any_series": data_failures,
            }
        )
        return summary

    def _train_microbatches(self, epoch: int, started: float) -> tuple[LossAccumulator, int, int]:
        """Original policy: class-normalise each micro-batch, average over the window."""
        accumulator = LossAccumulator()
        sizes = window_sizes(len(self.train_loader), self.accumulation_steps)
        window_supervised = False
        n_studies = 0
        data_failures = 0

        for index, batch in enumerate(self.train_loader):
            data_failures += int(sum(1 for m in batch["meta"] if m.get("n_present_slots", 0) == 0))
            logits, targets, weights = self._forward(batch)
            output = masked_class_normalized_bce(logits, targets, weights)
            accumulator.update(output, n_studies=logits.shape[0])
            n_studies += logits.shape[0]
            window_supervised = window_supervised or output.has_supervision

            scaled_loss = output.loss / float(sizes[index])
            self.scaler.scale(scaled_loss).backward()

            is_window_end = (index + 1) == len(self.train_loader) or ((index + 1) % self.accumulation_steps == 0)
            if not is_window_end:
                continue

            if window_supervised:
                self._optimizer_step()
            else:
                LOG.warning("Accumulation window ending at micro-batch %d had no supervision; step skipped", index)
            self.optimizer.zero_grad(set_to_none=True)
            window_supervised = False
            self._log_progress(epoch, accumulator, n_studies, started)
        return accumulator, n_studies, data_failures

    def _train_windows(self, epoch: int, started: float) -> tuple[LossAccumulator, int, int, float]:
        """Window / global policy: one optimizer step per window, contributions summed.

        window: every micro-batch is normalised by the whole window's label counts, so the
        window's micro-batches are buffered first (CPU tensors only) and D_c and |C+| are
        known before the first forward. global: the fixed training-set denominators. Images still go through the model one
        micro-batch at a time. A micro-batch without any valid label is not forwarded at
        all; a window without any valid label skips the optimizer and scheduler step.
        Returns the mean window objective - the quantity that is actually minimised.
        """
        accumulator = LossAccumulator()
        n_studies = 0
        data_failures = 0
        objectives: list[float] = []
        iterator = iter(self.train_loader)
        remaining = len(self.train_loader)

        while remaining > 0:
            size = min(self.accumulation_steps, remaining)
            remaining -= size
            window = [next(iterator) for _ in range(size)]
            window_supervised = any(bool((b["label_weights"] > 0).any()) for b in window)
            if self.global_normalizer is not None:
                normalizer: WindowNormalizer | GlobalNormalizer = self.global_normalizer
                loss_fn = global_normalized_bce
            else:
                normalizer = WindowNormalizer.from_weights([b["label_weights"] for b in window])
                loss_fn = window_normalized_bce
            window_supervised = window_supervised and normalizer.has_supervision
            window_objective = 0.0

            for batch in window:
                data_failures += int(sum(1 for m in batch["meta"] if m.get("n_present_slots", 0) == 0))
                n_batch = int(batch["label_weights"].shape[0])
                n_studies += n_batch
                if not bool((batch["label_weights"] > 0).any()) or not window_supervised:
                    accumulator.record_empty(n_batch)
                    continue
                logits, targets, weights = self._forward(batch)
                output = loss_fn(logits, targets, weights, normalizer)
                accumulator.update(output, n_studies=n_batch)
                self.scaler.scale(output.loss).backward()
                window_objective += float(output.loss.detach())

            if window_supervised:
                self._optimizer_step()
                objectives.append(window_objective)
            else:
                LOG.warning("Accumulation window of %d micro-batch(es) had no supervision; step skipped", size)
            self.optimizer.zero_grad(set_to_none=True)
            self._log_progress(epoch, accumulator, n_studies, started)

        objective = float(np.mean(objectives)) if objectives else float("nan")
        return accumulator, n_studies, data_failures, objective

    # -- validation ------------------------------------------------------------------

    @torch.inference_mode()
    def predict(self, loader: DataLoader) -> tuple[list[str], np.ndarray]:
        self.model.eval()
        study_ids: list[str] = []
        scores: list[np.ndarray] = []
        for batch in loader:
            images = batch["images"].to(self.device_spec.device, non_blocking=True)
            slice_valid = batch["slice_valid_mask"].to(self.device_spec.device, non_blocking=True)
            present = batch["series_present_mask"].to(self.device_spec.device, non_blocking=True)
            with autocast_ctx(self.device_spec):
                logits = self.model(images, slice_valid, present)
            scores.append(torch.sigmoid(logits.float()).cpu().numpy())
            study_ids.extend(batch["study_ids"])
        stacked = np.concatenate(scores, axis=0) if scores else np.zeros((0, len(TARGETS)), dtype=np.float32)
        return study_ids, stacked

    def validate(self, epoch: int) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
        study_ids, scores = self.predict(self.val_loader)
        order = {sid: i for i, sid in enumerate(study_ids)}
        missing = [s for s in self.reference.study_ids if s not in order]
        if missing:
            raise RuntimeError(
                f"{len(missing)} validation studies produced no prediction (e.g. {missing[:3]}). "
                "Validation must not silently drop studies and report metrics on the survivors."
            )
        rows = [order[s] for s in self.reference.study_ids]
        aligned = scores[rows]
        table, summary = evaluate_predictions(
            aligned,
            self.reference.values,
            self.reference.valid,
            threshold=float(self.cfg.eval.threshold),
            min_support=int(self.cfg.eval.min_class_support),
        )
        summary["epoch"] = epoch
        predictions = self._prediction_frame(aligned, epoch)
        return table, summary, predictions

    def _prediction_frame(self, scores: np.ndarray, epoch: int) -> pd.DataFrame:
        rows = []
        for i, study in enumerate(self.reference.study_ids):
            for c, target in enumerate(TARGETS):
                rows.append(
                    {
                        STUDY_ID: study,
                        "target": target,
                        "fold": int(self.cfg.split.fold),
                        "score": float(scores[i, c]),
                        "reference": float(self.reference.values[i, c]),
                        "reference_valid": bool(self.reference.valid[i, c]),
                        "epoch": epoch,
                        "mode": self.mode,
                        "checkpoint": "best.pt",
                        "label_source": self.provenance.get("label_source", ""),
                        "prep_hash": self.provenance.get("prep_hash", ""),
                    }
                )
        return pd.DataFrame(rows)

    # -- checkpoints -----------------------------------------------------------------

    def checkpoint_payload(self, extra: dict | None = None) -> dict:
        return {
            "version": CHECKPOINT_VERSION,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "best_score": self.state.best_score,
            "best_epoch": self.state.best_epoch,
            "selection_metric": self.selection_name,
            "target_order": list(TARGETS),
            "config": self.cfg.to_dict(),
            "model_description": self.model.describe(),
            "mode": self.mode,
            "versions": {
                "checkpoint": CHECKPOINT_VERSION,
                "labels": LABELS_VERSION,
                "preprocess": PREPROCESS_VERSION,
                "splits": SPLITS_VERSION,
                "prep_hash": self.provenance.get("prep_hash", ""),
            },
            "provenance": self.provenance,
            "resume_signature": self.resume_signature,
            "train_state": {
                "epochs_without_improvement": self.state.epochs_without_improvement,
                "skipped_steps": self.state.skipped_steps,
                "history": list(self.state.history),
            },
            "rng_state": {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "numpy": np.random.get_state(),
                "sampler": self.sampler_generator.get_state() if self.sampler_generator is not None else None,
                "augment": self.augment_generator.get_state(),
            },
            **(extra or {}),
        }

    def save_checkpoint(self, name: str, extra: dict | None = None) -> Path:
        path = self.run_dir / name
        torch.save(self.checkpoint_payload(extra), path)
        return path

    def _read_checkpoint(self, path: str | Path) -> dict:
        payload = torch.load(str(path), map_location=self.device_spec.device, weights_only=False)
        if payload.get("target_order") != list(TARGETS):
            raise ValueError(f"Checkpoint target order {payload.get('target_order')} differs from {TARGETS}")
        if payload.get("version") != CHECKPOINT_VERSION:
            raise ValueError(f"Checkpoint version {payload.get('version')} != {CHECKPOINT_VERSION}")
        stored_hash = payload.get("versions", {}).get("prep_hash", "")
        if stored_hash and stored_hash != self.provenance.get("prep_hash", stored_hash):
            LOG.warning(
                "Checkpoint was trained with preprocessing hash %s but the current cache is %s.",
                stored_hash,
                self.provenance.get("prep_hash"),
            )
        return payload

    def load_weights(self, path: str | Path) -> None:
        """Fine-tune start (`train.init_weights`): model weights only.

        Optimizer, scheduler, best score, history and early stopping start fresh, so the
        new run may use other labels, another reference or another selection metric.
        """
        payload = self._read_checkpoint(path)
        self.model.load_state_dict(payload["model"])
        LOG.info(
            "Initialised the model from %s (epoch %s of that run); optimizer, schedule, best score and "
            "history start fresh.",
            path,
            payload.get("epoch"),
        )

    def load_checkpoint(self, path: str | Path) -> None:
        """Exact resume (`train.resume`) at an epoch boundary.

        The run continues as if it had never stopped: model, optimizer, scheduler, scaler,
        early-stopping state, history and every RNG (python, numpy, torch CPU/CUDA, the
        sampler and the augmentation generator) come back. The checkpoint must come from the
        same experiment - same labels, reference, fold, selection metric and training setup;
        anything else is refused. Use `train.init_weights` to start a new run from its weights.
        """
        payload = self._read_checkpoint(path)
        stored = payload.get("resume_signature")
        if stored is None:
            raise ValueError(
                f"{path} predates exact resume (no resume_signature). Use train.init_weights to start a "
                "new run from its weights."
            )
        differing = sorted(k for k in set(stored) | set(self.resume_signature) if stored.get(k) != self.resume_signature.get(k))
        if differing:
            details = "; ".join(f"{k}: checkpoint={stored.get(k)!r} now={self.resume_signature.get(k)!r}" for k in differing[:6])
            raise ValueError(
                f"Cannot resume from {path}: the run differs in {differing}. {details}. A resume must "
                "continue the same experiment; use train.init_weights for a new run from these weights."
            )

        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload["scaler"])
        # The stored epoch is the last *completed* one; training continues with the next.
        self.state.epoch = int(payload["epoch"]) + 1
        self.state.global_step = int(payload["global_step"])
        self.state.best_score = float(payload["best_score"])
        self.state.best_epoch = int(payload["best_epoch"])
        train_state = payload["train_state"]
        self.state.epochs_without_improvement = int(train_state["epochs_without_improvement"])
        self.state.skipped_steps = int(train_state["skipped_steps"])
        self.state.history = list(train_state["history"])

        rng = payload["rng_state"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if rng["cuda"]:
            torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
        if rng["sampler"] is not None:
            self.sampler_generator.set_state(rng["sampler"].cpu())
        self.augment_generator.set_state(rng["augment"].cpu())
        LOG.info(
            "Resumed from %s at epoch %d (best %.4f @ epoch %d, %d epochs without improvement, %d history "
            "rows). Resume is exact at epoch boundaries only; mid-epoch resume is not implemented.",
            path,
            self.state.epoch,
            self.state.best_score,
            self.state.best_epoch,
            self.state.epochs_without_improvement,
            len(self.state.history),
        )

    # -- driver ----------------------------------------------------------------------

    def fit(self) -> dict:
        history_path = self.run_dir / "history.csv"
        start_epoch = self.state.epoch
        no_defined_auc_epochs = 0

        for epoch in range(start_epoch, int(self.cfg.train.max_epochs)):
            self.state.epoch = epoch
            train_summary = self.train_epoch(epoch)
            table, val_summary, predictions = self.validate(epoch)
            score = selection_metric(val_summary, self.selection_name)

            record = {
                "epoch": epoch,
                # Diagnostic epoch loss (accumulated weighted class means), comparable across policies.
                "train_loss": train_summary["loss"],
                # The objective actually minimised; only defined for the window and global policies.
                "train_objective": train_summary["objective"],
                "train_empty_microbatches": train_summary["n_empty_microbatches"],
                "optimizer_steps": train_summary["optimizer_steps"],
                "skipped_optimizer_steps": train_summary["skipped_optimizer_steps"],
                "encoder_lr": train_summary["encoder_lr"],
                "head_lr": train_summary["head_lr"],
                "seconds": train_summary["seconds"],
                "studies_per_second": train_summary["studies_per_second"],
                "peak_gpu_gb": train_summary["peak_gpu_gb"],
                # The early-stopping score (eval.selection_metric); the named macros are logged beside it.
                "val_selection_score": score,
                "val_macro_roc_auc": val_summary["macro_roc_auc"],
                "val_macro_soft_auc": val_summary["macro_soft_auc"],
                "val_macro_spearman": val_summary["macro_spearman"],
                "val_defined_targets": val_summary["n_defined_targets"],
                "val_macro_ap": val_summary["macro_average_precision"],
                "val_macro_f1": val_summary["macro_f1_at_threshold"],
            }
            self.state.history.append(record)
            atomic_write_dataframe(pd.DataFrame(self.state.history), history_path)
            LOG.info(
                "epoch %d | train loss %.4f | val %s %s | macro ROC-AUC %s (%s defined) | AP %s",
                epoch,
                record["train_loss"],
                self.selection_name,
                f"{score:.4f}" if np.isfinite(score) else "NA",
                f"{val_summary['macro_roc_auc']:.4f}" if np.isfinite(val_summary["macro_roc_auc"]) else "NA",
                val_summary["defined_fraction"],
                f"{val_summary['macro_average_precision']:.4f}"
                if np.isfinite(val_summary["macro_average_precision"])
                else "NA",
            )

            if not np.isfinite(score):
                no_defined_auc_epochs += 1
                LOG.error(
                    "No validation target has a defined %s (ROC-AUC undefined for: %s). The checkpoint "
                    "is NOT selected on a NaN, and the radiologist reference audit must not be used as "
                    "a substitute selection criterion.",
                    self.selection_name,
                    val_summary["undefined_targets"],
                )
                if no_defined_auc_epochs >= 2:
                    raise RuntimeError(
                        "Validation produced no defined ROC-AUC for two consecutive epochs. Fix the "
                        "validation label coverage (each target needs known positives AND negatives) "
                        "before training further."
                    )
            else:
                no_defined_auc_epochs = 0
                if score > self.state.best_score:
                    self.state.best_score = score
                    self.state.best_epoch = epoch
                    self.state.epochs_without_improvement = 0
                    self.save_checkpoint("best.pt")
                    atomic_write_dataframe(table, self.run_dir / "metrics_per_class.csv")
                    atomic_write_dataframe(predictions, self.run_dir / "validation_predictions.csv")
                    atomic_write_json(self.run_dir / "val_summary.json", val_summary)
                    LOG.info("New best %s %.4f at epoch %d -> best.pt", self.selection_name, score, epoch)
                else:
                    self.state.epochs_without_improvement += 1

            self.save_checkpoint("last.pt")
            if self.state.epochs_without_improvement >= int(self.cfg.train.early_stopping_patience):
                LOG.info(
                    "Early stopping: %d epochs without improvement (best %.4f @ epoch %d)",
                    self.state.epochs_without_improvement,
                    self.state.best_score,
                    self.state.best_epoch,
                )
                break

        return self.finalize()

    def finalize(self) -> dict:
        summary = {
            "mode": self.mode,
            "fold": int(self.cfg.split.fold),
            "selection_metric": self.selection_name,
            "best_score": self.state.best_score if np.isfinite(self.state.best_score) else None,
            "best_epoch": self.state.best_epoch,
            "epochs_run": self.state.epoch + 1,
            "optimizer_steps": self.state.global_step,
            "n_train_studies": len(self.train_loader.dataset),
            "n_val_studies": len(self.val_loader.dataset),
            "device": self.device_spec.describe(),
            "provenance": self.provenance,
            "metric_scope": (
                f"selection on the local evaluable-target {self.selection_name} (soft ROC-AUC equals ROC-AUC "
                "on a binary reference); not a verified official metric"
            ),
            "run_dir": str(self.run_dir),
        }
        if self.mode == "overfit":
            summary["warning"] = (
                "IN-SAMPLE DEBUG RUN. The same studies are used for fitting and scoring. These numbers "
                "are not held-out performance and must not be reported as a CV result."
            )
        if self.mode == "synthetic":
            summary["warning"] = "SYNTHETIC SMOKE TEST on generated images. No claim about real data."
        if self.reference.note:
            summary["reference_note"] = self.reference.note
        atomic_write_json(self.run_dir / "run_summary.json", summary)
        LOG.info("Run summary: %s", json.dumps(summary, indent=2, default=str))
        return summary


# --------------------------------------------------------------------------------------
# Run builders
# --------------------------------------------------------------------------------------


def _run_dir(cfg: Config, mode: str, name: str | None = None) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%S")
    label = name or f"{mode}_fold{int(cfg.split.fold)}_{stamp}"
    return Path(cfg.paths.output_dir) / label


def _apply_start(trainer: Trainer, cfg: Config) -> None:
    """`train.resume` continues a run exactly; `train.init_weights` starts a new one from its weights."""
    resume, init_weights = cfg.train.get("resume"), cfg.train.get("init_weights")
    if resume and init_weights:
        raise ValueError("Set train.resume (continue the same run) or train.init_weights (new run), not both.")
    if resume:
        trainer.load_checkpoint(resume)
    elif init_weights:
        trainer.load_weights(init_weights)


def run_synthetic(cfg: Config, n_studies: int = 16, epochs: int | None = None, name: str | None = None) -> dict:
    """Smoke test: full training path on generated images, no dataset required."""
    return build_synthetic_trainer(cfg, n_studies, epochs, name).fit()


def build_synthetic_trainer(
    cfg: Config, n_studies: int = 16, epochs: int | None = None, name: str | None = None
) -> Trainer:
    cfg = cfg.copy()
    if epochs is not None:
        cfg.train.max_epochs = int(epochs)
        cfg.train.warmup_epochs = min(int(cfg.train.warmup_epochs), max(1, int(epochs) - 1))
    seed_everything(int(cfg.seed))
    run_dir = _run_dir(cfg, "synthetic", name)
    run_dir.mkdir(parents=True, exist_ok=True)
    add_file_logging(run_dir / "run.log")
    atomic_write_json(run_dir / "environment.json", log_environment())
    save_config(cfg, run_dir / "config.yaml")

    train_ds = SyntheticBagDataset(cfg, n_studies=n_studies, seed=int(cfg.seed))
    val_ds = SyntheticBagDataset(cfg, n_studies=max(8, n_studies // 2), seed=int(cfg.seed) + 1)

    values = np.stack([val_ds[i]["targets"].numpy() for i in range(len(val_ds))])
    weights = np.stack([val_ds[i]["label_weights"].numpy() for i in range(len(val_ds))])
    reference = EvaluationReference(
        study_ids=[val_ds[i]["study_id"] for i in range(len(val_ds))],
        values=values.astype(np.float32),
        valid=weights > 0,
    )
    reference.save(run_dir / "validation_reference.csv")

    model = build_model(cfg)
    trainer = Trainer(
        cfg,
        model,
        train_ds,
        val_ds,
        reference,
        run_dir,
        mode="synthetic",
        provenance={"label_source": "synthetic", "prep_hash": "synthetic"},
    )
    _apply_start(trainer, cfg)
    return trainer


def _report_split_label_counts(
    label_table: LabelTable, train_ids: list[str], val_ids: list[str], run_dir: Path
) -> pd.DataFrame:
    """Known/unknown, positive/negative, borderline and weighted counts per target and split.

    Targets with no usable supervision, or without both classes for evaluation, are named
    explicitly - they must not be quietly reported as successfully trained.
    """
    frames = []
    for split_name, ids in (("train", train_ids), ("validation", val_ids)):
        counts = label_table.subset(list(ids)).counts_frame()
        counts.insert(0, "split", split_name)
        frames.append(counts)
    table = pd.concat(frames, ignore_index=True)
    atomic_write_dataframe(table, run_dir / "label_counts_per_split.csv")
    LOG.info("Label counts per target and split:\n%s", table.to_string(index=False))

    for split_name in ("train", "validation"):
        chunk = table[table["split"] == split_name]
        unsupervised = list(chunk.loc[chunk["n_supervised"] == 0, "target"])
        one_class = list(chunk.loc[(chunk["eff_positive"] == 0) | (chunk["eff_negative"] == 0), "target"])
        if unsupervised:
            LOG.warning("%s split: NO supervision at all for %s", split_name, unsupervised)
        if one_class:
            LOG.warning(
                "%s split: only one class present for %s - these targets are not learnable/evaluable here",
                split_name,
                one_class,
            )
    return table


def _prepare_real_run(
    cfg: Config,
    mode: str,
    study_limit: int | None,
    name: str | None,
    allow_unready: bool,
) -> tuple[Trainer, dict]:
    seed_everything(int(cfg.seed))
    run_dir = _run_dir(cfg, mode, name)
    run_dir.mkdir(parents=True, exist_ok=True)
    add_file_logging(run_dir / "run.log")
    atomic_write_json(run_dir / "environment.json", log_environment())
    save_config(cfg, run_dir / "config.yaml")

    splits = load_splits(cfg)
    fold = int(cfg.split.fold)
    assert_group_disjoint(splits, fold)
    train_ids, val_ids = fold_study_ids(splits, fold)

    all_ids = sorted(set(splits[STUDY_ID].astype(str)))
    label_table = build_label_table(cfg, all_ids)
    readiness = check_training_readiness(cfg, label_table, len(all_ids))
    readiness.log()
    if not readiness.ok and not allow_unready:
        readiness.raise_if_failed()

    if mode == "overfit":
        index = label_table.index
        supervised = [s for s in train_ids if label_table.weights[index[s]].sum() > 0]
        train_ids = supervised[: max(1, int(study_limit or 8))]
        val_ids = list(train_ids)  # in-sample by design; flagged everywhere
        cfg = cfg.copy()
        cfg.augment.enabled = False
        cfg.train.early_stopping_patience = int(cfg.train.max_epochs)
        LOG.warning(
            "Overfit diagnostic on %d studies. Training and 'validation' sets are IDENTICAL: the "
            "resulting numbers are in-sample debugging output, never gold performance.",
            len(train_ids),
        )
    elif study_limit:
        train_ids = train_ids[:study_limit]
        val_ids = val_ids[: max(1, study_limit // 2)]

    train_coverage = check_study_coverage(cfg, train_ids)
    val_coverage = check_study_coverage(cfg, val_ids)
    atomic_write_json(run_dir / "coverage.json", {"train": train_coverage, "validation": val_coverage})
    excluded_train = enforce_coverage_gate(cfg, train_coverage, "training")
    excluded_val = enforce_coverage_gate(cfg, val_coverage, "validation")
    train_ids = [s for s in train_ids if s not in set(excluded_train)]
    val_ids = [s for s in val_ids if s not in set(excluded_val)]

    reference = EvaluationReference.for_run(cfg, label_table, val_ids)
    reference.save(run_dir / "validation_reference.csv")
    if reference.note:
        LOG.warning("%s", reference.note)

    _report_split_label_counts(label_table, train_ids, val_ids, run_dir)

    train_ds = StudyBagDataset(cfg, train_ids, label_table, train=True)
    val_ds = StudyBagDataset(cfg, val_ids, label_table, train=False)

    reference_table = load_reference_table(cfg, all_ids)
    provenance = {
        "label_source": label_table.source,
        "label_policy": label_table.policy,
        "prep_hash": preprocess_hash(cfg),
        "splits_file": str(Path(cfg.paths.work_dir) / "splits" / "splits.csv"),
        "readiness": readiness.details,
        "readiness_ok": readiness.ok,
        "n_excluded_train": len(excluded_train),
        "n_excluded_val": len(excluded_val),
        "reference_audit_available": reference_table is not None,
    }
    if not readiness.ok:
        provenance["warning"] = (
            "Readiness check failed and was explicitly overridden. Results are diagnostic only."
        )

    model = build_model(cfg)
    trainer = Trainer(cfg, model, train_ds, val_ds, reference, run_dir, mode=mode, provenance=provenance)
    _apply_start(trainer, cfg)
    return trainer, provenance


def run_fold(cfg: Config, study_limit: int | None = None, name: str | None = None, allow_unready: bool = False) -> dict:
    trainer, _ = _prepare_real_run(cfg, "fold", study_limit, name, allow_unready)
    return trainer.fit()


def run_overfit(cfg: Config, n_studies: int = 8, name: str | None = None) -> dict:
    trainer, _ = _prepare_real_run(cfg, "overfit", n_studies, name, allow_unready=True)
    return trainer.fit()
