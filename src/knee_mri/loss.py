"""Masked, class-normalised BCE for partially supervised multi-label targets.

    L_c = sum_i w_ic * BCE(z_ic, y_ic) / sum_i w_ic     (classes with positive weight)
    L   = mean_c L_c                                    (over those classes only)

Targets are finite before the BCE is evaluated: multiplying a NaN by a zero weight is
not safe, so unknown cells carry a finite placeholder and weight 0.

Accumulation caveat
-------------------
Averaging class-normalised micro-batch losses is NOT algebraically identical to
normalising once over the whole effective batch, because each micro-batch has its own
per-class weight totals. We pick one policy and state it: micro-batch level class
normalisation, averaged over the accumulation window (the last, possibly shorter,
window is scaled by its real size). The reported epoch loss is computed separately
from accumulated class numerators/denominators, which is the unbiased quantity.

That micro-batch policy has a flaw with one-study micro-batches: w * BCE / w = BCE, so a
label weight only matters through being zero or not, and a class's share of the gradient
depends on how many other labels the same study happens to carry.

Window policy (`train.loss_normalization: window`)
--------------------------------------------------
Normalise over the whole accumulation window with MASK counts, not weight sums:

    D_c = sum_{i in window} m_ic                        (m = w > 0)
    L   = 1/|C+| * sum_{c: D_c > 0} sum_{i in window} m_ic * w_ic * BCE_ic / D_c

Every micro-batch contributes its own numerator over the window's D_c and |C+|, and the
contributions are summed - no further division by the window size. Changing one label's
weight from 1.0 to 0.2 then scales exactly that label's loss and logit gradient by 1/5 and
leaves every denominator unchanged. Caveat: switching a label ON (m 0 -> 1, e.g. a weak
negative) does raise D_c and so dilutes that class's other labels in the window.

Global policy (`train.loss_normalization: global`)
--------------------------------------------------
Fixed per-class denominators, computed once from the whole training set:

    D_c = K * sum_{i in train} w_ic / N_train           (K = studies per window)
    L   = 1/|C+| * sum_{c in C+} sum_{i in window} w_ic * BCE_ic / D_c

with C+ the classes that carry any training weight. D_c is the expected weight total of
class c in one window, so a class contributes ~1 per window on average, as under the
window policy. Every label keeps the same weight in every window: a 0.2 label counts
exactly 1/5 of a 1.0 label, switching a weak label on adds its own term and dilutes
nothing else, and a window holding only weak labels does not cancel the weight back to 1
(which a within-window weight-sum denominator would). Changing the label policy moves
D_c, so the per-label scale of the real labels changes a little between policies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .constants import N_TARGETS, TARGETS


@dataclass
class LossOutput:
    loss: torch.Tensor  # differentiable scalar
    numerators: torch.Tensor  # [12] detached sum of w * BCE
    denominators: torch.Tensor  # [12] detached sum of w
    n_valid_classes: int
    has_supervision: bool


def masked_class_normalized_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> LossOutput:
    """Class-normalised BCE over the supervised cells of one micro-batch."""
    if logits.shape != targets.shape or logits.shape != weights.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)}, targets {tuple(targets.shape)} and weights "
            f"{tuple(weights.shape)} must have the same shape"
        )
    logits32 = logits.float()
    targets32 = targets.float()
    weights32 = weights.float()

    if not torch.isfinite(targets32).all():
        raise ValueError("targets contain non-finite values; unknown cells must carry a finite placeholder")
    if (weights32 < 0).any() or not torch.isfinite(weights32).all():
        raise ValueError("label weights must be finite and non-negative")

    per_cell = F.binary_cross_entropy_with_logits(logits32, targets32, reduction="none")
    weighted = per_cell * weights32
    numerators = weighted.sum(dim=0)
    denominators = weights32.sum(dim=0)
    valid = denominators > 0

    if not bool(valid.any()):
        # Differentiable zero: keeps the graph intact for an empty-supervision micro-batch.
        zero = logits32.sum() * 0.0
        return LossOutput(zero, numerators.detach(), denominators.detach(), 0, False)

    class_losses = numerators[valid] / denominators[valid]
    return LossOutput(
        loss=class_losses.mean(),
        numerators=numerators.detach(),
        denominators=denominators.detach(),
        n_valid_classes=int(valid.sum()),
        has_supervision=True,
    )


@dataclass
class WindowNormalizer:
    """Per-class valid-label counts `D_c` and the active-class count `|C+|` of one window."""

    counts: torch.Tensor  # [12] float32, D_c
    n_active: int

    @classmethod
    def from_weights(cls, weights: list[torch.Tensor] | torch.Tensor) -> "WindowNormalizer":
        stacked = torch.cat(list(weights), dim=0) if isinstance(weights, list) else weights
        if (stacked < 0).any() or not torch.isfinite(stacked).all():
            raise ValueError("label weights must be finite and non-negative")
        counts = (stacked > 0).sum(dim=0).to(torch.float32)
        return cls(counts=counts, n_active=int((counts > 0).sum()))

    @property
    def has_supervision(self) -> bool:
        return self.n_active > 0


def window_normalized_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    normalizer: WindowNormalizer,
) -> LossOutput:
    """One micro-batch's contribution to the window objective (see module docstring).

    Summing the returned `loss` over every micro-batch of the window gives the window loss.
    `numerators`/`denominators` are the weighted BCE and weight sums, so LossAccumulator
    reports the same diagnostic epoch loss as under the micro-batch policy.
    """
    if logits.shape != targets.shape or logits.shape != weights.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)}, targets {tuple(targets.shape)} and weights "
            f"{tuple(weights.shape)} must have the same shape"
        )
    logits32, targets32, weights32 = logits.float(), targets.float(), weights.float()
    if not torch.isfinite(targets32).all():
        raise ValueError("targets contain non-finite values; unknown cells must carry a finite placeholder")
    if (weights32 < 0).any() or not torch.isfinite(weights32).all():
        raise ValueError("label weights must be finite and non-negative")

    per_cell = F.binary_cross_entropy_with_logits(logits32, targets32, reduction="none")
    numerators = (per_cell * weights32).sum(dim=0)  # m * w = w, since w > 0 exactly where m = 1
    denominators = weights32.sum(dim=0)
    mask = weights32 > 0
    n_local = int(mask.any(dim=0).sum())

    if not bool(mask.any()) or normalizer.n_active == 0:
        zero = logits32.sum() * 0.0
        return LossOutput(zero, numerators.detach(), denominators.detach(), 0, False)

    counts = normalizer.counts.to(logits32.device)
    if bool((mask.sum(dim=0) > counts).any()):
        raise ValueError("a micro-batch holds more valid labels than its window normaliser counted")
    active = counts > 0
    loss = (numerators[active] / counts[active]).sum() / float(normalizer.n_active)
    return LossOutput(loss, numerators.detach(), denominators.detach(), n_local, True)


@dataclass
class GlobalNormalizer:
    """Fixed per-class denominators `D_c` from the training set, and the class count `|C+|`."""

    denominators: torch.Tensor  # [12] float32; 0 for a class without any training weight
    n_active: int

    @classmethod
    def from_training_weights(cls, weights: np.ndarray | torch.Tensor, window_studies: int) -> "GlobalNormalizer":
        w = torch.as_tensor(np.asarray(weights, dtype=np.float64))
        if w.ndim != 2 or w.shape[1] != N_TARGETS or w.shape[0] == 0:
            raise ValueError(f"training weights must have shape [N>0, {N_TARGETS}], got {tuple(w.shape)}")
        if (w < 0).any() or not torch.isfinite(w).all():
            raise ValueError("label weights must be finite and non-negative")
        if int(window_studies) < 1:
            raise ValueError("window_studies must be >= 1")
        denominators = (float(window_studies) * w.sum(dim=0) / w.shape[0]).to(torch.float32)
        return cls(denominators=denominators, n_active=int((denominators > 0).sum()))

    @property
    def has_supervision(self) -> bool:
        return self.n_active > 0

    def describe(self) -> dict:
        return {target: round(float(d), 4) for target, d in zip(TARGETS, self.denominators)}


def global_normalized_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    normalizer: GlobalNormalizer,
) -> LossOutput:
    """One micro-batch's contribution to the window objective under fixed denominators."""
    if logits.shape != targets.shape or logits.shape != weights.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)}, targets {tuple(targets.shape)} and weights "
            f"{tuple(weights.shape)} must have the same shape"
        )
    logits32, targets32, weights32 = logits.float(), targets.float(), weights.float()
    if not torch.isfinite(targets32).all():
        raise ValueError("targets contain non-finite values; unknown cells must carry a finite placeholder")
    if (weights32 < 0).any() or not torch.isfinite(weights32).all():
        raise ValueError("label weights must be finite and non-negative")

    per_cell = F.binary_cross_entropy_with_logits(logits32, targets32, reduction="none")
    numerators = (per_cell * weights32).sum(dim=0)
    denominators = weights32.sum(dim=0)
    mask = weights32 > 0

    if not bool(mask.any()) or normalizer.n_active == 0:
        zero = logits32.sum() * 0.0
        return LossOutput(zero, numerators.detach(), denominators.detach(), 0, False)

    fixed = normalizer.denominators.to(logits32.device)
    active = fixed > 0
    if bool((mask.any(dim=0) & ~active).any()):
        raise ValueError("a micro-batch carries weight on a class the training-set normaliser never saw")
    loss = (numerators[active] / fixed[active]).sum() / float(normalizer.n_active)
    return LossOutput(loss, numerators.detach(), denominators.detach(), int(mask.any(dim=0).sum()), True)


class LossAccumulator:
    """Accumulate class numerators/denominators to report an honest epoch loss."""

    def __init__(self, n_targets: int = N_TARGETS) -> None:
        self.numerators = np.zeros(n_targets, dtype=np.float64)
        self.denominators = np.zeros(n_targets, dtype=np.float64)
        self.n_microbatches = 0
        self.n_empty_microbatches = 0
        self.n_studies = 0

    def record_empty(self, n_studies: int) -> None:
        """A micro-batch without supervision that was not pushed through the model."""
        self.n_microbatches += 1
        self.n_empty_microbatches += 1
        self.n_studies += int(n_studies)

    def update(self, output: LossOutput, n_studies: int) -> None:
        self.numerators += output.numerators.detach().cpu().numpy().astype(np.float64)
        self.denominators += output.denominators.detach().cpu().numpy().astype(np.float64)
        self.n_microbatches += 1
        self.n_studies += int(n_studies)
        if not output.has_supervision:
            self.n_empty_microbatches += 1

    @property
    def per_class(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            values = np.where(self.denominators > 0, self.numerators / np.maximum(self.denominators, 1e-12), np.nan)
        return values

    @property
    def macro(self) -> float:
        values = self.per_class
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if finite.size else float("nan")

    def summary(self) -> dict:
        per_class = self.per_class
        return {
            "loss": self.macro,
            "n_valid_classes": int(np.isfinite(per_class).sum()),
            "n_microbatches": self.n_microbatches,
            "n_empty_microbatches": self.n_empty_microbatches,
            "n_studies": self.n_studies,
            "per_class": {target: (None if not np.isfinite(v) else float(v)) for target, v in zip(TARGETS, per_class)},
            "class_weight_totals": {target: float(v) for target, v in zip(TARGETS, self.denominators)},
        }
