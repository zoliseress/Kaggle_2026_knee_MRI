"""Focused checks for the failure modes that actually break this pipeline.

They run on synthetic data so no real dataset is needed, and they exercise the real
model and gradient path - nothing that is being checked is mocked away.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from .config import Config, load_config, validate_config
from .constants import EFFICIENTNET_B0_FEATURES, ENCODER_FEATURES, N_TARGETS, SERIES_ID, STUDY_ID, TARGETS, slot_plane
from .dataset import (
    AUGMENT_PARAM_COLUMNS,
    EpochSampler,
    StudyBagDataset,
    SyntheticBagDataset,
    apply_intensity,
    apply_spatial,
    augment_batch,
    bin_centers,
    build_bag,
    collate_studies,
    normalization_stats,
    pack_augment_params,
    resolve_augment_device,
    sample_intensity,
    sample_spatial,
    triplet_indices,
)
from .geometry import assign_plane, canonical_inplane_transform, order_by_position, slice_normal
from .labels import build_from_wide_numeric
from .loss import (
    GlobalNormalizer,
    LossAccumulator,
    WindowNormalizer,
    global_normalized_bce,
    masked_class_normalized_bce,
    window_normalized_bce,
)
from .metrics import evaluate_predictions, soft_roc_auc
from .model import EfficientNetB0MIL, masked_max, masked_mean
from .splits import (
    assert_group_disjoint,
    assert_not_training,
    fold_study_ids,
    load_excluded_training_ids,
    make_fixed_split,
    make_splits,
    verify_cv_split,
    verify_fixed_split,
)
from .train import window_sizes
from .utils import LOG, remove_file_logging


# Smallest input each backbone accepts in the synthetic checks: EfficientNet 32 px (a 1x1 map),
# ResNet-50 64 px (2x2), DINOv2 56 px (4x4 patches; a multiple of 14 - 32 and 64 are refused),
# CoAtNet 64 px (2x2; a multiple of 32, and its attention stages want more than one token),
# ConvNeXt 32 px (1x1).
TINY_SIZES = {
    "efficientnet_b0": 32, "efficientnet_v2_s": 32, "radimagenet_resnet50": 64, "dinov2_vits14": 56,
    "coatnet_0_rw": 64, "coatnet_rmlp_1_rw2": 64,
    "convnext_nano": 32, "convnext_tiny": 32, "convnextv2_tiny": 32,
}
# Smallest input whose map is at least 2 wide, for checks that need two distinct column halves
# or a non-trivial spatial softmax: EfficientNet / ResNet / CoAtNet / ConvNeXt 64 px (2x2), DINOv2 56 px (4x4).
WIDE_SIZES = {
    "efficientnet_b0": 64, "efficientnet_v2_s": 64, "radimagenet_resnet50": 64, "dinov2_vits14": 56,
    "coatnet_0_rw": 64, "coatnet_rmlp_1_rw2": 64,
    "convnext_nano": 64, "convnext_tiny": 64, "convnextv2_tiny": 64,
}


def _backbone_of(cfg: Config) -> str:
    from .constants import DEFAULT_BACKBONE

    return str(cfg.model.get("backbone", DEFAULT_BACKBONE))


def tiny_config(cfg: Config) -> Config:
    """A small, fast variant of the real config: same code path, cheap tensors, the backbone's tiny size."""
    small = cfg.copy()
    small.data.image_size = TINY_SIZES[_backbone_of(cfg)]
    small.data.centers_per_series = 4
    small.model.weights = "none"  # deliberate: the check is about masking, not pretraining
    small.model.head_hidden = 32
    small.train.microbatch_studies = 2
    return small


def wide_config(cfg: Config) -> Config:
    """`cfg` at the smallest size its backbone maps to at least 2x2."""
    wide = cfg.copy()
    wide.data.image_size = WIDE_SIZES[_backbone_of(cfg)]
    return wide


def efficientnet_config(cfg: Config, backbone: str = "efficientnet_b0") -> Config:
    """`cfg` switched to an EfficientNet, for the checks about EfficientNet behaviour itself.

    Everything else (data, train, labels) stays as configured; the RadImageNet-only and the
    CoAtNet (inception) input profiles fall back to the project default, ConvNeXt stochastic
    depth is switched off.
    """
    chosen = cfg.copy()
    chosen.model.backbone = backbone
    chosen.model.weights = "none"
    chosen.model.encoder_trainable = "all"
    chosen.model.drop_path_rate = 0.0  # a ConvNeXt-only option
    chosen.data.image_size = TINY_SIZES[backbone]
    if chosen.data.encoder_normalization in ("radimagenet_torch", "inception"):
        chosen.data.encoder_normalization = "mri_scalar"
    return chosen


def _tiny_model(cfg: Config, seed: int = 0) -> EfficientNetB0MIL:
    torch.manual_seed(seed)
    return EfficientNetB0MIL(
        n_targets=N_TARGETS,
        n_slots=len(cfg.data.series_slots),
        slot_names=list(cfg.data.series_slots),
        weights="none",
        head_hidden=int(cfg.model.head_hidden),
        head_dropout=0.0,
        freeze_bn_running_stats=True,
    ).eval()


def _pool_model(cfg: Config, spatial_pool: str, seed: int = 0) -> EfficientNetB0MIL:
    torch.manual_seed(seed)
    return EfficientNetB0MIL(
        n_targets=N_TARGETS,
        n_slots=len(cfg.data.series_slots),
        slot_names=list(cfg.data.series_slots),
        weights="none",
        head_hidden=int(cfg.model.head_hidden),
        head_dropout=0.0,
        freeze_bn_running_stats=True,
        spatial_pool=spatial_pool,
    ).eval()


def check_attention_pool_starts_as_average(cfg: Config) -> str:
    """The zero-initialised attention scorer is exactly global average pooling.

    So an `attention` run starts from the `avg` model's behaviour and can only move away
    from it by learning; nothing about the comparison depends on a lucky initialisation.
    """
    model = _pool_model(cfg, "attention")
    assert model.attention_pool is not None and model.feature_dim == EFFICIENTNET_B0_FEATURES
    feature_map = torch.randn((5, EFFICIENTNET_B0_FEATURES, 7, 7), generator=torch.Generator().manual_seed(1))

    weights = model.attention_pool.attention_map(feature_map)
    assert torch.allclose(weights.sum(dim=(1, 2)), torch.ones(5), atol=1e-6), "attention weights must sum to 1"
    assert float((weights - 1.0 / 49).abs().max()) < 1e-7, "zero init must give uniform weights"

    pooled = model._pool_map(feature_map)
    averaged = feature_map.mean(dim=(2, 3))
    diff = float((pooled - averaged).abs().max())
    assert diff < 1e-5, f"attention pooling at init differs from the average by {diff:.2e}"

    # ... and it is not frozen there: a non-zero scorer concentrates the weights.
    with torch.no_grad():
        model.attention_pool.score[2].weight.normal_(0.0, 1.0)
    moved = model.attention_pool.attention_map(feature_map)
    assert float((moved - 1.0 / 49).abs().max()) > 1e-3, "the scorer cannot leave uniform weights"
    return f"attention == global average at init (max diff {diff:.1e}), weights sum to 1, and it can move away"


def check_spatial_pool_shapes(cfg: Config) -> str:
    """Each policy produces the documented feature width and a finite forward."""
    batch = _random_batch(cfg, b=2, seed=3)
    widths = {}
    for mode in ("avg", "avgmax", "attention"):
        model = _pool_model(cfg, mode)
        expected = EFFICIENTNET_B0_FEATURES * (2 if mode == "avgmax" else 1)
        assert model.feature_dim == expected, (mode, model.feature_dim, expected)
        assert model.head_in == len(cfg.data.series_slots) * 2 * expected + len(cfg.data.series_slots)
        with torch.no_grad():
            logits = model(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
        assert logits.shape == (2, N_TARGETS) and torch.isfinite(logits).all()
        widths[mode] = model.feature_dim

    avg_model = _pool_model(cfg, "avg")
    with torch.no_grad():
        baseline = avg_model(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
        repeat = _pool_model(cfg, "avg")(
            batch["images"], batch["slice_valid_mask"], batch["series_present_mask"]
        )
    assert torch.equal(baseline, repeat), "the avg path must be unchanged and deterministic"

    try:
        _pool_model(cfg, "centroid")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown spatial_pool must be rejected")
    return f"feature widths {widths}; avg path unchanged; unknown policy rejected"


def check_focal_signal_survives_pooling(cfg: Config) -> str:
    """A finding covering a few cells of the map is diluted by the average, not by max/attention.

    Quantifies the motivation for this experiment: on a 10x10 map (320 px) a 2x2 hot spot
    keeps 4% of its amplitude under the average.
    """
    torch.manual_seed(0)
    feature_map = torch.zeros((1, EFFICIENTNET_B0_FEATURES, 10, 10))
    feature_map[0, :, 4:6, 4:6] = 1.0  # a focal finding, 4 of 100 cells

    averaged = _pool_model(cfg, "avg")._pool_map(feature_map)
    avgmax = _pool_model(cfg, "avgmax")._pool_map(feature_map)
    maxed = avgmax[:, EFFICIENTNET_B0_FEATURES:]
    assert abs(float(averaged.mean()) - 0.04) < 1e-6, float(averaged.mean())
    assert abs(float(maxed.mean()) - 1.0) < 1e-6, "the maximum must keep the full amplitude"

    attention = _pool_model(cfg, "attention")
    with torch.no_grad():  # a scorer that has learned to look at the hot spot
        attention.attention_pool.score[0].weight.zero_()
        attention.attention_pool.score[0].bias.zero_()
        attention.attention_pool.score[2].bias.zero_()
        attention.attention_pool.score[2].weight.zero_()
        attention.attention_pool.score[0].weight[0, 0] = 10.0
        attention.attention_pool.score[2].weight[0, 0] = 10.0
    focused = attention._pool_map(feature_map)
    assert float(focused.mean()) > 0.9, f"a trained attention should recover the signal, got {float(focused.mean())}"
    return (
        f"2x2 hot spot on a 10x10 map: average keeps {float(averaged.mean()):.2f}, "
        f"maximum {float(maxed.mean()):.2f}, focused attention {float(focused.mean()):.2f}"
    )


def check_spatial_pool_training_step(cfg: Config) -> str:
    """Attention pooling trains end to end and its scorer actually receives gradient."""
    import tempfile
    from pathlib import Path

    from .train import run_synthetic

    small = cfg.copy()
    small.model.spatial_pool = "attention"
    small.train.num_workers = 0
    with tempfile.TemporaryDirectory(prefix="knee_mri_pool_") as tmp:
        small.paths.output_dir = str(Path(tmp))
        try:
            run_synthetic(small, n_studies=8, epochs=1, name="pool_check")
            history = pd.read_csv(Path(tmp) / "pool_check" / "history.csv")
        finally:
            remove_file_logging(tmp)  # also on failure: Windows cannot delete an open run.log
    assert np.isfinite(history["train_loss"]).all()

    # A map wider than 1x1 is needed: with a single position the softmax is always 1 and
    # the scorer has, correctly, no gradient. The selftest config would give exactly that.
    wide = wide_config(efficientnet_config(cfg))  # _pool_model is an EfficientNet-B0: 64 px -> 2x2
    model = _pool_model(wide, "attention").train()
    batch = _random_batch(wide, b=2, seed=5)
    logits = model(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
    logits.sum().backward()
    assert model.attention_pool is not None
    grad = model.attention_pool.score[2].weight.grad
    assert grad is not None and float(grad.abs().max()) > 0, "the attention scorer received no gradient"

    flat_cfg = efficientnet_config(cfg)  # B0 at 32 px -> a 1x1 map, whatever backbone is configured
    flat = _pool_model(flat_cfg, "attention").train()
    single = _random_batch(flat_cfg, b=2, seed=5)
    flat(single["images"], single["slice_valid_mask"], single["series_present_mask"]).sum().backward()
    assert float(flat.attention_pool.score[2].weight.grad.abs().max()) == 0.0, "a 1x1 map cannot be re-weighted"
    groups = model.parameter_groups(encoder_lr=1e-4, head_lr=3e-4, weight_decay=0.0)
    head_params = {id(p) for p in groups[1]["params"]}
    assert all(id(p) in head_params for p in model.attention_pool.parameters()), "scorer must use the head lr"
    return "attention pooling trains; scorer gets gradient and sits in the head parameter group"


def _random_batch(cfg: Config, b: int = 2, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    p, s = len(cfg.data.series_slots), int(cfg.data.centers_per_series)
    size = int(cfg.data.image_size)
    images = torch.randn((b, p, s, 3, size, size), generator=g)
    slice_valid = torch.ones((b, p, s), dtype=torch.bool)
    present = torch.ones((b, p), dtype=torch.bool)
    return {"images": images, "slice_valid_mask": slice_valid, "series_present_mask": present}


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------


def check_geometric_sorting() -> str:
    """Slices must be ordered by projected position, not by file order."""
    normal = np.array([0.0, 0.0, 1.0])
    true_z = np.array([-8.0, -4.0, 0.0, 4.0, 8.0])
    shuffled = [3, 0, 4, 1, 2]
    positions = np.stack([[10.0, -5.0, true_z[i]] for i in shuffled])
    result = order_by_position(positions, normal)
    got = positions[result.order][:, 2]
    assert np.allclose(got, np.sort(true_z)), f"expected sorted z, got {got}"
    assert math.isclose(result.median_spacing, 4.0, rel_tol=1e-6), result.median_spacing
    assert not result.irregular

    irregular = np.stack([[0.0, 0.0, z] for z in [0.0, 4.0, 20.0, 24.0]])
    assert order_by_position(irregular, normal).irregular
    return "sorted by projection; irregular spacing detected"


def check_plane_assignment() -> str:
    sagittal = np.array([0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
    coronal = np.array([1.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    axial = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    for iop, expected in ((sagittal, "sagittal"), (coronal, "coronal"), (axial, "axial")):
        got = assign_plane(slice_normal(iop))
        assert got.plane == expected, f"{expected} misassigned as {got.plane}"
        assert not got.ambiguous
    # Tilting the row direction rotates the normal itself: 45 degrees between x and y.
    oblique = np.array([0.7071, 0.7071, 0.0, 0.0, 0.0, -1.0])
    tilted = assign_plane(slice_normal(oblique))
    assert tilted.ambiguous, f"45-degree oblique should be flagged ambiguous (angle {tilted.angle_deg:.1f})"
    return "sagittal/coronal/axial correct; oblique flagged"


def check_inplane_transform() -> str:
    """Canonical orientation must be a pure array reordering, and be idempotent."""
    iop = np.array([0.0, -1.0, 0.0, 0.0, 0.0, 1.0])  # flipped sagittal
    transform = canonical_inplane_transform(iop, "sagittal")
    array = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    out = transform.apply(array)
    assert sorted(out.ravel().tolist()) == sorted(array.ravel().tolist()), "values changed: not a pure reordering"
    assert out.shape[0] == 2
    return f"pure reorder ops={transform.describe()}"


def check_true_neighbour_triplets() -> str:
    """Triplets must come from the original stack, not from the sampled centre list."""
    n_slices, n_centers = 40, 4
    centers, valid = bin_centers(n_slices, n_centers, None)
    image = np.arange(n_slices, dtype=np.float32).reshape(n_slices, 1, 1) * np.ones((1, 2, 2), dtype=np.float32)
    bag, _ = build_bag(image, centers, valid, None)
    for s in range(n_centers):
        centre = int(centers[s])
        expected = [max(0, centre - 1), centre, min(n_slices - 1, centre + 1)]
        got = [float(bag[s, k, 0, 0]) for k in range(3)]
        assert got == [float(v) for v in expected], f"centre {centre}: expected {expected}, got {got}"
    # boundary clipping
    assert triplet_indices(0, 10, None) == (0, 0, 1)
    assert triplet_indices(9, 10, None) == (8, 9, 9)
    # flagged physical gap: repeat the centre instead of crossing it
    gap_ok = np.ones(9, dtype=bool)
    gap_ok[4] = False  # gap between slice 4 and 5
    assert triplet_indices(4, 10, gap_ok) == (3, 4, 4)
    assert triplet_indices(5, 10, gap_ok) == (5, 5, 6)
    return "triplets are true neighbours; boundaries clipped; gaps not crossed"


def check_short_stack_padding() -> str:
    centers, valid = bin_centers(3, 8, None)
    assert valid.sum() == 3 and valid[:3].all() and not valid[3:].any()
    assert list(centers[:3]) == [0, 1, 2]
    centers_t, valid_t = bin_centers(100, 8, np.random.default_rng(0))
    assert valid_t.all() and len(set(centers_t.tolist())) == 8
    assert bin_centers(0, 8, None)[1].sum() == 0
    # In-bin offsets (inference views): 0 / 1 = the bin's first / last slice, always inside the
    # bin and the stack; short stacks are unaffected (every slice is used anyway).
    edges = np.linspace(0, 30, 25)
    first, _ = bin_centers(30, 24, None, offset=0.0)
    last, _ = bin_centers(30, 24, None, offset=1.0)
    for i in range(24):
        lo, hi = int(np.floor(edges[i])), min(30, max(int(np.floor(edges[i])) + 1, int(np.ceil(edges[i + 1]))))
        assert first[i] == lo and last[i] == hi - 1, f"bin {i}: [{lo}, {hi}) -> {first[i]}, {last[i]}"
    assert not np.array_equal(first, last)
    assert np.array_equal(bin_centers(3, 8, None, offset=1.0)[0], centers)
    for bad in (-0.1, 1.5):
        try:
            bin_centers(30, 24, None, offset=bad)
        except ValueError:
            continue
        raise AssertionError(f"offset {bad} accepted")
    return "short stacks pad with invalid centres; full stacks use one centre per bin; offsets stay in-bin"


def check_masked_pooling() -> str:
    features = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    mask = torch.tensor([[[True, True, False], [False, False, False]]])
    mean = masked_mean(features, mask)
    expected_mean = features[0, 0, :2].mean(dim=0)
    assert torch.allclose(mean[0, 0], expected_mean), (mean[0, 0], expected_mean)
    assert torch.equal(mean[0, 1], torch.zeros(4)), "all-masked slot must produce exact zeros"
    maxed = masked_max(features, mask)
    assert torch.allclose(maxed[0, 0], features[0, 0, 1]), "max must ignore masked entries"
    assert torch.equal(maxed[0, 1], torch.zeros(4)), "all-masked max must be exactly zero, not -inf"
    assert torch.isfinite(maxed).all()
    return "mean denominator and max both ignore padding; empty slot -> exact zeros"


def check_padding_invariance(cfg: Config) -> str:
    """Garbage in padded slots must not change the logits."""
    model = _tiny_model(cfg)
    batch = _random_batch(cfg, b=2, seed=1)
    batch["slice_valid_mask"][:, :, 2:] = False
    with torch.no_grad():
        base = model(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
    polluted = batch["images"].clone()
    polluted[:, :, 2:] = 1234.0
    with torch.no_grad():
        other = model(polluted, batch["slice_valid_mask"], batch["series_present_mask"])
    max_diff = float((base - other).abs().max())
    assert max_diff == 0.0, f"padded content leaked into the logits (max diff {max_diff})"
    return "identical logits with polluted padding"


def check_missing_slot(cfg: Config) -> str:
    """A completely missing series must give an exactly zero, finite feature block."""
    model = _tiny_model(cfg)
    batch = _random_batch(cfg, b=1, seed=2)
    batch["series_present_mask"][0, 1] = False
    batch["slice_valid_mask"][0, 1] = False
    with torch.no_grad():
        logits = model(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
    assert torch.isfinite(logits).all(), "missing slot produced non-finite logits"

    # the same study with different content in the absent slot must be identical
    other_images = batch["images"].clone()
    other_images[0, 1] = torch.randn_like(other_images[0, 1])
    with torch.no_grad():
        other = model(other_images, batch["slice_valid_mask"], batch["series_present_mask"])
    assert torch.equal(logits, other), "an absent slot still influenced the output"

    all_missing = torch.zeros_like(batch["series_present_mask"])
    with torch.no_grad():
        empty_logits = model(batch["images"], torch.zeros_like(batch["slice_valid_mask"]), all_missing)
    assert torch.isfinite(empty_logits).all(), "all-missing study produced NaN/inf"
    return "absent slots contribute exactly zero and stay finite"


def check_masked_label_gradients(cfg: Config) -> str:
    """A zero-weight target must produce no gradient in its head output row."""
    model = _tiny_model(cfg)
    batch = _random_batch(cfg, b=2, seed=3)
    logits = model(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
    targets = torch.zeros((2, N_TARGETS))
    weights = torch.zeros((2, N_TARGETS))
    weights[:, 0] = 1.0  # only the first target is supervised
    output = masked_class_normalized_bce(logits, targets, weights)
    output.loss.backward()

    final = model.head[-1]
    grad = final.weight.grad
    assert grad is not None
    assert float(grad[0].abs().sum()) > 0, "supervised target received no gradient"
    assert float(grad[1:].abs().sum()) == 0.0, "unsupervised targets received gradient"
    assert output.n_valid_classes == 1

    # Empty supervision: differentiable zero, no NaN. A fresh forward, because the
    # previous graph was already consumed by backward().
    model.zero_grad(set_to_none=True)
    fresh_logits = model(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
    empty = masked_class_normalized_bce(fresh_logits, targets, torch.zeros_like(weights))
    assert not empty.has_supervision and float(empty.loss.detach()) == 0.0
    empty.loss.backward()
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0.0 for p in model.parameters())
    return "gradients flow only through supervised targets; empty supervision is a differentiable zero"


def check_nan_target_rejected() -> str:
    logits = torch.zeros((2, N_TARGETS), requires_grad=True)
    targets = torch.zeros((2, N_TARGETS))
    targets[0, 0] = float("nan")
    weights = torch.zeros((2, N_TARGETS))
    try:
        masked_class_normalized_bce(logits, targets, weights)
    except ValueError:
        return "non-finite targets rejected even at weight 0"
    raise AssertionError("a NaN target with weight 0 was accepted; multiplying NaN by 0 is not safe")


def check_class_normalization() -> str:
    """L_c = sum(w*BCE)/sum(w), averaged over classes with positive weight."""
    torch.manual_seed(0)
    logits = torch.randn((5, N_TARGETS))
    targets = (torch.rand((5, N_TARGETS)) > 0.5).float()
    weights = (torch.rand((5, N_TARGETS)) > 0.3).float()
    weights[:, 3] = 0.0
    output = masked_class_normalized_bce(logits, targets, weights)

    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    per_class = []
    for c in range(N_TARGETS):
        denominator = float(weights[:, c].sum())
        if denominator > 0:
            per_class.append(float((bce[:, c] * weights[:, c]).sum() / denominator))
    expected = float(np.mean(per_class))
    assert abs(float(output.loss.detach()) - expected) < 1e-6, (float(output.loss.detach()), expected)
    assert output.n_valid_classes == len(per_class)
    return f"matches the reference formula over {len(per_class)}/12 classes"


def check_accumulation(cfg: Config) -> str:
    """Accumulated windows must scale correctly, including a short final window."""
    assert window_sizes(8, 4) == [4] * 8
    assert window_sizes(6, 4) == [4, 4, 4, 4, 2, 2], window_sizes(6, 4)
    assert window_sizes(3, 8) == [3, 3, 3]

    model = _tiny_model(cfg)
    batch = _random_batch(cfg, b=4, seed=5)
    targets = (torch.rand((4, N_TARGETS), generator=torch.Generator().manual_seed(6)) > 0.5).float()
    weights = torch.ones((4, N_TARGETS))

    logits = model(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
    full = masked_class_normalized_bce(logits, targets, weights)
    full.loss.backward()
    reference = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

    model.zero_grad(set_to_none=True)
    for start in (0, 2):
        chunk = {k: v[start : start + 2] for k, v in batch.items()}
        chunk_logits = model(chunk["images"], chunk["slice_valid_mask"], chunk["series_present_mask"])
        out = masked_class_normalized_bce(chunk_logits, targets[start : start + 2], weights[start : start + 2])
        (out.loss / 2.0).backward()
    accumulated = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

    diffs = [float((reference[k] - accumulated[k]).abs().max()) for k in reference]
    # With uniform weights the two policies coincide; that is what makes this a usable check.
    assert max(diffs) < 1e-4, f"accumulated gradients diverge (max {max(diffs):.2e})"
    return f"window scaling correct; grad diff {max(diffs):.2e} with uniform weights"


def _window_case(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eight one-study rows: mixed 0/1 weights, one 0.2 weak label, one class never supervised."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn((8, N_TARGETS), generator=g, dtype=torch.float32)
    targets = (torch.rand((8, N_TARGETS), generator=g) > 0.5).float()
    weights = (torch.rand((8, N_TARGETS), generator=g) > 0.35).float()
    weights[:, 5] = 0.0  # a class with no valid label anywhere in the window
    weights[2, 0] = 0.2  # one weak label
    weights[6] = 0.0  # one study without any supervision
    return logits, targets, weights


def check_window_loss_matches_full() -> str:
    """Window policy: eight accumulated one-study contributions == one full-window loss.

    Checked on fixed logits in FP32, value and logit gradient, against the reference
    formula L = 1/|C+| * sum_c sum_i m*w*BCE / D_c with D_c = mask counts.
    """
    logits, targets, weights = _window_case()
    normalizer = WindowNormalizer.from_weights(weights)

    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    mask = (weights > 0).float()
    counts = mask.sum(dim=0)
    active = counts > 0
    expected = float(((bce * mask * weights).sum(dim=0)[active] / counts[active]).sum() / active.sum())

    full_logits = logits.clone().requires_grad_(True)
    full = window_normalized_bce(full_logits, targets, weights, normalizer)
    full.loss.backward()

    part_logits = logits.clone().requires_grad_(True)
    total = 0.0
    for i in range(8):
        part = window_normalized_bce(part_logits[i : i + 1], targets[i : i + 1], weights[i : i + 1], normalizer)
        part.loss.backward()
        total += float(part.loss.detach())

    assert abs(float(full.loss.detach()) - expected) < 1e-6, (float(full.loss), expected)
    assert abs(total - expected) < 1e-6, (total, expected)
    diff = float((full_logits.grad - part_logits.grad).abs().max())
    assert diff < 1e-7, f"accumulated logit gradient differs from the full-window one by {diff:.2e}"
    assert float(full_logits.grad[weights == 0].abs().max()) == 0.0, "an excluded label received gradient"
    return f"accumulated == full window (value and logit gradient, max diff {diff:.1e}); excluded labels get 0"


def check_window_weight_scaling() -> str:
    """Changing one label's weight 1.0 -> 0.2 scales exactly its own contribution by 1/5."""
    logits, targets, weights = _window_case(seed=1)
    weights[2, 0] = 1.0
    grads = []
    normalizers = []
    for w in (1.0, 0.2):
        cell_weights = weights.clone()
        cell_weights[3, 1] = w if weights[3, 1] > 0 else 0.0
        if weights[3, 1] == 0:
            cell_weights[3, 1] = w  # make sure the probed cell is a valid label
        normalizer = WindowNormalizer.from_weights(cell_weights)
        normalizers.append(normalizer)
        leaf = logits.clone().requires_grad_(True)
        window_normalized_bce(leaf, targets, cell_weights, normalizer).loss.backward()
        grads.append(leaf.grad.clone())

    assert torch.equal(normalizers[0].counts, normalizers[1].counts), "denominators moved with the weight"
    assert normalizers[0].n_active == normalizers[1].n_active
    ratio = float(grads[1][3, 1] / grads[0][3, 1])
    assert abs(ratio - 0.2) < 1e-6, f"probed cell gradient ratio {ratio}, expected 0.2"
    others = torch.ones_like(grads[0], dtype=torch.bool)
    others[3, 1] = False
    assert torch.equal(grads[0][others], grads[1][others]), "other labels' gradients changed"
    return "weight 1.0 -> 0.2 scales that label's gradient by exactly 0.2; denominators and other labels unchanged"


def check_window_empty_cases() -> str:
    """Empty micro-batch, fully empty window and a short final window."""
    logits, targets, weights = _window_case(seed=2)
    normalizer = WindowNormalizer.from_weights(weights)

    empty_leaf = logits[6:7].clone().requires_grad_(True)
    empty = window_normalized_bce(empty_leaf, targets[6:7], weights[6:7], normalizer)
    assert not empty.has_supervision and float(empty.loss.detach()) == 0.0
    empty.loss.backward()
    assert float(empty_leaf.grad.abs().max()) == 0.0, "an empty micro-batch produced gradient"

    zero_weights = torch.zeros_like(weights)
    dead = WindowNormalizer.from_weights(zero_weights)
    assert dead.n_active == 0 and not dead.has_supervision
    dead_leaf = logits.clone().requires_grad_(True)
    out = window_normalized_bce(dead_leaf, targets, zero_weights, dead)
    out.loss.backward()
    assert float(out.loss.detach()) == 0.0 and float(dead_leaf.grad.abs().max()) == 0.0

    short = WindowNormalizer.from_weights(weights[:3])
    assert torch.equal(short.counts, (weights[:3] > 0).sum(dim=0).float())
    parts = sum(
        float(window_normalized_bce(logits[i : i + 1], targets[i : i + 1], weights[i : i + 1], short).loss)
        for i in range(3)
    )
    whole = float(window_normalized_bce(logits[:3], targets[:3], weights[:3], short).loss)
    assert abs(parts - whole) < 1e-6, (parts, whole)

    try:
        window_normalized_bce(logits, targets, weights, short)
    except ValueError:
        pass
    else:
        raise AssertionError("a micro-batch with more labels than its window counted must be rejected")
    return "empty micro-batch and empty window contribute exactly 0; short final window uses its own counts"


def check_global_loss_matches_formula() -> str:
    """Global policy: fixed training-set D_c; accumulated micro-batches == one full window.

    D_c = K * sum_i w_ic / N over a synthetic training set; the window loss must equal
    1/|C+| * sum_c sum_i w*BCE / D_c in value and logit gradient, and a class without any
    training weight is excluded from |C+|.
    """
    g = torch.Generator().manual_seed(3)
    train_weights = (torch.rand((40, N_TARGETS), generator=g) > 0.4).float()
    train_weights[:, 7] = 0.0  # never supervised in training
    train_weights[::3, 4] = 0.2  # some weak labels
    normalizer = GlobalNormalizer.from_training_weights(train_weights.numpy(), window_studies=8)
    expected_d = 8.0 * train_weights.sum(dim=0) / 40.0
    assert torch.allclose(normalizer.denominators, expected_d), "D_c != K * mean training weight"
    assert normalizer.n_active == N_TARGETS - 1 and float(normalizer.denominators[7]) == 0.0

    logits, targets, weights = _window_case(seed=4)
    weights[:, 7] = 0.0
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    active = expected_d > 0
    expected = float(((bce * weights).sum(dim=0)[active] / expected_d[active]).sum() / active.sum())

    full_logits = logits.clone().requires_grad_(True)
    full = global_normalized_bce(full_logits, targets, weights, normalizer)
    full.loss.backward()
    part_logits = logits.clone().requires_grad_(True)
    total = 0.0
    for i in range(8):
        part = global_normalized_bce(part_logits[i : i + 1], targets[i : i + 1], weights[i : i + 1], normalizer)
        part.loss.backward()
        total += float(part.loss.detach())
    assert abs(float(full.loss.detach()) - expected) < 1e-6, (float(full.loss), expected)
    assert abs(total - expected) < 1e-6, (total, expected)
    diff = float((full_logits.grad - part_logits.grad).abs().max())
    assert diff < 1e-7, f"accumulated logit gradient differs from the full window by {diff:.2e}"
    assert float(full_logits.grad[weights == 0].abs().max()) == 0.0, "an excluded label received gradient"

    bad = weights.clone()
    bad[0, 7] = 1.0
    try:
        global_normalized_bce(logits, targets, bad, normalizer)
    except ValueError:
        pass
    else:
        raise AssertionError("weight on a class the training set never supervised must be rejected")
    return f"accumulated == full window == formula (logit gradient max diff {diff:.1e}); unsupervised class excluded"


def check_global_no_dilution() -> str:
    """Switching a weak label ON adds exactly its own term; a weak-only window is not rescaled.

    This is what the window policy cannot do: there a new label raises D_c and dilutes the
    class's other labels, and a within-window weight sum would cancel a lone 0.2 back to 1.
    """
    logits, targets, weights = _window_case(seed=5)
    weights[:, 3] = 0.0
    weights[0, 3] = 1.0  # one real label of class 3
    normalizer = GlobalNormalizer.from_training_weights(torch.ones((10, N_TARGETS)).numpy(), window_studies=8)

    grads = []
    for weak in (0.0, 0.2):
        w = weights.clone()
        w[4, 3] = weak  # a weak label switched on (or not) in the same window
        leaf = logits.clone().requires_grad_(True)
        global_normalized_bce(leaf, targets, w, normalizer).loss.backward()
        grads.append(leaf.grad.clone())
    others = torch.ones_like(grads[0], dtype=torch.bool)
    others[4, 3] = False
    assert torch.equal(grads[0][others], grads[1][others]), "switching a weak label on changed other labels' gradients"
    assert float(grads[0][4, 3]) == 0.0 and float(grads[1][4, 3]) != 0.0

    # weak-only window: one lone label with weight 0.2 vs 1.0 -> gradient ratio exactly 0.2
    lone = []
    for w_value in (1.0, 0.2):
        w = torch.zeros_like(weights)
        w[2, 6] = w_value
        leaf = logits.clone().requires_grad_(True)
        global_normalized_bce(leaf, targets, w, normalizer).loss.backward()
        lone.append(float(leaf.grad[2, 6]))
    ratio = lone[1] / lone[0]
    assert abs(ratio - 0.2) < 1e-6, f"lone weak label gradient ratio {ratio}, expected 0.2 (no cancellation)"

    # contrast: under the window policy the same switch dilutes the real label
    w_off, w_on = weights.clone(), weights.clone()
    w_on[4, 3] = 0.2
    g_window = []
    for w in (w_off, w_on):
        leaf = logits.clone().requires_grad_(True)
        window_normalized_bce(leaf, targets, w, WindowNormalizer.from_weights(w)).loss.backward()
        g_window.append(float(leaf.grad[0, 3]))
    dilution = g_window[1] / g_window[0]
    assert abs(dilution - 0.5) < 1e-6, f"window policy dilution {dilution}, expected 0.5 (D_c 1 -> 2)"
    return "weak label on: other gradients bit-identical; lone 0.2 label = 0.2x; window policy would halve the real label"


def check_global_training_step(cfg: Config) -> str:
    """The trainer's global path runs end to end on synthetic data and records its objective."""
    import tempfile
    from pathlib import Path

    from .train import run_synthetic

    small = cfg.copy()
    small.train.loss_normalization = "global"
    small.train.accumulation_steps = 3  # 8 studies -> windows of 3, 3 and a short 2
    small.train.microbatch_studies = 1
    small.train.num_workers = 0
    with tempfile.TemporaryDirectory(prefix="knee_mri_global_") as tmp:
        small.paths.output_dir = str(Path(tmp))
        try:
            run_synthetic(small, n_studies=8, epochs=2, name="global_check")
            history = pd.read_csv(Path(tmp) / "global_check" / "history.csv")
        finally:
            remove_file_logging(tmp)
    assert history["train_objective"].notna().all() and np.isfinite(history["train_objective"]).all()
    assert int(history["optimizer_steps"].iloc[0]) <= 3
    # Synthetic validation labels are the reference itself (unit weights): the training-label loss
    # through the logits and the reference BCE through the clipped sigmoids are the same number.
    assert np.allclose(history["val_loss"], history["val_ref_bce"], atol=1e-4), history[["val_loss", "val_ref_bce"]]
    return f"synthetic run with the global policy: {len(history)} epochs, objective {history['train_objective'].round(4).tolist()}"


def check_window_training_step(cfg: Config) -> str:
    """The trainer's window path runs end to end and records the minimised objective."""
    import tempfile
    from pathlib import Path

    from .train import run_synthetic

    small = cfg.copy()
    small.train.loss_normalization = "window"
    small.train.accumulation_steps = 3  # 8 studies -> windows of 3, 3 and a short 2
    small.train.microbatch_studies = 1
    small.train.num_workers = 0
    with tempfile.TemporaryDirectory(prefix="knee_mri_window_") as tmp:
        small.paths.output_dir = str(Path(tmp))
        try:
            run_synthetic(small, n_studies=8, epochs=2, name="window_check")
            history = pd.read_csv(Path(tmp) / "window_check" / "history.csv")
        finally:
            remove_file_logging(tmp)
    assert history["train_objective"].notna().all() and np.isfinite(history["train_objective"]).all()
    assert int(history["optimizer_steps"].iloc[0]) <= 3
    return f"synthetic run with the window policy: {len(history)} epochs, objective {history['train_objective'].round(4).tolist()}"


def check_epoch_loss_accounting() -> str:
    accumulator = LossAccumulator()
    logits = torch.zeros((2, N_TARGETS))
    targets = torch.zeros((2, N_TARGETS))
    weights = torch.zeros((2, N_TARGETS))
    weights[0, 0] = 2.0
    accumulator.update(masked_class_normalized_bce(logits, targets, weights), n_studies=2)
    accumulator.update(masked_class_normalized_bce(logits, targets, torch.zeros_like(weights)), n_studies=2)
    summary = accumulator.summary()
    assert summary["n_empty_microbatches"] == 1
    assert abs(summary["loss"] - math.log(2)) < 1e-6, summary["loss"]
    assert summary["n_valid_classes"] == 1
    return "epoch loss uses accumulated numerators/denominators, not batch averages"


def check_sigmoid_outputs(cfg: Config) -> str:
    model = _tiny_model(cfg)
    batch = _random_batch(cfg, b=2, seed=7)
    with torch.no_grad():
        logits = model(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
    assert logits.shape == (2, N_TARGETS)
    assert (logits.abs() > 0).any(), "logits look degenerate"
    probabilities = torch.sigmoid(logits)
    assert float(probabilities.min()) >= 0.0 and float(probabilities.max()) <= 1.0
    assert not torch.allclose(probabilities.sum(dim=1), torch.ones(2)), "outputs must not be a softmax over targets"
    return "model returns logits; sigmoid gives per-target probabilities"


def check_single_class_auc() -> str:
    scores = np.random.default_rng(0).random((20, N_TARGETS))
    reference = np.zeros((20, N_TARGETS))
    valid = np.ones((20, N_TARGETS), dtype=bool)
    reference[:10, 0] = 1.0  # only target 0 has both classes
    table, summary = evaluate_predictions(scores, reference, valid)
    assert np.isfinite(table.loc[0, "roc_auc"]), "target with both classes should have an AUC"
    assert table.loc[1:, "roc_auc"].isna().all(), "single-class targets must be NA"
    assert (table.loc[1:, "roc_auc"] == 0.5).sum() == 0, "0.5 must never be substituted"
    assert summary["n_defined_targets"] == 1 and summary["defined_fraction"] == f"1/{N_TARGETS}"
    return "single-class targets return NA; macro reports n_defined/12"


def check_label_join() -> str:
    """Labels must join by StudyInstanceUID, never by row position."""
    cfg = load_config(resolve=False)
    study_ids = ["s1", "s2", "s3"]
    frame = pd.DataFrame(
        {
            STUDY_ID: ["s3", "s1"],  # deliberately different order, s2 absent
            **{target: [1.0, 0.0] for target in TARGETS},
        }
    )
    table = build_from_wide_numeric(frame, study_ids, cfg, source="unit-test")
    assert table.targets[0, 0] == 0.0 and table.weights[0, 0] == 1.0, "s1 mis-joined"
    assert table.weights[1, 0] == 0.0, "absent study must stay unknown, not negative"
    assert table.targets[2, 0] == 1.0 and table.weights[2, 0] == 1.0, "s3 mis-joined"
    values, valid = table.binary_reference()
    assert not valid[1].any(), "unknown cells must be invalid in the evaluation reference"
    return "join is key-based; missing rows stay unknown"


def check_empty_numeric_is_unknown() -> str:
    cfg = load_config(resolve=False)
    frame = pd.DataFrame({STUDY_ID: ["s1"], **{t: [np.nan] for t in TARGETS}})
    table = build_from_wide_numeric(frame, ["s1"], cfg, source="unit-test")
    assert float(table.weights.sum()) == 0.0, "empty cells became supervised"
    assert np.isfinite(table.targets).all(), "placeholders must be finite"
    return "empty numeric cells are unknown with weight 0"


def _details_frame(statuses: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """One study, every target given as (status, basis)."""
    return pd.DataFrame(
        [{STUDY_ID: "s1", "target": t, "status": s, "basis": b} for t, (s, b) in statuses.items()]
    )


def check_unmentioned_weight_per_target() -> str:
    from .labels import KIND_NOT_MENTIONED, KIND_UNCERTAIN, build_from_details

    statuses = {t: ("not_mentioned", "not_mentioned") for t in TARGETS}
    statuses["ACL"] = ("positive", "meets_criteria")
    statuses["Effusion"] = ("uncertain", "insufficient_detail")
    frame = _details_frame(statuses)

    tables = []
    for overrides in ({}, {"unmentioned_weight": 0.2}, {"unmentioned_weight": 0.2, "unmentioned_weight_per_target": {"Synovitis": 0.0}}):
        cfg = load_config(resolve=False)
        for key, value in overrides.items():
            cfg.labels[key] = value
        tables.append(build_from_details(frame, ["s1"], cfg))

    syn, frac, eff = TARGETS.index("Synovitis"), TARGETS.index("Fracture"), TARGETS.index("Effusion")
    assert tables[0].weights[0, frac] == 0.0, "default policy must keep not_mentioned unsupervised"
    assert tables[1].weights[0, frac] == np.float32(0.2) and tables[1].targets[0, frac] == 0.0
    assert tables[2].weights[0, syn] == 0.0 and tables[2].weights[0, frac] == np.float32(0.2), "per-target override ignored"
    assert all(t.kinds[0, frac] == KIND_NOT_MENTIONED for t in tables)
    assert all(t.kinds[0, eff] == KIND_UNCERTAIN and t.weights[0, eff] == 0.0 for t in tables), "uncertain leaked"
    references = [t.binary_reference() for t in tables]
    for values, valid in references[1:]:
        assert np.array_equal(valid, references[0][1]) and np.array_equal(values, references[0][0]), (
            "the binary reference changed with the not_mentioned training weight"
        )
    return "per-target not_mentioned weights applied; uncertain stays out; reference unchanged"


def check_details_soft_roundtrip(tmp_dir) -> str:
    """The notebooks' `p_positive` survives the merge and loads the same as the direct export."""
    from pathlib import Path

    from .labels import KIND_BORDERLINE, KIND_NEGATIVE, KIND_NOT_MENTIONED, KIND_POSITIVE, KIND_SOFT, build_from_details
    from .merge_label_details import merge_label_details
    from .schema import read_id_csv

    tmp = Path(tmp_dir) / "details_soft"
    tmp.mkdir(parents=True, exist_ok=True)
    ids = ["s1", "s2", "r1"]
    cells = {t: ("not_mentioned", "not_mentioned", np.nan) for t in TARGETS}
    cells.update(
        {
            "ACL": ("positive", "meets_criteria", 0.7),
            "MCL": ("uncertain", "insufficient_detail", 0.4),
            "Medial Meniscus": ("negative", "borderline", 0.1),
            "Lateral Meniscus": ("negative", "explicit_absence", 0.05),
        }
    )
    rows = [  # the notebook export: `label` and `p_positive`
        {STUDY_ID: sid, "label": t, "status": s, "basis": b, "p_positive": p}
        for sid in ("s1", "s2")
        for t, (s, b, p) in cells.items()
    ]
    rows += [{STUDY_ID: "r1", "label": t, "status": "positive", "basis": "meets_criteria", "p_positive": 0.9} for t in TARGETS]
    llm = pd.DataFrame(rows)
    llm_path = tmp / "labels_details.csv"
    llm.to_csv(llm_path, index=False)
    pd.DataFrame({STUDY_ID: ids}).to_csv(tmp / "train.csv", index=False)
    pd.DataFrame({STUDY_ID: ["r1"], **{t: [0] for t in TARGETS}}).to_csv(tmp / "reference.csv", index=False)

    cfg = load_config(resolve=False)
    cfg.paths.train_csv = str(tmp / "train.csv")
    cfg.paths.reference_csv = str(tmp / "reference.csv")
    cfg.labels.allow_soft_targets = True
    cfg.labels.borderline_policy = "exclude"
    cfg.labels.uncertain_weight = 0.0
    merged_path = merge_label_details(cfg, [llm_path], tmp / "labels_details_all.csv")
    assert "soft_target" in read_id_csv(merged_path).columns, "merge dropped the soft column"

    direct = build_from_details(read_id_csv(llm_path), ids, cfg)
    merged = build_from_details(read_id_csv(merged_path), ids, cfg)
    for name in ("targets", "weights", "kinds"):
        assert np.array_equal(getattr(direct, name)[:2], getattr(merged, name)[:2]), f"merge changed {name}"
    acl, mcl, med, lat, frac = (TARGETS.index(t) for t in ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Fracture"))
    # Uncertain (uncertain_weight 0) and borderline (policy exclude) cells train on their soft value.
    for c, value in ((acl, 0.7), (mcl, 0.4), (med, 0.1), (lat, 0.05)):
        assert merged.kinds[0, c] == KIND_SOFT and merged.targets[0, c] == np.float32(value) and merged.weights[0, c] == 1.0
    assert merged.kinds[0, frac] == KIND_NOT_MENTIONED and merged.weights[0, frac] == 0.0
    assert (merged.kinds[2] == KIND_NEGATIVE).all() and not merged.targets[2].any() and (merged.weights[2] == 1.0).all(), (
        "the radiologist reference must replace the LLM soft value"
    )

    cfg.labels.allow_soft_targets = False
    hard = build_from_details(read_id_csv(merged_path), ids, cfg)
    assert hard.kinds[0, acl] == KIND_POSITIVE and hard.targets[0, acl] == 1.0, "allow_soft_targets=false must use the status"
    assert hard.kinds[0, med] == KIND_BORDERLINE and hard.weights[0, med] == 0.0, "status path must keep the exclude policy"

    for broken, where in (
        (llm.assign(soft_target=llm["p_positive"]), "both soft columns"),
        (llm.assign(p_positive=llm["p_positive"].fillna(0.5)), "soft value on not_mentioned"),
        (llm.assign(p_positive=llm["p_positive"].replace(0.7, np.inf)), "infinite soft value"),
    ):
        path = tmp / "broken.csv"
        broken.to_csv(path, index=False)
        for call in (
            lambda: merge_label_details(cfg, [path], tmp / "broken_all.csv"),
            lambda: build_from_details(read_id_csv(path), ids, cfg),
        ):
            try:
                call()
            except ValueError:
                continue
            raise AssertionError(f"{where} was accepted")
    return "p_positive survives the merge as soft_target; direct == merged; borderline/uncertain soft kept; reference wins"


def check_details_missing_values(tmp_dir) -> str:
    """Empty status/basis cells (NaN from a CSV, pd.NA in memory) are unknown, never a crash."""
    from pathlib import Path

    from .labels import KIND_NEGATIVE, KIND_POSITIVE, KIND_SOFT, KIND_UNKNOWN, build_from_details
    from .merge_label_details import merge_label_details
    from .schema import read_id_csv

    tmp = Path(tmp_dir) / "details_missing"
    tmp.mkdir(parents=True, exist_ok=True)
    acl, mcl, med, lat = range(4)
    cells = {t: ("not_mentioned", "not_mentioned", None) for t in TARGETS}
    cells.update(
        {
            TARGETS[acl]: ("negative", None, None),  # basis cell empty -> masked
            TARGETS[mcl]: ("negative", None, 0.1),  # ... unless a soft value is supplied
            TARGETS[med]: ("positive", None, None),  # a positive does not need a basis
        }
    )
    rows = [{STUDY_ID: "s1", "label": t, "status": s, "basis": b, "p_positive": p} for t, (s, b, p) in cells.items()]
    rows += [{STUDY_ID: "s2", "label": t, "status": None, "basis": None, "p_positive": None} for t in TARGETS]  # unprocessed
    frame = pd.DataFrame(rows)
    path = tmp / "labels_details.csv"
    frame.to_csv(path, index=False)

    cfg = load_config(resolve=False)
    cfg.labels.allow_soft_targets = True
    from_csv = build_from_details(read_id_csv(path), ["s1", "s2"], cfg)
    in_memory = build_from_details(frame.astype({"status": "string", "basis": "string"}), ["s1", "s2"], cfg)
    for table in (from_csv, in_memory):
        assert table.kinds[0, acl] == KIND_UNKNOWN and table.weights[0, acl] == 0.0, "empty basis became a firm negative"
        assert table.kinds[0, mcl] == KIND_SOFT and table.weights[0, mcl] == 1.0
        assert table.kinds[0, med] == KIND_POSITIVE and table.weights[0, med] == 1.0
        assert (table.kinds[1] == KIND_UNKNOWN).all() and not table.weights[1].any(), "an unprocessed study was supervised"

    no_basis = build_from_details(frame.drop(columns="basis"), ["s1"], cfg)
    assert no_basis.kinds[0, acl] == KIND_NEGATIVE and no_basis.weights[0, acl] == 1.0, "no basis column: firm negative"

    pd.DataFrame({STUDY_ID: ["s1", "s2"]}).to_csv(tmp / "train.csv", index=False)
    cfg.paths.train_csv = str(tmp / "train.csv")
    cfg.paths.reference_csv = None
    try:
        merge_label_details(cfg, [path], tmp / "merged.csv")
    except ValueError as error:
        assert "no status" in str(error), f"unclear merge error: {error}"
    else:
        raise AssertionError("the merge accepted rows without a status")
    return "empty basis masks a negative; empty status is unknown; no basis column keeps firm negatives"


def check_statuses_all_targets(tmp_dir) -> str:
    """Every target column of a wide status export is read, spaces and apostrophes included."""
    from pathlib import Path

    from .labels import KIND_NEGATIVE, KIND_NOT_MENTIONED, KIND_POSITIVE, KIND_UNCERTAIN, build_label_table

    cycle = [("positive", KIND_POSITIVE, 1.0), ("negative", KIND_NEGATIVE, 1.0),
             ("uncertain", KIND_UNCERTAIN, 0.0), ("not_mentioned", KIND_NOT_MENTIONED, 0.0)]
    rows = [
        {STUDY_ID: "s1", **{t: "positive" for t in TARGETS}},
        {STUDY_ID: "s2", **{t: cycle[c % 4][0] for c, t in enumerate(TARGETS)}},
    ]
    frame = pd.DataFrame(rows)[[STUDY_ID, *reversed(TARGETS)]]  # column order must not matter
    path = Path(tmp_dir) / "labels_statuses.csv"
    frame.to_csv(path, index=False)

    cfg = load_config(resolve=False)
    cfg.labels.source = "auto"
    cfg.labels.borderline_policy = "as_negative"
    cfg.labels.uncertain_weight = 0.0
    cfg.labels.unmentioned_weight = 0.0
    cfg.labels.unmentioned_weight_per_target = {}
    cfg.paths.labels_details_csv = None
    cfg.paths.labels_statuses_csv = str(path)
    table = build_label_table(cfg, ["s1", "s2"])
    assert table.source == "statuses"
    lost = [t for c, t in enumerate(TARGETS) if table.weights[0, c] != 1.0 or table.kinds[0, c] != KIND_POSITIVE]
    assert not lost, f"all-positive study lost targets {lost}"
    for c, target in enumerate(TARGETS):
        _, kind, weight = cycle[c % 4]
        assert table.kinds[1, c] == kind and table.weights[1, c] == weight, f"{target}: {table.kinds[1, c]} != {kind}"
    return "all 12 status columns read by name, including 'Medial Meniscus' and \"Baker's\""


def check_wide_soft_source(tmp_dir) -> str:
    """Under borderline_policy='exclude' a soft wide export keeps its borderline soft values."""
    from pathlib import Path

    from .labels import KIND_SOFT, build_label_table

    tmp = Path(tmp_dir) / "wide_soft"
    tmp.mkdir(parents=True, exist_ok=True)
    acl, mcl, med = (TARGETS.index(t) for t in ("ACL", "MCL", "Medial Meniscus"))

    def write(name: str, row: dict[str, float]) -> str:
        path = tmp / name
        pd.DataFrame([{STUDY_ID: "s1", **{t: row.get(t, np.nan) for t in TARGETS}}]).to_csv(path, index=False)
        return str(path)

    # The soft notebook: positive 0.7, uncertain 0.4, borderline negative 0.1; the exclude file blanks the borderline cell.
    soft = {"ACL": 0.7, "MCL": 0.4, "Medial Meniscus": 0.1}
    binary = {"ACL": 1.0, "Medial Meniscus": 0.0}  # binary exports: uncertain blank, borderline written as 0
    exports = {
        "soft": (write("soft.csv", soft), write("soft_excl.csv", {**soft, "Medial Meniscus": np.nan})),
        "binary": (write("bin.csv", binary), write("bin_excl.csv", {**binary, "Medial Meniscus": np.nan})),
    }
    for source in ("wide", "auto"):
        for kind, (plain, excluded) in exports.items():
            cfg = load_config(resolve=False)
            cfg.labels.source = source
            cfg.labels.borderline_policy = "exclude"
            cfg.labels.allow_soft_targets = True
            cfg.paths.labels_details_csv = None
            cfg.paths.labels_statuses_csv = None
            cfg.paths.labels_predictions_csv = plain
            cfg.paths.labels_predictions_exclude_borderline_csv = excluded
            table = build_label_table(cfg, ["s1"])
            if kind == "soft":
                assert table.source == "wide", f"{source}: a soft export must win over its exclude_borderline copy"
                for c, value in ((acl, 0.7), (mcl, 0.4), (med, 0.1)):
                    assert table.kinds[0, c] == KIND_SOFT and table.targets[0, c] == np.float32(value)
                    assert table.weights[0, c] == 1.0, f"{source}: soft cell {TARGETS[c]} lost its weight"
            else:
                assert table.source == "wide_exclude_borderline", f"{source}: a binary export must honour exclude"
                assert table.weights[0, med] == 0.0, f"{source}: a binary borderline 0 entered training"
    return "soft wide export keeps borderline/uncertain soft values; binary export still excludes borderline"


def check_frozen_reference_roundtrip(tmp_dir) -> str:
    from pathlib import Path

    from .labels import build_from_wide_numeric
    from .train import EvaluationReference

    cfg = load_config(resolve=False)
    rng = np.random.default_rng(0)
    ids = [f"s{i}" for i in range(6)]
    raw = rng.choice([0.0, 1.0, np.nan], size=(len(ids), len(TARGETS)))
    frame = pd.DataFrame(raw, columns=TARGETS)
    frame.insert(0, STUDY_ID, ids)
    table = build_from_wide_numeric(frame, ids, cfg, source="unit-test")
    path = Path(tmp_dir) / "frozen_reference.csv"
    EvaluationReference.from_table(table, ids).save(path)

    subset = ["s4", "s1", "s3"]
    loaded = EvaluationReference.from_csv(path, subset)
    direct = EvaluationReference.from_table(table, subset)
    assert loaded.study_ids == subset
    assert np.array_equal(loaded.values, direct.values) and np.array_equal(loaded.valid, direct.valid)
    try:
        EvaluationReference.from_csv(path, ["missing"])
    except KeyError:
        pass
    else:
        raise AssertionError("a study absent from the frozen reference must be an error")
    return "frozen reference reloads key-aligned and refuses unknown studies"


def check_label_mask(tmp_dir) -> str:
    """paths.label_mask_csv: every target of a listed study is unknown/weight 0, also in a frozen reference."""
    from pathlib import Path

    from .labels import KIND_UNKNOWN, apply_label_mask, build_from_wide_numeric, load_label_mask_ids
    from .train import EvaluationReference

    cfg = load_config(resolve=False)
    ids = ["s0", "s1", "s2"]
    frame = pd.DataFrame(np.ones((len(ids), len(TARGETS))), columns=TARGETS)
    frame.insert(0, STUDY_ID, ids)
    frozen = Path(tmp_dir) / "mask" / "frozen.csv"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    EvaluationReference.from_table(build_from_wide_numeric(frame, ids, cfg, source="unit-test"), ids).save(frozen)

    mask_csv = Path(tmp_dir) / "mask" / "mask.csv"
    pd.DataFrame({STUDY_ID: ["s1", "not_in_table"], "reason": ["both knees", "x"]}).to_csv(mask_csv, index=False)
    cfg.paths.label_mask_csv = str(mask_csv)
    assert load_label_mask_ids(cfg) == {"s1", "not_in_table"}
    table = apply_label_mask(build_from_wide_numeric(frame, ids, cfg, source="unit-test"), load_label_mask_ids(cfg))
    assert (table.weights[1] == 0).all() and (table.kinds[1] == KIND_UNKNOWN).all() and (table.targets[1] == 0).all()
    assert (table.weights[[0, 2]] == 1).all() and table.policy["n_label_masked_studies"] == 1
    assert not EvaluationReference.from_table(table, ids).valid[1].any()

    cfg.paths.frozen_reference_csv = str(frozen)  # written before the mask: still all valid on disk
    reference = EvaluationReference.for_run(cfg, table, ["s2", "s1"])
    assert reference.valid[0].all() and not reference.valid[1].any(), "the frozen file must honour the mask too"

    cfg.paths.label_mask_csv = str(Path(tmp_dir) / "mask" / "missing.csv")
    try:
        load_label_mask_ids(cfg)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("a configured but missing label mask must be an error")
    cfg.paths.label_mask_csv = None
    assert load_label_mask_ids(cfg) == set()
    return "listed studies -> all targets unknown at weight 0, in the table and in a frozen reference; missing file refused"


def check_soft_auc_matches_roc_auc() -> str:
    """On a binary reference the soft ROC-AUC is exactly sklearn's ROC-AUC, ties included."""
    from sklearn.metrics import roc_auc_score

    for seed in range(5):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, size=60).astype(float)
        y[:2] = (0.0, 1.0)
        scores = rng.integers(0, 8, size=60) / 7.0  # coarse grid -> many tied scores
        assert math.isclose(soft_roc_auc(y, scores), roc_auc_score(y.astype(int), scores), abs_tol=1e-12)
    assert np.isnan(soft_roc_auc(np.ones(5), np.arange(5.0))), "a single-class reference must be NA"
    return "soft ROC-AUC equals ROC-AUC on binary references, NA on a single class"


def check_subset_metrics(tmp_dir) -> str:
    """Per-subset validation report: the full slice is computed directly from every prediction,
    each subset keeps the same per-cell mask, an absent subset is skipped, a one-class target is NA."""
    from pathlib import Path

    from .metrics import subset_metrics
    from .report import build_html_report

    rng = np.random.default_rng(3)
    n = 40
    ids = [f"s{i:02d}" for i in range(n)]
    reference = rng.integers(0, 2, size=(n, N_TARGETS)).astype(float)
    reference[[0, 20], :] = 0.0
    reference[[1, 21], :] = 1.0  # both classes in each half for every target ...
    reference[:20, -1] = 0.0  # ... except the last target, single-class in the first half
    scores = np.clip(0.6 * reference + 0.4 * rng.random((n, N_TARGETS)), 0.0, 1.0)
    valid = np.ones((n, N_TARGETS), dtype=bool)
    valid[39, :] = False  # a label-masked study
    valid[5, 2] = False  # a single masked cell
    predictions = pd.DataFrame(
        {
            STUDY_ID: np.repeat(ids, N_TARGETS),
            "target": np.tile(TARGETS, n),
            "score": scores.ravel(),
            "reference": reference.ravel(),
            "reference_valid": valid.ravel(),
        }
    )
    subsets = pd.DataFrame(
        {STUDY_ID: ids + ["unseen"], "subset": ["first"] * 20 + ["second"] * 20 + ["absent"]}
    )
    per_class, summary = subset_metrics(predictions, subsets)
    assert list(summary["subset"]) == [f"full_{n}", "first", "second"], summary["subset"].tolist()
    table, direct = evaluate_predictions(scores, reference, valid)
    full = summary.iloc[0]
    assert math.isclose(full["macro_soft_auc"], direct["macro_soft_auc"]), "full slice must be computed directly"
    full_rows = per_class[per_class["subset"] == f"full_{n}"].set_index("target")
    assert np.allclose(full_rows.loc[list(TARGETS), "soft_auc"], table["soft_auc"], equal_nan=True)
    for column in ("n_pos", "n_neg"):
        by_subset = per_class[per_class["subset"] != f"full_{n}"].groupby("target")[column].sum()
        assert (by_subset.loc[list(TARGETS)].to_numpy() == full_rows.loc[list(TARGETS), column].to_numpy()).all()
    first = per_class[per_class["subset"] == "first"].set_index("target")
    assert np.isnan(first.loc[TARGETS[-1], "soft_auc"]), "a single-class target in a subset must be NA"
    assert first.loc[TARGETS[2], "n_pos"] + first.loc[TARGETS[2], "n_neg"] == 19, "a masked cell must stay masked"
    second = summary.set_index("subset").loc["second"]
    assert second["n_studies"] == 20 and second["n_evaluable_studies"] == 19

    run_dir = Path(tmp_dir) / "subset_report_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
    page = build_html_report(run_dir, subsets=subsets).read_text(encoding="utf-8")
    assert "Validation metrics by reference subset" in page
    written = pd.read_csv(run_dir / "subset_summary.csv")
    assert list(written["subset"]) == list(summary["subset"])
    assert (run_dir / "subset_metrics_per_class.csv").exists()
    return "full slice computed directly; subsets share the cell mask; absent subset skipped; one-class subset target NA"


def check_soft_auc_bruteforce() -> str:
    """Continuous reference: the O(n log n) formula matches the explicit pair sum."""
    rng = np.random.default_rng(1)
    cases = [
        (np.array([0.0, 0.3, 0.7, 1.0, 0.5, 0.5]), np.array([0.1, 0.4, 0.4, 0.9, 0.2, 0.6])),
        (rng.integers(0, 5, size=40) / 4.0, rng.integers(0, 6, size=40) / 5.0),  # ties on both sides
    ]
    for y, scores in cases:
        numerator = denominator = 0.0
        for i in range(len(y)):
            for j in range(len(y)):
                pair = max(y[i] - y[j], 0.0)
                step = 1.0 if scores[i] > scores[j] else 0.5 if scores[i] == scores[j] else 0.0
                numerator += pair * step
                denominator += pair
        assert math.isclose(soft_roc_auc(y, scores), numerator / denominator, abs_tol=1e-9)
    perfect = rng.random(30)
    assert math.isclose(soft_roc_auc(perfect, perfect), 1.0), "a score equal to the reference must rank perfectly"
    assert math.isclose(soft_roc_auc(perfect, -perfect), 0.0), "a reversed score must rank at 0"
    return "soft ROC-AUC matches the brute-force pair sum on a continuous reference"


def check_nonfinite_scores_rejected() -> str:
    """A NaN/Inf prediction must fail the evaluation, never rank as the top score."""
    references = {
        "binary": np.tile(np.array([0.0, 1.0, 0.0, 1.0])[:, None], (1, N_TARGETS)),
        "mixed": np.tile(np.array([0.0, 1.0, 0.3, 0.8])[:, None], (1, N_TARGETS)),
        "soft": np.tile(np.array([0.2, 0.8, 0.4, 0.6])[:, None], (1, N_TARGETS)),
    }
    valid = np.ones((4, N_TARGETS), dtype=bool)
    for name, reference in references.items():
        for bad in (np.nan, np.inf, -np.inf):
            scores = np.tile(np.array([0.1, 0.9, 0.3, 0.7])[:, None], (1, N_TARGETS))
            scores[3, N_TARGETS - 1] = bad  # a single broken cell
            for call in (
                lambda: evaluate_predictions(scores, reference, valid),
                lambda: soft_roc_auc(reference[:, -1], scores[:, -1]),
            ):
                try:
                    call()
                except ValueError:
                    continue
                raise AssertionError(f"a {bad} score on a {name} reference was accepted")

    # The original failure: NaN on the high-reference study scored a perfect 1.0.
    try:
        evaluate_predictions(
            np.tile([[0.1], [np.nan]], (1, N_TARGETS)), np.tile([[0.2], [0.8]], (1, N_TARGETS)), valid[:2]
        )
    except ValueError:
        pass
    else:
        raise AssertionError("NaN scores produced a macro soft ROC-AUC instead of an error")

    # A broken output is an error even where the reference is not valid.
    scores = np.tile(np.array([0.1, 0.9, 0.3, 0.7])[:, None], (1, N_TARGETS))
    scores[0, 0] = np.nan
    masked = valid.copy()
    masked[0, 0] = False
    try:
        evaluate_predictions(scores, references["binary"], masked)
    except ValueError:
        return "non-finite scores raise on binary, mixed and soft references, masked cells included"
    raise AssertionError("a NaN score in a masked cell was accepted")


def check_soft_reference() -> str:
    """Continuous wide-export values become soft cells that the evaluation reference keeps."""
    from .labels import KIND_SOFT
    from .train import EvaluationReference

    cfg = load_config(resolve=False)
    cfg.labels.allow_soft_targets = True
    ids = [f"s{i}" for i in range(6)]
    column = [0.0, 0.3, 0.7, 1.0, np.nan, 0.5]
    frame = pd.DataFrame({STUDY_ID: ids, **{t: column for t in TARGETS}})
    table = build_from_wide_numeric(frame, ids, cfg, source="unit-test")
    assert table.kinds[1, 0] == KIND_SOFT and table.targets[1, 0] == np.float32(0.3) and table.weights[1, 0] == 1.0

    _, binary_valid = table.binary_reference()
    values, valid = table.continuous_reference()
    assert not binary_valid[1, 0] and valid[1, 0] and not valid[4, 0], "soft kept, unknown excluded"
    reference = EvaluationReference.from_table(table, ids)
    assert np.array_equal(reference.values, values) and np.array_equal(reference.valid, valid)

    counts = table.counts_frame().iloc[0]
    assert int(counts["n_soft"]) == 3 and math.isclose(counts["eff_positive"], 2.5, abs_tol=1e-6)

    scores = np.tile(np.array([0.1, 0.35, 0.8, 0.9, 0.5, 0.6])[:, None], (1, N_TARGETS))
    metrics, summary = evaluate_predictions(scores, reference.values, reference.valid)
    assert int(metrics.loc[0, "n_soft"]) == 3 and summary["n_soft_cells"] == 3 * N_TARGETS
    assert math.isclose(metrics.loc[0, "soft_auc"], 1.0) and math.isclose(summary["macro_soft_auc"], 1.0)
    assert math.isclose(metrics.loc[0, "roc_auc"], 1.0) and int(metrics.loc[0, "n_known"]) == 5

    cfg.labels.allow_soft_targets = False
    try:
        build_from_wide_numeric(frame, ids, cfg, source="unit-test")
    except ValueError:
        pass
    else:
        raise AssertionError("continuous values must be refused while labels.allow_soft_targets is off")
    return "continuous targets become soft cells, enter the reference and the soft ROC-AUC"


def check_bootstrap_keeps_soft() -> str:
    """The bootstrap must not truncate a 0.7 reference to 0."""
    from .evaluate import bootstrap_intervals

    rng = np.random.default_rng(2)
    ids = [f"s{i}" for i in range(40)]
    reference = rng.random(len(ids))
    predictions = pd.DataFrame(
        {
            STUDY_ID: ids,
            "target": TARGETS[0],
            "score": reference + rng.normal(0.0, 0.1, len(ids)),
            "reference": reference,
            "reference_valid": True,
        }
    )
    splits = pd.DataFrame({STUDY_ID: ids, "group": ids})
    result = bootstrap_intervals(predictions, splits, n_boot=20, seed=0).set_index("target")
    expected = soft_roc_auc(reference, predictions["score"].to_numpy())
    assert math.isclose(result.loc[TARGETS[0], "soft_auc"], expected), "bootstrap point estimate truncated the reference"
    assert result.loc[TARGETS[0], "ci_low"] <= expected <= result.loc[TARGETS[0], "ci_high"]
    return "bootstrap scores continuous references without truncation"


def check_crop_edge_fill() -> str:
    """Tissue touching one border band must register on exactly that edge."""
    from .qc import edge_fill_fractions

    image = np.zeros((9, 32, 32), dtype=np.float32)
    image[:, 8:24, 8:32] = 0.6  # tissue running out of the right edge only
    image[:3] = 0.0  # outer slices are ignored (middle third only)
    fills = edge_fill_fractions(image, band_px=4, tissue_threshold=0.1)
    assert fills["right"] == 0.5, fills  # 16 of 32 rows are tissue
    assert fills["left"] == fills["top"] == fills["bottom"] == 0.0, fills
    return "edge bands measure tissue per edge over the middle slices"


def check_foreground_extent_center() -> str:
    """The extent midpoint sits between the skin lines; the centroid follows the tissue bulk."""
    from .preprocess import estimate_center

    # The foreground centroid is an AREA centroid of the thresholded mask. A limb whose
    # posterior part is tall and whose anterior part is thin (as on sagittal images, where
    # the posterior musculature fills the frame) drags it posteriorly (left).
    stack = np.zeros((9, 100, 100), dtype=np.float32)
    stack[:, 10:90, 10:35] = 1.0  # tall posterior part, columns 10..34
    stack[:, 40:60, 35:60] = 1.0  # thin anterior part, columns 35..59 -> limb spans 10..59
    stack[:, 50, 99] = 1.0  # one stray bright pixel at the far edge
    row_c, col_c = estimate_center(stack, "foreground")
    row_e, col_e = estimate_center(stack, "foreground_extent")
    assert abs(col_e - 34.5) <= 1.0, f"extent midpoint {col_e}, expected ~34.5"
    assert abs(row_e - 49.5) <= 1.0, f"extent row midpoint {row_e}, expected ~49.5"
    assert col_c < col_e - 5, f"centroid {col_c} should sit clearly left of the extent midpoint {col_e}"
    return f"centroid col {col_c:.1f} vs extent midpoint {col_e:.1f} (limb spans 10..59)"


def check_group_separation(tmp_dir) -> str:
    from pathlib import Path

    cfg = load_config()
    cfg = cfg.copy()
    cfg.paths.work_dir = str(tmp_dir)
    cfg.split.n_folds = 3
    cfg.split.holdout_reference = False

    studies = [f"study{i:03d}" for i in range(60)]
    audit = pd.DataFrame(
        {
            STUDY_ID: studies,
            "patient_id": [f"p{i // 2:03d}" for i in range(60)],  # two studies per patient
            "n_distinct_patient_ids": 1,
            "patient_id_missing": False,
            "patient_id_inconsistent": False,
            "patient_id_site_reused": False,
            "n_studies_for_id": 2,
        }
    )
    splits = make_splits(
        cfg,
        study_ids=studies,
        label_table=None,
        patient_audit=audit,
        out_path=Path(tmp_dir) / "splits.csv",
    )
    assert splits["group_source"].iloc[0] == "patient"
    for fold in range(3):
        assert_group_disjoint(splits, fold)
    pairs = splits.groupby("group")["fold"].nunique()
    assert int(pairs.max()) == 1, "a patient group was split across folds"
    return "patient groups stay inside one fold"


def check_fixed_split(tmp_dir) -> str:
    """make-fixed-split: validation studies are fold 0, training fold 1; overlap is refused."""
    from pathlib import Path

    cfg = load_config(resolve=False)
    train = [f"t{i:02d}" for i in range(10)]
    val = [f"v{i:02d}" for i in range(4)]
    splits = make_fixed_split(cfg, train, val, out_path=Path(tmp_dir) / "fixed" / "splits.csv")
    assert_group_disjoint(splits, 0)
    got_train, got_val = fold_study_ids(splits, 0)
    assert got_train == sorted(train) and got_val == sorted(val), (got_train, got_val)
    try:
        make_fixed_split(cfg, train, val + ["t03"], out_path=Path(tmp_dir) / "fixed" / "overlap.csv")
    except ValueError:
        pass
    else:
        raise AssertionError("a study in both sets must be an error")
    return "fold 0 = validation set, fold 1 = training set, overlap refused"


def check_training_exclusion(tmp_dir) -> str:
    """The reference never trains: empty hold-out refused, CV split checked, stale split caught."""
    import json
    from pathlib import Path

    cfg = load_config()
    cfg = cfg.copy()
    cfg.paths.work_dir = str(tmp_dir)
    cfg.split.n_folds = 3
    studies = [f"s{i:03d}" for i in range(30)]
    reference = studies[:6]

    cfg.split.holdout_reference = True
    try:
        make_splits(cfg, study_ids=studies, reference_study_ids=[], out_path=Path(tmp_dir) / "excl" / "empty.csv")
    except ValueError:
        pass
    else:
        raise AssertionError("holdout_reference=true without a reference study must be an error")

    splits = make_splits(cfg, study_ids=studies, reference_study_ids=reference, out_path=Path(tmp_dir) / "excl" / "held.csv")
    held = splits.set_index(STUDY_ID).loc[reference]
    assert (held["fold"] == -1).all() and (held["role"] == "reference_holdout").all(), held
    pool = studies[6:]
    meta = json.loads((Path(tmp_dir) / "excl" / "splits_meta.json").read_text(encoding="utf-8"))
    errors, _ = verify_cv_split(splits, meta, 3, pool)
    assert not errors, errors
    errors, _ = verify_cv_split(splits, meta, 5, pool)
    assert any("folds" in e for e in errors), errors
    errors, _ = verify_cv_split(splits, meta, 3, pool[1:])
    assert any("no row in the training CSV" in e for e in errors), errors
    for fold in range(3):
        train_ids, _ = fold_study_ids(splits, fold)
        assert_not_training(train_ids, set(reference))

    # A stale split that trains on the reference is caught at training time.
    cfg.split.holdout_reference = False
    stale = make_splits(cfg, study_ids=studies, out_path=Path(tmp_dir) / "excl" / "stale.csv")
    exclude_csv = Path(tmp_dir) / "excl" / "reference.csv"
    pd.DataFrame({STUDY_ID: reference}).to_csv(exclude_csv, index=False)
    cfg.paths.exclude_from_training_csv = str(exclude_csv)
    excluded = load_excluded_training_ids(cfg)
    assert excluded == set(reference), excluded
    try:
        assert_not_training(fold_study_ids(stale, 0)[0], excluded)
    except AssertionError:
        pass
    else:
        raise AssertionError("a split with reference studies in the training pool must be refused")

    cfg.paths.exclude_from_training_csv = str(Path(tmp_dir) / "excl" / "missing.csv")
    try:
        load_excluded_training_ids(cfg)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("a configured but missing exclusion file must be an error")
    cfg.paths.exclude_from_training_csv = None
    assert load_excluded_training_ids(cfg) is None
    return "empty hold-out refused, reference held out, stale split and missing list caught"


def check_verify_fixed_split(tmp_dir) -> str:
    """check-fixed-split: a training/validation CSV that no longer matches the split is refused."""
    import json
    from pathlib import Path

    cfg = load_config(resolve=False)
    train = [f"t{i:02d}" for i in range(10)]
    val = [f"v{i:02d}" for i in range(4)]
    out = Path(tmp_dir) / "fixed_verify" / "splits.csv"
    splits = make_fixed_split(cfg, train, val, out_path=out, sources={"train_csv": "a/train_new.csv"})
    meta = json.loads(out.with_name("splits_meta.json").read_text(encoding="utf-8"))

    errors, warnings = verify_fixed_split(splits, meta, train, val, train_csv="a/train_new.csv")
    assert not errors and not warnings, (errors, warnings)
    # the old, larger training CSV still lists the validation studies: label leak
    errors, _ = verify_fixed_split(splits, meta, train + val[:2], val, train_csv="a/train_old.csv")
    assert any("also have a row in the training CSV" in e for e in errors), errors
    # a validation CSV the split was not made from
    errors, _ = verify_fixed_split(splits, meta, train, val + ["v99"], train_csv="a/train_new.csv")
    assert any("validates 4 studies" in e for e in errors), errors
    # training studies without labels are an error, extra unused rows only a warning
    errors, _ = verify_fixed_split(splits, meta, train[:-1], val, train_csv="a/train_new.csv")
    assert any("have no row in the training CSV" in e for e in errors), errors
    errors, warnings = verify_fixed_split(splits, meta, train + ["x01"], val, train_csv="a/other.csv")
    assert not errors and len(warnings) == 2, (errors, warnings)
    errors, _ = verify_fixed_split(splits, {"mode": "cv"}, train, val)
    assert errors, "a CV split is not a fixed hold-out"
    return "leaking / mismatched CSVs refused, unused rows and a different source path only warned"


def check_freeze_reference_from_csv(tmp_dir) -> str:
    """freeze-reference --from-csv: empty cells are invalid, non-binary values are refused."""
    from pathlib import Path

    from .cli import main as cli_main
    from .train import EvaluationReference

    ids = ["r2", "r0", "r1"]
    frame = pd.DataFrame(np.zeros((3, len(TARGETS))), columns=TARGETS)
    frame.iloc[0, 0] = 1.0
    frame.iloc[1, 3] = np.nan
    frame.insert(0, STUDY_ID, ids)
    src = Path(tmp_dir) / "ref_wide.csv"
    frame.to_csv(src, index=False)
    out = Path(tmp_dir) / "frozen_from_csv.csv"
    assert cli_main(["freeze-reference", "--from-csv", str(src), "--out", str(out)]) == 0
    loaded = EvaluationReference.from_csv(out, ids)
    assert loaded.values[0, 0] == 1.0 and loaded.valid[0, 0]
    assert not loaded.valid[1, 3] and int((~loaded.valid).sum()) == 1

    frame.iloc[2, 5] = 0.5
    frame.to_csv(src, index=False)
    try:
        cli_main(["freeze-reference", "--from-csv", str(src), "--out", str(out), "--force"])
    except ValueError:
        pass
    else:
        raise AssertionError("a non-binary reference value must be an error")
    return "empty cells frozen as invalid, 0.5 refused"


def check_pseudonymous_patient_id(tmp_dir) -> str:
    """A PatientID that never repeats must not be sold as patient-level grouping."""
    from pathlib import Path

    cfg = load_config().copy()
    cfg.paths.work_dir = str(tmp_dir)
    cfg.split.n_folds = 3
    cfg.split.holdout_reference = False

    studies = [f"study{i:03d}" for i in range(30)]
    audit = pd.DataFrame(
        {
            STUDY_ID: studies,
            "patient_id": [f"pseudo{i:03d}" for i in range(30)],  # one id per study
            "n_distinct_patient_ids": 1,
            "patient_id_missing": False,
            "patient_id_inconsistent": False,
            "patient_id_site_reused": False,
            "n_studies_for_id": 1,
        }
    )
    splits = make_splits(
        cfg,
        study_ids=studies,
        label_table=None,
        patient_audit=audit,
        out_path=Path(tmp_dir) / "pseudo_splits.csv",
    )
    source = splits["group_source"].iloc[0]
    assert source == "study", f"per-study-unique PatientID was reported as '{source}' grouping"
    assert splits["group"].nunique() == len(studies)
    return "per-study-unique PatientID is reported as study-level grouping, not patient-level"


def check_dataset_contract(cfg: Config) -> str:
    dataset = SyntheticBagDataset(cfg, n_studies=4)
    item = dataset[0]
    p, s = len(cfg.data.series_slots), int(cfg.data.centers_per_series)
    size = int(cfg.data.image_size)
    assert tuple(item["images"].shape) == (p, s, 3, size, size), item["images"].shape
    assert tuple(item["slice_valid_mask"].shape) == (p, s)
    assert tuple(item["series_present_mask"].shape) == (p,)
    assert item["targets"].shape == (N_TARGETS,) and torch.isfinite(item["targets"]).all()
    assert (item["label_weights"] >= 0).all()
    batch = collate_studies([dataset[i] for i in range(3)])
    assert tuple(batch["images"].shape) == (3, p, s, 3, size, size)
    assert len(batch["study_ids"]) == 3
    return "item and batch shapes match the documented contract"


class _InMemoryStudyBagDataset(StudyBagDataset):
    """StudyBagDataset with a synthetic volume per (study, slot) instead of the disk cache.

    Everything downstream of `_load_slot` - centre sampling, the per-(seed, epoch, index,
    slot) generator, augmentation - is the real code. Module level so spawned DataLoader
    workers can unpickle it.
    """

    N_SLICES = 20
    VERIFY_SELECTION = False  # synthetic volumes: no cache entries to check against a selection

    def _load_slot(self, study: str, slot: str) -> tuple[np.ndarray | None, dict]:
        seed = [int(study.removeprefix("study")), self.slots.index(slot)]
        image = np.random.default_rng(seed).random((self.N_SLICES, self.image_size, self.image_size)).astype(np.float32)
        return image, {}


class _Float16CacheDataset(_InMemoryStudyBagDataset):
    """Like the in-memory dataset, but its volumes are float16-quantised, as the real cache is."""

    def _load_slot(self, study: str, slot: str) -> tuple[np.ndarray | None, dict]:
        image, meta = super()._load_slot(study, slot)
        return image.astype(np.float16).astype(np.float32), meta


def check_float16_transport(cfg: Config) -> str:
    """Deferred bags travel as float16 without changing a single value, and halve the bytes.

    Only the deferred (GPU-augmented) path may do this: its bag is gathered slices of a
    float16 cache, so widening back to float32 is exact. The CPU path normalises in the
    worker and must stay float32.

    Hardware-independent: the deferred decision is pinned to "cuda" here, because packing a
    deferred bag needs no GPU (augmentation happens later, in the trainer). Whether a real
    host resolves to cuda or falls back to cpu is `augment_device_resolution`'s business.
    """
    from unittest import mock

    from . import dataset as dataset_module

    small = cfg.copy()
    small.augment.device = "cuda"
    small.data.cache_dtype = "float16"
    wide = small.copy()
    wide.data.cache_dtype = "float32"
    ids = [f"study{i}" for i in range(2)]
    with mock.patch.object(dataset_module, "resolve_augment_device", return_value="cuda"):
        deferred = _Float16CacheDataset(small, ids, None, train=True)
        reference = _Float16CacheDataset(wide, ids, None, train=True)
        val = _Float16CacheDataset(small, ["study0"], None, train=False)
    assert deferred.defer_augment and deferred.transport_dtype == torch.float16, "deferred bags should travel as float16"
    assert reference.transport_dtype == torch.float32
    assert val.transport_dtype == torch.float32, "the validation path normalises in the worker"

    for index in range(2):
        packed = deferred[index]["images"]
        plain = reference[index]["images"]
        assert packed.dtype == torch.float16 and plain.dtype == torch.float32
        assert torch.equal(packed.float(), plain), "float16 transport changed a pixel value"
        assert packed.element_size() * packed.numel() * 2 == plain.element_size() * plain.numel()

    cpu_path = small.copy()
    cpu_path.augment.device = "cpu"
    assert _Float16CacheDataset(cpu_path, ["study0"], None, train=True).transport_dtype == torch.float32
    mb = packed.element_size() * packed.numel() / 1024**2
    return f"deferred bags float16 and bit-identical ({mb:.1f} MB/study instead of {2 * mb:.1f}); CPU and val paths float32"


def check_explicit_run_selection() -> str:
    """`--runs` names the runs; the prefix form cannot separate a second seed from a fold."""
    import tempfile
    from pathlib import Path

    from .diagnose import resolve_run_dirs

    with tempfile.TemporaryDirectory(prefix="knee_mri_runs_") as tmp:
        root = Path(tmp)
        names = ["g_fold0_s42", "g_fold0_s43", "g_fold1_s42", "g_fold2_s42"]
        for name in names:
            (root / name).mkdir()
            (root / name / "validation_predictions.csv").write_text("StudyInstanceUID,target,score\n", encoding="utf-8")
        (root / "empty_fold3").mkdir()  # no predictions: never selected

        by_prefix = [p.name for p in resolve_run_dirs(root / "g")]
        assert by_prefix == names, by_prefix  # both fold-0 seeds, which is what breaks the OOF

        chosen = ["g_fold0_s42", "g_fold1_s42", "g_fold2_s42"]
        explicit = [p.name for p in resolve_run_dirs(None, [root / n for n in chosen])]
        assert explicit == chosen, explicit
        assert [p.name for p in resolve_run_dirs(root / "g_fold1_s42")] == ["g_fold1_s42"]

        for bad, exc in (
            ([root / "empty_fold3"], FileNotFoundError),
            ([root / "g_fold0_s42", root / "g_fold0_s42"], ValueError),
            (None, ValueError),
        ):
            try:
                resolve_run_dirs(None, bad)
            except exc:
                continue
            raise AssertionError(f"resolve_run_dirs should have raised {exc.__name__} for {bad}")
    return "prefix takes all 4 runs (2 seeds of fold 0); --runs takes exactly the 3 named; bad lists rejected"


def check_eval_loader_budget(cfg: Config) -> str:
    """The validation loader uses its own worker budget and never keeps workers alive."""
    from .train import eval_loader_settings

    small = cfg.copy()
    small.train.num_workers = 8
    small.train.prefetch_factor = 4
    small.train.eval_num_workers = 4
    small.train.eval_prefetch_factor = 2
    assert eval_loader_settings(small, "train") == (8, 4, True)
    assert eval_loader_settings(small, "eval") == (4, 2, False)

    fallback = small.copy()
    fallback.train.eval_num_workers = None
    fallback.train.eval_prefetch_factor = None
    assert eval_loader_settings(fallback, "eval") == (8, 4, False), "unset keys must fall back to the train values"

    off = small.copy()
    off.train.eval_num_workers = 0
    assert eval_loader_settings(off, "eval") == (0, 2, False)
    return "eval loader: own worker/prefetch budget, never persistent; unset keys fall back to the training values"


def _collect_epoch(loader, epoch: int) -> dict[str, torch.Tensor]:
    loader.sampler.set_epoch(epoch)
    loader.dataset.set_epoch(epoch)  # what the trainer does; must not be what makes it work
    out: dict[str, torch.Tensor] = {}
    for batch in loader:
        for i, study in enumerate(batch["study_ids"]):
            out[study] = batch["images"][i].clone()
    return out


def check_epoch_reaches_workers(cfg: Config) -> str:
    """The epoch must reach persistent DataLoader workers, not only the main-process dataset.

    With `persistent_workers=True` each worker keeps its own dataset copy, so the epoch has
    to travel with the indices (EpochSampler). Asserted on real StudyBagDataset items:
      (a) the same study gets different centres/augmentation in epoch 0 and epoch 1;
      (b) repeating an epoch reproduces it exactly;
      (c) two persistent workers give exactly what the in-process num_workers=0 path gives.
    """
    from torch.utils.data import DataLoader, SequentialSampler

    small = cfg.copy()
    small.augment.enabled = True
    small.augment.device = "cpu"  # augment in the worker, so the images carry the draw
    small.augment.noise_std = 0.0
    studies = [f"study{i}" for i in range(4)]

    def loader(num_workers: int) -> DataLoader:
        dataset = _InMemoryStudyBagDataset(small, studies, None, train=True, strict_cache=False)
        kwargs = {"num_workers": num_workers, "collate_fn": collate_studies, "batch_size": 1}
        if num_workers:
            kwargs["persistent_workers"] = True
        return DataLoader(dataset, sampler=EpochSampler(SequentialSampler(dataset)), **kwargs)

    workers = loader(2)
    epoch0 = _collect_epoch(workers, 0)
    epoch1 = _collect_epoch(workers, 1)
    epoch0_again = _collect_epoch(workers, 0)
    inline1 = _collect_epoch(loader(0), 1)

    for study in studies:
        assert not torch.equal(epoch0[study], epoch1[study]), f"{study}: epoch 1 repeated epoch 0 in the workers"
        assert torch.equal(epoch0[study], epoch0_again[study]), f"{study}: epoch 0 is not reproducible"
        assert torch.equal(epoch1[study], inline1[study]), f"{study}: workers disagree with num_workers=0"
    return "persistent workers see the epoch: draws change per epoch, repeat exactly, match num_workers=0"


def check_augment_device_equivalence(cfg: Config) -> str:
    """Augmenting in the worker and augmenting in one batched call must be the same function.

    Run on one device with the noise field off, the two are required to be *bit-identical*:
    both are driven from the same drawn parameters, so any difference at all is an
    implementation divergence. (Across devices they differ by ~1e-4, which is PyTorch's
    float32 CPU and CUDA `grid_sample` kernels, not this code.)
    """
    small = cfg.copy()
    small.augment.enabled = True
    small.augment.noise_std = 0.0
    # No model is built here, so the backbone's input size does not apply: the check keeps its
    # 32 px. Bit-identity is a property of the kernels at that size - at 64 px PyTorch's CPU
    # pow with a broadcast tensor exponent (batched gamma) differs from pow(float) (per-series
    # gamma) by 1 ulp in single elements (seen with torch 2.13; 32/56/96/224 px are exact).
    small.data.image_size = 32
    n_slots, n_centers = len(small.data.series_slots), int(small.data.centers_per_series)
    size = int(small.data.image_size)

    rng = np.random.default_rng(5)
    raw = torch.from_numpy(rng.random((1, n_slots, n_centers, 3, size, size)).astype(np.float32))
    slice_valid = torch.ones((1, n_slots, n_centers), dtype=torch.bool)
    slice_valid[0, 0, -1] = False  # a padded centre
    present = torch.ones((1, n_slots), dtype=torch.bool)
    present[0, -1] = False  # an absent slot
    slice_valid[0, -1] = False

    mean, std = normalization_stats(small)
    mean_bag = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_bag = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

    # float64, as StudyBagDataset emits them: affine_theta needs the undiminished values.
    params = torch.zeros((1, n_slots, len(AUGMENT_PARAM_COLUMNS)), dtype=torch.float64)
    expected = torch.zeros_like(raw)
    for slot in range(n_slots):
        prng = np.random.default_rng([7, slot])
        spatial = sample_spatial(small, prng)
        intensity = sample_intensity(small, prng)
        params[0, slot] = torch.tensor(pack_augment_params(spatial, intensity), dtype=torch.float64)
        if not bool(present[0, slot]):
            continue  # the CPU path never writes an absent slot: it stays zero
        bag = apply_spatial(raw[0, slot], spatial)
        bag = apply_intensity(bag, small, prng, params=intensity)
        bag = (bag - mean_bag) / std_bag
        expected[0, slot] = bag * slice_valid[0, slot].to(torch.float32).view(-1, 1, 1, 1)

    got = augment_batch(
        raw,
        params,
        slice_valid,
        present,
        mean_bag.view(1, 1, 1, 3, 1, 1),
        std_bag.view(1, 1, 1, 3, 1, 1),
        noise_std=0.0,
    )

    max_diff = float((got - expected).abs().max())
    assert max_diff == 0.0, f"batched augmentation is not bit-identical (max abs diff {max_diff:.2e})"

    mask = slice_valid & present.unsqueeze(-1)
    leaked = float(got[~mask].abs().max()) if (~mask).any() else 0.0
    assert leaked == 0.0, f"padded centre or absent slot is not exactly zero ({leaked})"
    return "batched augmentation is bit-identical on one device; padding stays exactly zero"


def check_augment_device_resolution(cfg: Config) -> str:
    """`augment.device` must resolve the same way everywhere, and tolerate an old config."""
    forced_cpu = cfg.copy()
    forced_cpu.augment.device = "cpu"
    assert resolve_augment_device(forced_cpu) == "cpu", "augment.device=cpu was not honoured"

    auto = cfg.copy()
    auto.augment.device = "auto"
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_augment_device(auto) == expected, "auto did not follow device availability"

    legacy = cfg.copy()
    legacy.augment.pop("device", None)  # a run config written before the key existed
    assert resolve_augment_device(legacy) == expected, "a config without augment.device did not default to auto"

    forced_cuda = cfg.copy()
    forced_cuda.augment.device = "cuda"
    resolved = resolve_augment_device(forced_cuda)
    assert resolved == expected, "cuda must fall back to cpu when no device is visible"
    return f"cpu|auto|cuda resolve correctly; missing key defaults to auto ({expected})"


def check_checkpoint_roundtrip(cfg: Config, tmp_dir) -> str:
    from pathlib import Path

    model = _tiny_model(cfg, seed=11)
    batch = _random_batch(cfg, b=1, seed=12)
    with torch.no_grad():
        before = model(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
    path = Path(tmp_dir) / "ckpt.pt"
    torch.save({"model": model.state_dict(), "target_order": list(TARGETS)}, path)

    restored = _tiny_model(cfg, seed=99)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["target_order"] == list(TARGETS), "target order must travel with the checkpoint"
    restored.load_state_dict(payload["model"])
    restored.eval()
    with torch.no_grad():
        after = restored(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
    assert torch.allclose(before, after, atol=1e-6), "reloaded model does not reproduce the outputs"
    return "checkpoint reload reproduces identical logits"


def check_skipped_amp_step() -> str:
    """A step the GradScaler skips (inf gradients) must not advance the LR schedule or global_step."""
    from types import SimpleNamespace

    from .train import Trainer, TrainState, make_scheduler

    torch.manual_seed(0)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = load_config(resolve=False)
    cfg.train.grad_clip = 1.0
    trainer = SimpleNamespace(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=make_scheduler(optimizer, warmup_steps=2, total_steps=10),
        scaler=torch.amp.GradScaler("cpu", init_scale=2.0**8),  # a real scaler, no CUDA needed
        state=TrainState(),
    )
    x = torch.ones(4, 2)

    def step(loss_scale: float) -> None:
        optimizer.zero_grad(set_to_none=True)
        trainer.scaler.scale(model(x).sum() * loss_scale).backward()
        Trainer._optimizer_step(trainer)

    step(1.0)
    assert trainer.state.global_step == 1 and trainer.scheduler.last_epoch == 1 and trainer.state.skipped_steps == 0
    weights, lr = model.weight.detach().clone(), optimizer.param_groups[0]["lr"]
    step(float("inf"))  # overflow: the scaler must skip the update
    assert torch.equal(model.weight, weights), "the scaler did not skip the overflowing step"
    assert trainer.state.global_step == 1 and trainer.scheduler.last_epoch == 1, "a skipped step advanced the schedule"
    assert trainer.state.skipped_steps == 1 and optimizer.param_groups[0]["lr"] == lr
    step(1.0)
    assert trainer.state.global_step == 2 and trainer.scheduler.last_epoch == 2
    return "an overflow-skipped step leaves the LR schedule and global_step alone and is counted"


def check_inference_architecture(cfg: Config, tmp_dir) -> str:
    """Inference rebuilds the checkpoint's architecture and refuses silently different inputs."""
    from pathlib import Path

    from .constants import CHECKPOINT_VERSION
    from .evaluate import load_checkpoint_for_inference
    from .model import build_model

    trained = cfg.copy()
    trained.model.spatial_pool = "attention"
    trained.model.weights = "none"
    torch.manual_seed(5)
    model = build_model(trained).eval()
    path = Path(tmp_dir) / "attention.pt"
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "target_order": list(TARGETS),
            "config": trained.to_dict(),
            "model_description": model.describe(),
            "model": model.state_dict(),
        },
        path,
    )
    batch = _random_batch(cfg, b=1, seed=6)
    with torch.no_grad():
        expected = model(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])

    caller = cfg.copy()
    caller.model.spatial_pool = "avg"  # the base config: must not decide the architecture
    loaded, _ = load_checkpoint_for_inference(caller, path)
    with torch.no_grad():
        actual = loaded(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
    assert loaded.describe()["spatial_pool"] == "attention" and torch.allclose(expected, actual, atol=1e-6)

    other_norm = cfg.copy()
    other_norm.data.encoder_normalization = "imagenet" if cfg.data.encoder_normalization != "imagenet" else "mri_scalar"
    reordered = cfg.copy()
    reordered.data.series_slots = list(reversed(cfg.data.series_slots))  # same shapes, other meaning
    for caller, key in ((other_norm, "data.encoder_normalization"), (reordered, "data.series_slots")):
        try:
            load_checkpoint_for_inference(caller, path)
        except ValueError as error:
            assert key in str(error), f"refusal does not name {key}: {error}"
        else:
            raise AssertionError(f"a different {key} was accepted silently")
        load_checkpoint_for_inference(caller, path, allow_data_overrides=[key])  # explicit override works
    try:
        load_checkpoint_for_inference(cfg, path, allow_data_overrides=["model.spatial_pool"])
    except ValueError:
        pass
    else:
        raise AssertionError("an override outside the data keys was accepted")
    return "architecture rebuilt from the checkpoint (attention under an avg config); input mismatches need an explicit override"


def check_backbone_choice(cfg: Config) -> str:
    """V2-S builds, keeps the 1280-wide slice features and says so in its description."""
    from .model import build_model

    widths = {}
    for backbone in ("efficientnet_b0", "efficientnet_v2_s"):
        chosen = cfg.copy()
        chosen.model.backbone = backbone
        chosen.model.weights = "none"
        torch.manual_seed(0)
        model = build_model(chosen).eval()
        description = model.describe()
        assert description["architecture"] == f"{backbone}_2p5d_mil", description["architecture"]
        assert description["backbone"] == backbone and model.feature_dim == ENCODER_FEATURES[backbone]
        batch = _random_batch(chosen, b=1, seed=2)
        with torch.no_grad():
            logits = model(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
        assert logits.shape == (1, N_TARGETS) and torch.isfinite(logits).all(), backbone
        widths[backbone] = (model.feature_dim, description["n_parameters"])
    unknown = cfg.copy()
    unknown.model.backbone = "resnet50"
    try:
        validate_config(unknown)
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown model.backbone must be rejected")
    return f"(feature_dim, parameters) per backbone: {widths}; unknown backbone rejected"


def check_grad_checkpointing_is_exact(cfg: Config) -> str:
    """Gradient checkpointing changes memory, not the loss or the gradients."""
    from .model import EfficientNetMIL

    batch = _random_batch(cfg, b=1, seed=4)
    results = []
    for enabled in (False, True):
        torch.manual_seed(0)
        model = EfficientNetMIL(
            n_targets=N_TARGETS,
            n_slots=len(cfg.data.series_slots),
            slot_names=list(cfg.data.series_slots),
            weights="none",
            head_hidden=int(cfg.model.head_hidden),
            head_dropout=0.0,
            grad_checkpointing=enabled,
        ).train()
        torch.manual_seed(1)  # stochastic depth draws the same masks in both runs
        logits = model(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
        logits.square().mean().backward()
        results.append((logits.detach(), [p.grad.clone() for p in model.features.parameters() if p.grad is not None]))
    (plain, plain_grads), (ckpt, ckpt_grads) = results
    assert torch.allclose(plain, ckpt, atol=1e-6), "checkpointing changed the forward pass"
    assert len(plain_grads) == len(ckpt_grads) > 0
    worst = max(float((a - b).abs().max()) for a, b in zip(plain_grads, ckpt_grads))
    assert worst < 1e-5, f"checkpointing changed the encoder gradients by {worst:.2e}"
    return f"identical logits and encoder gradients with and without checkpointing (max diff {worst:.1e})"


def check_legacy_checkpoint_is_b0(cfg: Config, tmp_dir) -> str:
    """A checkpoint from before model.backbone existed loads as B0 even under a V2-S config."""
    from pathlib import Path

    from .constants import CHECKPOINT_VERSION
    from .evaluate import load_checkpoint_for_inference
    from .model import build_model

    trained = cfg.copy()
    trained.model.backbone = "efficientnet_b0"
    trained.model.weights = "none"
    torch.manual_seed(7)
    model = build_model(trained).eval()
    stored = trained.to_dict()
    del stored["model"]["backbone"]  # what an older run wrote
    description = model.describe()
    del description["backbone"]
    path = Path(tmp_dir) / "legacy_b0.pt"
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "target_order": list(TARGETS),
            "config": stored,
            "model_description": description,
            "model": model.state_dict(),
        },
        path,
    )
    caller = cfg.copy()
    caller.model.backbone = "efficientnet_v2_s"
    loaded, _ = load_checkpoint_for_inference(caller, path)
    assert loaded.describe()["backbone"] == "efficientnet_b0"
    return "a checkpoint without model.backbone rebuilds as efficientnet_b0 under a V2-S config"


def check_exact_resume(cfg: Config) -> str:
    """Stopping after epoch 0 and resuming from last.pt must repeat the uninterrupted run exactly."""
    import tempfile
    from pathlib import Path

    small = cfg.copy()
    small.train.num_workers = 0
    small.train.eval_num_workers = 0
    small.train.microbatch_studies = 1
    small.train.accumulation_steps = 2
    small.train.early_stopping_patience = 10
    small.train.resume = None
    small.train.init_weights = None
    with tempfile.TemporaryDirectory(prefix="knee_mri_resume_") as tmp:
        small.paths.output_dir = str(Path(tmp))
        try:
            return _exact_resume_checks(small, Path(tmp))
        finally:
            remove_file_logging(tmp)  # Windows cannot delete an open run.log


def _exact_resume_checks(small: Config, tmp) -> str:
    from .train import build_synthetic_trainer

    def build(run_cfg: Config, name: str):
        return build_synthetic_trainer(run_cfg, n_studies=6, epochs=3, name=name)

    full = build(small, "full")
    full.fit()

    interrupted = build(small, "interrupted")
    interrupted.cfg.train.max_epochs = 1  # stop after epoch 0; schedule and signature were built for 3
    interrupted.fit()

    # The first epoch always improves, so probe a non-zero early-stopping counter directly.
    interrupted.state.epochs_without_improvement = 4
    interrupted.save_checkpoint("probe.pt")
    probe_cfg = small.copy()
    probe_cfg.train.resume = str(tmp / "interrupted" / "probe.pt")
    assert build(probe_cfg, "probe").state.epochs_without_improvement == 4, "early-stopping counter reset on resume"

    resume_cfg = small.copy()
    resume_cfg.train.resume = str(tmp / "interrupted" / "last.pt")
    resumed = build(resume_cfg, "resumed")
    assert resumed.state.epoch == 1 and len(resumed.state.history) == 1, "history/epoch not restored"
    resumed.fit()

    volatile = ["seconds", "studies_per_second", "peak_gpu_gb"]
    expected = pd.DataFrame(full.state.history).drop(columns=volatile)
    actual = pd.DataFrame(resumed.state.history).drop(columns=volatile)
    pd.testing.assert_frame_equal(expected, actual, check_exact=False, rtol=1e-6, atol=1e-7)
    assert len(pd.read_csv(tmp / "resumed" / "history.csv")) == 3, "history.csv lost the epochs before resume"
    for key in ("global_step", "best_epoch", "epochs_without_improvement"):
        assert getattr(full.state, key) == getattr(resumed.state, key), f"{key} differs after resume"
    assert math.isclose(full.state.best_score, resumed.state.best_score, rel_tol=1e-6)
    assert torch.equal(full.sampler_generator.get_state(), resumed.sampler_generator.get_state()), "sampler order drifted"
    full_params, resumed_params = full.model.state_dict(), resumed.model.state_dict()
    drift = max(float((full_params[k].float() - resumed_params[k].float()).abs().max()) for k in full_params)
    assert drift <= 1e-6, f"model weights differ after resume (max {drift:.2e})"
    if full.ema is not None:
        full_ema, resumed_ema = full.ema.state_dict(), resumed.ema.state_dict()
        ema_drift = max(float((full_ema[k].float() - resumed_ema[k].float()).abs().max()) for k in full_ema)
        assert ema_drift <= 1e-6, f"EMA weights differ after resume (max {ema_drift:.2e})"
        assert full.ema.updates == resumed.ema.updates == full.state.global_step, "EMA update count not restored"
    saved_params = full.selected_model.state_dict()  # what last.pt stores as "model"

    # Another experiment must not resume from this checkpoint ...
    other = resume_cfg.copy()
    other.eval.selection_metric = "macro_roc_auc"
    try:
        build(other, "other")
    except ValueError as error:
        assert "eval.selection_metric" in str(error), f"mismatch not named: {error}"
    else:
        raise AssertionError("a resume with another selection metric was accepted")

    # ... but may start from its weights, with fresh state.
    other.train.resume = None
    other.train.init_weights = str(tmp / "full" / "last.pt")
    tuned = build(other, "tuned")
    assert tuned.state.epoch == 0 and not tuned.state.history and tuned.state.best_score == float("-inf")
    assert all(torch.equal(v, saved_params[k]) for k, v in tuned.model.state_dict().items()), "init_weights not loaded"
    if tuned.ema is not None:
        assert tuned.ema.updates == 0 and all(
            torch.equal(v, saved_params[k]) for k, v in tuned.ema.state_dict().items()
        ), "init_weights must restart the EMA from the loaded weights"
    other.train.resume = other.train.init_weights
    try:
        build(other, "both")
    except ValueError:
        pass
    else:
        raise AssertionError("resume and init_weights together were accepted")
    return f"resume after epoch 0 repeats the uninterrupted run (max weight diff {drift:.1e}); mismatches refused"


def check_train_step_reduces_loss(cfg: Config) -> str:
    """A real forward/backward/step on synthetic bags must be able to reduce the loss."""
    model = _tiny_model(cfg, seed=13).train()
    dataset = SyntheticBagDataset(cfg, n_studies=4, missing_slot_probability=0.0, unknown_probability=0.0)
    batch = collate_studies([dataset[i] for i in range(4)])
    optimizer = torch.optim.AdamW(model.parameter_groups(1e-3, 3e-3, 1e-4))

    losses = []
    for _ in range(6):
        logits = model(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
        output = masked_class_normalized_bce(logits, batch["targets"], batch["label_weights"])
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        optimizer.step()
        losses.append(float(output.loss.detach()))
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
    return f"loss {losses[0]:.4f} -> {losses[-1]:.4f} over 6 steps"


# --------------------------------------------------------------------------------------
# Laterality: one medial/lateral frame for right and left knees
# --------------------------------------------------------------------------------------


def check_laterality_rules() -> str:
    """Tag parsing, the geometry threshold, the flip table and the gap_ok reversal."""
    from .laterality import (
        apply_canonical,
        canonical_ops,
        image_center_x,
        side_from_center_x,
        side_from_tag,
        side_from_tag_votes,
    )

    assert side_from_tag([None, "", "B", "RIGHT"]) == "R" and side_from_tag(["l"]) == "L" and side_from_tag(["B", float("nan")]) == ""
    assert side_from_tag_votes(["L", "R", "RIGHT", None, ""]) == "R" and side_from_tag_votes(["l", "", "B"]) == "L"
    assert side_from_tag_votes(["L", "R"]) == "" and side_from_tag_votes([None, float("nan")]) == "", "a tie or no tag gives no side"
    assert side_from_center_x(-80.0, 20.0) == "R" and side_from_center_x(35.0, 20.0) == "L"
    assert side_from_center_x(-12.0, 20.0) == "" and side_from_center_x(float("nan"), 20.0) == ""
    # identity orientation, 100 x 200 pixels of 0.5 mm: centre = IPP + (200*0.5/2, 100*0.5/2)
    assert math.isclose(image_center_x([-100.0, 0, 0], [1, 0, 0, 0, 1, 0], [0.5, 0.5], 100, 200), -50.0)

    table = {
        ("R", "coronal", -1.0): (False, True),
        ("L", "coronal", -1.0): (False, False),
        ("R", "axial", 1.0): (False, True),
        ("L", "axial", 1.0): (False, False),
        # sagittal: index follows the normal; lateral is -x for a right knee, +x for a left one
        ("R", "sagittal", -1.0): (False, False),
        ("R", "sagittal", 1.0): (True, False),
        ("L", "sagittal", -1.0): (True, False),
        ("L", "sagittal", 1.0): (False, False),
        ("L", "sagittal", float("nan")): (False, False),
        ("", "coronal", -1.0): (False, False),
        ("", "sagittal", 1.0): (False, False),
    }
    for (side, plane, nx), expected in table.items():
        assert canonical_ops(side, plane, nx) == expected, f"{side} {plane} nx={nx}: {canonical_ops(side, plane, nx)} != {expected}"

    image = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
    gap_ok = np.array([True, False, True])  # gap between slices 1 and 2 is unusable
    out, gaps = apply_canonical(image, gap_ok, reverse_slices=True, flip_columns=True)
    assert np.array_equal(out, image[::-1, :, ::-1]) and out.flags["C_CONTIGUOUS"]
    assert gaps.tolist() == [True, False, True][::-1], "gap_ok must follow the slice reversal"
    out, gaps = apply_canonical(image, gap_ok, reverse_slices=False, flip_columns=False)
    assert out is image and gaps is gap_ok, "no-op must not copy"
    return "geometry (|x| >= offset) > tag vote, right knees mirrored on coronal/axial, sagittal ordered to lateral, gaps follow"


def check_laterality_derivation() -> str:
    """Geometry first, the plane slots' tag vote second, unresolved otherwise; normal_x from the selected sagittal volume."""
    from .laterality import derive_laterality

    manifest = pd.DataFrame(
        {
            STUDY_ID: ["a", "a", "b", "c", "d", "d", "e", "e", "e", "e", "f", "f", "f"],
            "volume_id": ["a_sag", "a_cor", "b_sag", "c_cor", "d_sag", "d_other_sag",
                          "e_sag", "e_cor", "e_ax", "e_other", "f_sag", "f_cor", "f_other"],
            "laterality": ["L", "", "", "", "RIGHT", "RIGHT", "R", "L", "R", "L", "L", "R", "R"],
            "normal_x": [-1.0, 0.0, 1.0, 0.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        }
    )
    selection = pd.DataFrame(
        {
            STUDY_ID: ["a", "a", "b", "b", "c", "d", "d", "e", "e", "e", "f", "f"],
            "slot": ["sagittal", "coronal", "sagittal", "axial", "coronal", "sagittal", "coronal",
                     "sagittal", "coronal", "axial", "sagittal", "coronal"],
            "volume_id": ["a_sag", "a_cor", "b_sag", "b_ax", "c_cor", "d_sag", "d_cor",
                          "e_sag", "e_cor", "e_ax", "f_sag", "f_cor"],
            "selected": [True] * 12,
            "path": ["-", "a_cor", "-", "b_ax", "c_cor", "-", "d_cor", "-", "e_cor", "e_ax", "-", "f_cor"],
        }
    )
    # a: geometry R beats the tag L; c: near the midline, no tag; e/f: near the midline, tag vote decides
    centre_x = {"a_cor": -90.0, "b_ax": -75.0, "c_cor": 5.0, "d_cor": 60.0, "e_cor": 8.0, "e_ax": 12.0, "f_cor": -4.0}
    table = derive_laterality(manifest, selection, min_offset_mm=20.0, workers=1, center_x_reader=centre_x.get).set_index(STUDY_ID)
    assert table.loc["a", "side"] == "R" and table.loc["a", "side_source"] == "geometry" and table.loc["a", "tag_side"] == "L"
    assert table.loc["b", "side"] == "R" and table.loc["b", "side_source"] == "geometry"
    assert table.loc["c", "side"] == "" and table.loc["c", "side_source"] == "unresolved"
    assert table.loc["d", "side"] == "L" and table.loc["d", "side_source"] == "geometry" and table.loc["d", "tag_side"] == "R"
    assert table.loc["d", "sagittal_normal_x"] == 1.0, "normal_x of the SELECTED sagittal volume"
    assert table.loc["e", "side"] == "R" and table.loc["e", "side_source"] == "tag", "2 R : 1 L among the selected series"
    assert table.loc["f", "side"] == "" and table.loc["f", "tag_side"] == "", "selected L : R is a tie; the unselected R must not vote"
    assert table.loc["a", "sagittal_normal_x"] == -1.0 and np.isnan(table.loc["c", "sagittal_normal_x"])

    # An extra slot (5-slot layout) neither votes nor locates: f's R tag and far-right centre must not decide.
    extra = pd.concat([selection, pd.DataFrame({STUDY_ID: ["f"], "slot": ["coronal_second"], "volume_id": ["f_other"],
                                                "selected": [True], "path": ["f_other"]})], ignore_index=True)
    centre_x["f_other"] = -80.0
    with_extra = derive_laterality(manifest, extra, min_offset_mm=20.0, workers=1, center_x_reader=centre_x.get).set_index(STUDY_ID)
    assert with_extra.loc["f", "side"] == "" and with_extra.loc["f", "tag_side"] == "", "an extra slot changed the side"
    assert with_extra.equals(table), "extra slots changed the table"
    return ("geometry wins, the plane slots' tag vote fills in (ties, unselected series and extra slots ignored), "
            "sagittal normal from the selected volume")


def check_override_keys() -> str:
    """A malformed --set key (stray quotes, typo in the section) is refused, not silently added."""
    for bad in ("'data.series_slots=[sagittal,coronal]'", "dat.image_size=320", "data..image_size=320"):
        try:
            load_config(overrides=[bad], resolve=False)
        except ValueError:
            continue
        raise AssertionError(f"override {bad!r} was accepted")
    cfg = load_config(overrides=["data.series_slots=[sagittal,coronal,axial,coronal_t1]"], resolve=False)
    assert list(cfg.data.series_slots) == ["sagittal", "coronal", "axial", "coronal_t1"]
    return "stray quotes and unknown sections are refused; a valid list override applies"


def check_filtered_slot_selection(cfg: Config, tmp_dir) -> str:
    """coronal_t1 takes a non-FS T1 that the coronal slot did not take; plane slots are unchanged."""
    from .manifest import select_series

    def volume(study, vid, plane, tr, te, seq="SE", options="", fs=0):
        return {STUDY_ID: study, SERIES_ID: f"{study}.{vid}", "volume_id": f"{study}.{vid}#g0", "volume_key": "g0",
                "path": f"/{study}/{vid}", "plane": plane, "series_description": vid, "echo_time": te,
                "repetition_time": tr, "scanning_sequence": seq, "scan_options": options, "n_slices": 24,
                "row_spacing_mm": 0.4, "col_spacing_mm": 0.4, "plane_angle_deg": 1.0, "quality_flags": "",
                "usable": True, "_fs": fs}

    rows = [
        # a: the coronal slot takes the PD FS; T1 FS (by ScanOptions), GRE and IR never qualify as T1
        volume("a", "sag_pd", "sagittal", 3000, 35, fs=1), volume("a", "ax_pd", "axial", 3000, 35, fs=1),
        volume("a", "cor_pdfs", "coronal", 3000, 35, fs=1), volume("a", "cor_t1", "coronal", 600, 10),
        volume("a", "cor_t1fs", "coronal", 600, 10, options="['SP', 'FS']"),
        volume("a", "cor_gre", "coronal", 500, 5, seq="GR"), volume("a", "cor_ir", "coronal", 600, 10, seq="['IR', 'SE']"),
        # b: the only coronal volume is a T1 - the coronal slot takes it, coronal_t1 must not duplicate it
        volume("b", "sag_pd", "sagittal", 3000, 35, fs=1), volume("b", "cor_t1", "coronal", 600, 10),
        # c: no T1 at all
        volume("c", "cor_pdfs", "coronal", 3000, 35, fs=1), volume("c", "cor_t2", "coronal", 4000, 80),
        # e: header edge cases - none of them is a usable T1, whatever the score says
        volume("e", "cor_pdfs", "coronal", 3000, 35, fs=1),
        volume("e", "tr_zero", "coronal", 0, 10), volume("e", "tr_missing", "coronal", np.nan, 10),
        volume("e", "te_missing", "coronal", 600, np.nan), volume("e", "te_long", "coronal", 900, 40),
        volume("e", "seq_missing", "coronal", 600, 10, seq=None), volume("e", "gr_se", "coronal", 600, 10, seq="['GR', 'SE']"),
        volume("e", "research", "coronal", 600, 10, seq="RM"),
    ]
    ti_set = volume("e", "ti_set", "coronal", 600, 10)
    ti_set["inversion_time"] = 150.0  # an inversion pulse without the IR token
    rows.append(ti_set)
    manifest = pd.DataFrame(rows)
    meta = pd.DataFrame({SERIES_ID: manifest[SERIES_ID], "Fluid_Sensitive": manifest["_fs"], "Fat_Suppression": manifest["_fs"]})
    manifest = manifest.drop(columns="_fs")

    four = cfg.copy()
    four.data.series_slots = ["sagittal", "coronal_t1", "coronal", "axial"]  # filtered slot listed first on purpose
    validate_config(four)
    selection = select_series(four, manifest, meta, out_path=Path(tmp_dir) / "selection_four.csv")
    chosen = {(r[STUDY_ID], r["slot"]): (r["volume_id"] if r["selected"] else None) for _, r in selection.iterrows()}
    assert chosen[("a", "coronal")] == "a.cor_pdfs#g0" and chosen[("a", "coronal_t1")] == "a.cor_t1#g0", chosen
    assert chosen[("b", "coronal")] == "b.cor_t1#g0" and chosen[("b", "coronal_t1")] is None, "duplicated the coronal volume"
    assert chosen[("c", "coronal_t1")] is None, "a T2 passed the T1 filter"
    assert chosen[("e", "coronal_t1")] is None, f"a header edge case passed the T1 filter: {chosen[('e', 'coronal_t1')]}"
    assert list(selection[selection[STUDY_ID] == "a"]["slot"]) == list(four.data.series_slots), "rows must follow the slot order"

    three = cfg.copy()
    three.data.series_slots = ["sagittal", "coronal", "axial"]
    base = select_series(three, manifest, meta, out_path=Path(tmp_dir) / "selection_three.csv")
    same = selection[selection["slot"].isin(three.data.series_slots)].reset_index(drop=True)
    assert same[["slot", "volume_id", "selected"]].equals(base[["slot", "volume_id", "selected"]]), "plane slots changed"

    for bad in (["sagittal", "sagittal_t2"], ["coronal", "coronal_t2"], ["coronal", "coronal"]):
        broken = cfg.copy()
        broken.data.series_slots = bad
        try:
            validate_config(broken)
        except ValueError:
            continue
        raise AssertionError(f"series_slots {bad} was accepted")
    return "coronal_t1 = best non-FS spin-echo T1 besides the coronal volume; missing/odd headers refused; plane slots unchanged"


def check_knee_consistent_selection(cfg: Config, tmp_dir) -> str:
    """A study holding both knees takes every slot from the knee of its best sagittal volume."""
    from .manifest import select_series

    def volume(study, vid, plane, x, res=0.4):  # a coarser in-plane resolution scores lower
        return {STUDY_ID: study, SERIES_ID: f"{study}.{vid}", "volume_id": f"{study}.{vid}#g0", "volume_key": "g0",
                "path": f"/{study}/{vid}", "plane": plane, "series_description": vid, "echo_time": 35,
                "repetition_time": 3000, "scanning_sequence": "SE", "scan_options": "", "n_slices": 24,
                "row_spacing_mm": res, "col_spacing_mm": res, "plane_angle_deg": 1.0, "quality_flags": "",
                "usable": True, "center_x_mm": x}

    rows = [
        # a: both knees; the best sagittal is the right one, the best axial the left one
        volume("a", "sag_r", "sagittal", -105.0), volume("a", "sag_l", "sagittal", 110.0, res=0.8),
        volume("a", "cor_r", "coronal", -108.0), volume("a", "cor_l", "coronal", 112.0),
        volume("a", "ax_r", "axial", -104.0, res=0.8), volume("a", "ax_l", "axial", 104.0),
        # b: both knees, but the kept (left) knee has no axial series -> the axial slot stays empty
        volume("b", "sag_l", "sagittal", 100.0), volume("b", "sag_r", "sagittal", -100.0, res=0.8),
        volume("b", "cor_l", "coronal", 98.0), volume("b", "ax_r", "axial", -101.0),
        # c: one knee whose series spread 100 mm, all on the right: untouched
        volume("c", "sag", "sagittal", -152.0), volume("c", "cor", "coronal", -80.0), volume("c", "ax_far", "axial", -52.0),
        # d: one knee near the midline, series on both sides but only 60 mm apart: untouched
        volume("d", "sag", "sagittal", -30.0), volume("d", "cor", "coronal", 30.0), volume("d", "ax", "axial", 25.0),
    ]
    manifest = pd.DataFrame(rows)
    three = cfg.copy()
    three.data.series_slots = ["sagittal", "coronal", "axial"]

    def chosen(selection):
        return {(r[STUDY_ID], r["slot"]): (r["volume_id"] if r["selected"] else None) for _, r in selection.iterrows()}

    on = chosen(select_series(three, manifest, None, out_path=Path(tmp_dir) / "selection_knee_on.csv"))
    assert (on[("a", "sagittal")], on[("a", "coronal")], on[("a", "axial")]) == ("a.sag_r#g0", "a.cor_r#g0", "a.ax_r#g0"), on
    assert on[("b", "sagittal")] == "b.sag_l#g0" and on[("b", "coronal")] == "b.cor_l#g0" and on[("b", "axial")] is None, on
    assert on[("c", "axial")] == "c.ax_far#g0" and on[("d", "coronal")] == "d.cor#g0" and on[("d", "axial")] == "d.ax#g0", on

    off_cfg = three.copy()
    off_cfg.selection.knee_consistency = False
    off = chosen(select_series(off_cfg, manifest, None, out_path=Path(tmp_dir) / "selection_knee_off.csv"))
    assert off[("a", "axial")] == "a.ax_l#g0" and off[("b", "axial")] == "b.ax_r#g0", "switched off: the old per-slot choice"
    no_column = chosen(select_series(three, manifest.drop(columns="center_x_mm"), None, out_path=Path(tmp_dir) / "selection_knee_nocol.csv"))
    assert no_column == off, "without center_x_mm the selection falls back to the per-slot choice"
    return "both knees -> every slot from the best sagittal's knee (a missing plane stays empty); one knee, spread or midline, untouched"


def check_non_knee_selection(cfg: Config, tmp_dir) -> str:
    """Ankle-coil / ankle-described series are never selected; knee-foot coils and empty coils stay."""
    from .manifest import select_series

    def volume(study, vid, plane, coil, desc=None, res=0.4):  # a coarser in-plane resolution scores lower
        return {STUDY_ID: study, SERIES_ID: f"{study}.{vid}", "volume_id": f"{study}.{vid}#g0", "volume_key": "g0",
                "path": f"/{study}/{vid}", "plane": plane, "series_description": desc or vid, "echo_time": 35,
                "repetition_time": 3000, "scanning_sequence": "SE", "scan_options": "", "n_slices": 24,
                "row_spacing_mm": res, "col_spacing_mm": res, "plane_angle_deg": 1.0, "quality_flags": "",
                "usable": True, "center_x_mm": -100.0, "coil_name": coil}

    rows = [
        # a: knee + ankle exam in one study (the train case): the sharper ankle series lose every slot
        volume("a", "sag_ak", "sagittal", "SENSE_ANKLE_8_AC", res=0.25), volume("a", "sag_knee", "sagittal", "SENSE_KNEE_16_AC"),
        volume("a", "ax_ak", "axial", "SENSE_ANKLE_8_AC", res=0.25), volume("a", "ax_knee", "axial", "SENSE_KNEE_16_AC"),
        # b: knee-foot coil and an empty coil name are knee; an ankle description is not
        volume("b", "sag", "sagittal", "QD Knee Foot"), volume("b", "ax", "axial", ""),
        volume("b", "cor_ak", "coronal", "", desc="COR T2 TOBILLO", res=0.25), volume("b", "cor", "coronal", ""),
        # c: only ankle series in a plane: the slot stays empty
        volume("c", "sag", "sagittal", "KNEE_FOOT"), volume("c", "ax_ak", "axial", "Foot_Ankle_8ch"),
    ]
    manifest = pd.DataFrame(rows)
    three = cfg.copy()
    three.data.series_slots = ["sagittal", "coronal", "axial"]

    def chosen(selection):
        return {(r[STUDY_ID], r["slot"]): (r["volume_id"] if r["selected"] else None) for _, r in selection.iterrows()}

    on = chosen(select_series(three, manifest, None, out_path=Path(tmp_dir) / "selection_nonknee_on.csv"))
    assert on[("a", "sagittal")] == "a.sag_knee#g0" and on[("a", "axial")] == "a.ax_knee#g0", on
    assert (on[("b", "sagittal")], on[("b", "coronal")], on[("b", "axial")]) == ("b.sag#g0", "b.cor#g0", "b.ax#g0"), on
    assert on[("c", "sagittal")] == "c.sag#g0" and on[("c", "axial")] is None, on

    off_cfg = three.copy()
    off_cfg.selection.non_knee_patterns = []
    off = chosen(select_series(off_cfg, manifest, None, out_path=Path(tmp_dir) / "selection_nonknee_off.csv"))
    assert off[("a", "sagittal")] == "a.sag_ak#g0" and off[("c", "axial")] == "c.ax_ak#g0", "switched off: the old choice"
    no_column = chosen(select_series(three, manifest.drop(columns="coil_name"), None, out_path=Path(tmp_dir) / "selection_nonknee_nocol.csv"))
    assert no_column[("a", "sagittal")] == "a.sag_ak#g0" and no_column[("b", "coronal")] == "b.cor#g0", (
        "without coil_name only the description filters")
    return "ankle coil / ankle description never selected (a plane with only those stays empty); knee-foot and empty coils kept"


def check_nonfs_slot_selection(cfg: Config, tmp_dir) -> str:
    """sagittal_nonfs takes a non-FS spin echo, PD > T1 > T2 before the score; fluid_te_preference=pd picks PDFS."""
    from .manifest import select_series

    def volume(study, vid, plane, tr, te, seq="SE", options="", fluid=0, fat=0, normal_x=-1.0):
        return {STUDY_ID: study, SERIES_ID: f"{study}.{vid}", "volume_id": f"{study}.{vid}#g0", "volume_key": "g0",
                "path": f"/{study}/{vid}", "plane": plane, "series_description": vid, "echo_time": te,
                "repetition_time": tr, "scanning_sequence": seq, "scan_options": options, "n_slices": 24,
                "row_spacing_mm": 0.4, "col_spacing_mm": 0.4, "plane_angle_deg": 1.0, "quality_flags": "",
                "usable": True, "normal_x": normal_x, "_fluid": fluid, "_fat": fat}

    def pdfs(study, plane="sagittal", te=35):
        return volume(study, f"{plane[:3]}_fs{te}", plane, 3000, te, fluid=1, fat=1)

    rows = [
        # a: PD beats a higher-scoring T2 (longer TE) and a T1; its own normal is recorded
        pdfs("a"), volume("a", "sag_t2", "sagittal", 4000, 80), volume("a", "sag_t1", "sagittal", 600, 10),
        volume("a", "sag_pd", "sagittal", 3000, 30, normal_x=0.99),
        # b: no PD - T1 beats T2;  c: only a T2 is left
        pdfs("b"), volume("b", "sag_t2", "sagittal", 4000, 80), volume("b", "sag_t1", "sagittal", 600, 10),
        pdfs("c"), volume("c", "sag_t2", "sagittal", 4000, 80),
        # d: nothing qualifies - GRE, IR, an inversion time, FS by ScanOptions or by the series csv,
        # fluid-sensitive without the FS flag, missing headers
        pdfs("d"), volume("d", "gre", "sagittal", 500, 5, seq="GR"), volume("d", "stir", "sagittal", 4000, 40, seq="['IR', 'SE']"),
        volume("d", "pd_fsopt", "sagittal", 3000, 30, options="['FS']"), volume("d", "pd_fatcsv", "sagittal", 3000, 30, fat=1),
        volume("d", "pd_fluid", "sagittal", 3000, 30, fluid=1), volume("d", "seq_missing", "sagittal", 3000, 30, seq=None),
        volume("d", "te_missing", "sagittal", 3000, np.nan), volume("d", "tr_missing", "sagittal", np.nan, 30),
        volume("d", "pd_nometa", "sagittal", 3000, 30),  # no series csv row: FS status unknown
        volume("d", "intermediate", "sagittal", 600, 40),  # short TR, TE above T1: neither PD, T1 nor T2
        # g: a short-TE PD TSE (TE 9 ms, long TR) is still PD and beats a T1
        pdfs("g"), volume("g", "sag_t1", "sagittal", 600, 10), volume("g", "sag_pd9", "sagittal", 3000, 9),
        # e: the only sagittal volume is a non-FS PD - the sagittal slot takes it, no duplicate
        volume("e", "sag_pd", "sagittal", 3000, 30),
        # f: PDFS and T2FS in one plane - fluid_te_preference decides
        pdfs("f", "coronal", 35), pdfs("f", "coronal", 80), pdfs("f"),
    ]
    ti_set = volume("d", "ti_set", "sagittal", 3000, 30)
    ti_set["inversion_time"] = 150.0
    rows.append(ti_set)
    manifest = pd.DataFrame(rows)
    meta = pd.DataFrame({SERIES_ID: manifest[SERIES_ID], "Fluid_Sensitive": manifest["_fluid"], "Fat_Suppression": manifest["_fat"]})
    meta = meta[meta[SERIES_ID] != "d.pd_nometa"]
    manifest = manifest.drop(columns=["_fluid", "_fat"])

    five = cfg.copy()
    five.data.series_slots = ["sagittal_nonfs", "sagittal", "coronal", "axial", "coronal_t1"]
    validate_config(five)
    selection = select_series(five, manifest, meta, out_path=Path(tmp_dir) / "selection_nonfs.csv")
    chosen = {(r[STUDY_ID], r["slot"]): (r["volume_id"] if r["selected"] else None) for _, r in selection.iterrows()}
    assert chosen[("a", "sagittal")] == "a.sag_fs35#g0", chosen
    assert chosen[("a", "sagittal_nonfs")] == "a.sag_pd#g0", f"PD did not win: {chosen[('a', 'sagittal_nonfs')]}"
    row_a = selection[(selection[STUDY_ID] == "a") & (selection["slot"] == "sagittal_nonfs")].iloc[0]
    assert row_a["filter_priority"] == 2 and row_a["normal_x"] == 0.99, row_a
    assert chosen[("b", "sagittal_nonfs")] == "b.sag_t1#g0", f"T1 did not beat T2: {chosen[('b', 'sagittal_nonfs')]}"
    assert chosen[("c", "sagittal_nonfs")] == "c.sag_t2#g0", chosen[("c", "sagittal_nonfs")]
    assert chosen[("g", "sagittal_nonfs")] == "g.sag_pd9#g0", f"a short-TE PD lost: {chosen[('g', 'sagittal_nonfs')]}"
    assert chosen[("d", "sagittal_nonfs")] is None, f"an excluded volume passed nonfs: {chosen[('d', 'sagittal_nonfs')]}"
    assert chosen[("e", "sagittal")] == "e.sag_pd#g0" and chosen[("e", "sagittal_nonfs")] is None, "duplicated the sagittal volume"
    assert chosen[("f", "coronal")] == "f.cor_fs80#g0", "fluid_te_preference=t2 (default) must keep the T2FS choice"

    four = cfg.copy()
    four.data.series_slots = ["sagittal", "coronal", "axial", "coronal_t1"]
    base = select_series(four, manifest, meta, out_path=Path(tmp_dir) / "selection_nonfs_base.csv")
    same = selection[selection["slot"].isin(four.data.series_slots)].reset_index(drop=True)
    base = base.set_index([STUDY_ID, "slot"]).loc[list(zip(same[STUDY_ID], same["slot"]))].reset_index()
    assert same[["slot", "volume_id", "selected"]].equals(base[["slot", "volume_id", "selected"]]), "other slots changed"

    # Switching off the fluid-sensitive score bonus must not let fluid-sensitive / unknown series in.
    no_bonus = five.copy()
    no_bonus.selection.prefer_fluid_sensitive = False
    loose = select_series(no_bonus, manifest, meta, out_path=Path(tmp_dir) / "selection_nobonus.csv")
    loose = loose.set_index([STUDY_ID, "slot"])
    assert not bool(loose.loc[("d", "sagittal_nonfs"), "selected"]), (
        f"prefer_fluid_sensitive=false opened the nonfs filter: {loose.loc[('d', 'sagittal_nonfs'), 'volume_id']}"
    )
    # Without any series csv nothing is known to be non-FS, so nonfs stays empty everywhere.
    blind = select_series(five, manifest, None, out_path=Path(tmp_dir) / "selection_nometa.csv")
    assert not blind[blind["slot"] == "sagittal_nonfs"]["selected"].any(), "nonfs passed without FS metadata"

    pd_pref = five.copy()
    pd_pref.selection.fluid_te_preference = "pd"
    validate_config(pd_pref)
    picked = select_series(pd_pref, manifest, meta, out_path=Path(tmp_dir) / "selection_pdpref.csv")
    picked = picked.set_index([STUDY_ID, "slot"])["volume_id"]
    assert picked[("f", "coronal")] == "f.cor_fs35#g0", "fluid_te_preference=pd did not pick the PDFS"
    bad = five.copy()
    bad.selection.fluid_te_preference = "t1"
    try:
        validate_config(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("fluid_te_preference=t1 was accepted")
    return "sagittal_nonfs = non-FS spin echo, PD > T1 > T2, FS/fluid/GRE/IR/missing headers or csv refused (also without the fluid bonus), no duplicate; te preference pd -> PDFS"


def check_second_slot_selection(cfg: Config, tmp_dir) -> str:
    """<plane>_second (Raptor 5-slot layout): nonfs PD > T1 > T2 > csv non-fluid > fluid fallback, another series."""
    from .manifest import select_series

    def volume(study, vid, plane, tr, te, seq="SE", fluid=0, fat=0, key="g0"):
        return {STUDY_ID: study, SERIES_ID: f"{study}.{vid}", "volume_id": f"{study}.{vid}#{key}", "volume_key": key,
                "path": f"/{study}/{vid}", "plane": plane, "series_description": vid, "echo_time": te,
                "repetition_time": tr, "scanning_sequence": seq, "scan_options": "", "n_slices": 24,
                "row_spacing_mm": 0.4, "col_spacing_mm": 0.4, "plane_angle_deg": 1.0, "quality_flags": "",
                "usable": True, "normal_x": -1.0, "_fluid": fluid, "_fat": fat}

    def pdfs(study, plane="sagittal"):
        return volume(study, f"{plane[:3]}_pdfs", plane, 3000, 35, fluid=1, fat=1)

    rows = [
        # a: non-FS PD beats T1 and T2 (and the T2's better TE score); coronal_second takes the coronal PD
        pdfs("a"), volume("a", "sag_t2", "sagittal", 4000, 80), volume("a", "sag_t1", "sagittal", 600, 10),
        volume("a", "sag_pd", "sagittal", 3000, 30), pdfs("a", "coronal"), volume("a", "cor_pd", "coronal", 3000, 30),
        pdfs("a", "axial"),
        # b: no strict nonfs candidate - a csv non-fluid GRE beats a fluid-sensitive T2 FS
        pdfs("b"), volume("b", "sag_gre", "sagittal", 500, 5, seq="GR"),
        volume("b", "sag_t2fs", "sagittal", 4000, 80, fluid=1, fat=1),
        # c: only fluid-sensitive series left - the soft fallback still fills the slot
        pdfs("c"), volume("c", "sag_stir", "sagittal", 4000, 40, seq="['IR', 'SE']", fluid=1, fat=1),
        # d: a single sagittal series - nothing to add
        pdfs("d"),
        # e: one series split into two volumes - its second stack is not a second series
        pdfs("e"), volume("e", "sag_pdfs", "sagittal", 3000, 35, fluid=1, fat=1, key="g1"),
    ]
    manifest = pd.DataFrame(rows)
    meta = pd.DataFrame({SERIES_ID: manifest[SERIES_ID], "Fluid_Sensitive": manifest["_fluid"],
                         "Fat_Suppression": manifest["_fat"]}).drop_duplicates(SERIES_ID)
    manifest = manifest.drop(columns=["_fluid", "_fat"])

    five = cfg.copy()
    five.data.series_slots = ["sagittal", "sagittal_second", "coronal", "coronal_second", "axial"]
    validate_config(five)
    selection = select_series(five, manifest, meta, out_path=Path(tmp_dir) / "selection_second.csv")
    chosen = {(r[STUDY_ID], r["slot"]): (r["volume_id"] if r["selected"] else None) for _, r in selection.iterrows()}
    priority = {(r[STUDY_ID], r["slot"]): r.get("filter_priority") for _, r in selection.iterrows()}
    assert chosen[("a", "sagittal")] == "a.sag_pdfs#g0", chosen
    assert chosen[("a", "sagittal_second")] == "a.sag_pd#g0" and priority[("a", "sagittal_second")] == 4, chosen
    assert chosen[("a", "coronal_second")] == "a.cor_pd#g0", chosen
    assert chosen[("b", "sagittal_second")] == "b.sag_gre#g0" and priority[("b", "sagittal_second")] == 1, chosen
    assert {chosen[("c", "sagittal")], chosen[("c", "sagittal_second")]} == {"c.sag_pdfs#g0", "c.sag_stir#g0"}, chosen
    assert priority[("c", "sagittal_second")] == 0, priority
    assert chosen[("d", "sagittal_second")] is None, chosen
    assert chosen[("e", "sagittal_second")] is None, f"another stack of the same series: {chosen[('e', 'sagittal_second')]}"
    assert list(selection[selection[STUDY_ID] == "a"]["slot"]) == list(five.data.series_slots), "rows must follow the slot order"

    three = cfg.copy()
    three.data.series_slots = ["sagittal", "coronal", "axial"]
    base = select_series(three, manifest, meta, out_path=Path(tmp_dir) / "selection_second_base.csv")
    same = selection[selection["slot"].isin(three.data.series_slots)].reset_index(drop=True)
    base = base.set_index([STUDY_ID, "slot"]).loc[list(zip(same[STUDY_ID], same["slot"]))].reset_index()
    assert same[["slot", "volume_id", "selected"]].equals(base[["slot", "volume_id", "selected"]]), "plane slots changed"

    model = _tiny_model(five)
    assert model.slot_sides == ("zones", "zones", "halves", "halves", "halves"), model.slot_sides
    return "second = another series of the plane, nonfs PD > T1 > T2 > csv non-fluid > fluid fallback; plane slots unchanged; sagittal_second pools by depth zone"


def check_unselected_cache_removed(cfg: Config, tmp_dir) -> str:
    """A (study, slot) the selection leaves empty loses its old cache entry; selected ones are kept."""
    from .preprocess import cache_path, remove_unselected_entries

    local = cfg.copy()
    local.paths.cache_dir = str(Path(tmp_dir) / "cache_prune")
    selection = pd.DataFrame(
        {
            STUDY_ID: ["a", "a", "b"],
            "slot": ["coronal", "coronal_t1", "coronal_t1"],
            "selected": [True, False, False],
        }
    )
    for study, slot in (("a", "coronal"), ("a", "coronal_t1")):  # b/coronal_t1 never existed
        path = cache_path(local, study, slot)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"old")
    removed = remove_unselected_entries(local, selection[~selection["selected"]])
    assert removed == 1, f"removed {removed}, expected 1"
    assert not cache_path(local, "a", "coronal_t1").exists(), "a stale entry of an unselected slot survived"
    assert cache_path(local, "a", "coronal").exists(), "a selected entry was removed"
    return "an unselected (study, slot) loses its old cache file, selected entries stay"


def check_dataset_rejects_stale_cache(cfg: Config, tmp_dir) -> str:
    """The dataset refuses a cache entry whose volume differs from the selection, or whose slot is empty."""
    from .dataset import StudyBagDataset
    from .preprocess import PreprocessedSeries, cache_path, preprocess_hash, write_cache_entry

    local = cfg.copy()
    local.paths.cache_dir = str(Path(tmp_dir) / "cache_stale")
    local.paths.series_selection_csv = str(Path(tmp_dir) / "selection_stale.csv")
    local.data.series_slots = ["sagittal", "coronal", "axial", "coronal_t1"]
    local.data.laterality_canonical = False
    pd.DataFrame(
        {
            STUDY_ID: ["s1"] * 4,
            "slot": local.data.series_slots,
            "volume_id": ["sag#g0", "cor#g0", pd.NA, "t1#g0"],
            "selected": [True, True, False, True],
        }
    ).to_csv(local.paths.series_selection_csv, index=False)

    def write(slot: str, volume_id: str) -> None:
        image = np.random.default_rng(0).random((6, local.data.image_size, local.data.image_size)).astype(np.float32)
        series = PreprocessedSeries(image, {"volume_id": volume_id, "prep_hash": preprocess_hash(local)})
        write_cache_entry(cache_path(local, "s1", slot), series, str(local.data.cache_dtype))

    for slot, volume in (("sagittal", "sag#g0"), ("coronal", "cor#g0"), ("coronal_t1", "t1#g0")):
        write(slot, volume)
    item = StudyBagDataset(local, ["s1"], None, train=False)[0]
    assert item["series_present_mask"].tolist() == [True, True, False, True], item["series_present_mask"]

    for slot, volume, what in (("coronal_t1", "other#g0", "another volume"), ("axial", "ax#g0", "an empty slot")):
        write(slot, volume)
        try:
            StudyBagDataset(local, ["s1"], None, train=False)[0]
        except ValueError as error:
            assert "Stale cache entry" in str(error), error
        else:
            raise AssertionError(f"a cache entry holding {what} was read")
        cache_path(local, "s1", slot).unlink()
        if slot == "coronal_t1":
            write(slot, "t1#g0")

    def expect_refusal(ids: list[str], needle: str, what: str) -> None:
        try:
            StudyBagDataset(local, ids, None, train=False)
        except ValueError as error:
            assert needle in str(error), error
        else:
            raise AssertionError(f"{what} was accepted")

    # A 3-slot selection under the 4-slot config: coronal_t1 must not silently become "missing".
    three_slot = pd.read_csv(local.paths.series_selection_csv)
    three_slot[three_slot["slot"] != "coronal_t1"].to_csv(local.paths.series_selection_csv, index=False)
    cache_path(local, "s1", "coronal_t1").unlink()
    expect_refusal(["s1"], "has no row", "a selection without the coronal_t1 slot")
    three_slot.to_csv(local.paths.series_selection_csv, index=False)

    # One selected series without a cache entry (a failed decode) is tolerated and counts as missing ...
    item = StudyBagDataset(local, ["s1"], None, train=False)[0]
    assert item["series_present_mask"].tolist() == [True, True, False, False], item["series_present_mask"]
    # ... a cache never built for the selection is not.
    more = pd.concat([three_slot.assign(**{STUDY_ID: f"s{i}"}) for i in (1, 2, 3)], ignore_index=True)
    more.to_csv(local.paths.series_selection_csv, index=False)
    expect_refusal(["s1", "s2", "s3"], "have no cache entry", "a selection whose cache was never built")
    # Inference builds the cache itself just before and reports failures: it only warns.
    item = StudyBagDataset(local, ["s1", "s2", "s3"], None, train=False, allow_unbuilt_cache=True)[1]
    assert not item["series_present_mask"].any(), "an unbuilt study must read as all-missing"
    return (
        "another volume or a file for an empty slot stops the run; so does a selection without the run's slots "
        "or an unbuilt cache; a single failed series counts as missing"
    )


class _SideAwareExpectedDataset(_InMemoryStudyBagDataset):
    """The expected bags, built by flipping the synthetic volume by hand (laterality off)."""

    SIDES: dict = {}

    def _load_slot(self, study: str, slot: str) -> tuple[np.ndarray | None, dict]:
        image, meta = super()._load_slot(study, slot)
        side, reverse_sagittal = self.SIDES[study]
        if slot in ("coronal", "axial") and side == "R":
            image = np.flip(image, axis=-1).copy()
        if slot == "sagittal" and reverse_sagittal:
            image = np.flip(image, axis=0).copy()
        return image, meta


class _OwnNormalDataset(_InMemoryStudyBagDataset):
    """Synthetic volumes whose cache meta carries a slice normal per slot (None: no normal_x key)."""

    NORMALS: dict = {}

    def _load_slot(self, study: str, slot: str) -> tuple[np.ndarray | None, dict]:
        image, meta = super()._load_slot(study, slot)
        normal = self.NORMALS.get(slot)
        return image, ({**meta, "normal_x": normal} if normal is not None else meta)


def check_sagittal_own_normal(cfg: Config, tmp_dir) -> str:
    """Each sagittal slot is ordered by its own cached normal; entries without one use the table's."""
    from pathlib import Path

    path = Path(tmp_dir) / "laterality_own_normal.csv"
    pd.DataFrame(
        {STUDY_ID: ["study0"], "side": ["L"], "side_source": ["tag"], "tag_side": ["L"], "geometry_side": ["L"],
         "center_x_mm": [70.0], "sagittal_normal_x": [-1.0]}
    ).to_csv(path, index=False)
    on = cfg.copy()
    on.data.series_slots = ["sagittal", "coronal", "axial", "sagittal_nonfs"]
    on.data.laterality_canonical = True
    on.paths.laterality_csv = str(path)
    off = on.copy()
    off.data.laterality_canonical = False
    plain = _InMemoryStudyBagDataset(off, ["study0"], None, train=False)[0]["images"]

    def reversed_slots(normals: dict) -> list[bool]:
        _OwnNormalDataset.NORMALS = normals
        images = _OwnNormalDataset(on, ["study0"], None, train=False)[0]["images"]
        return [not torch.equal(images[p], plain[p]) for p in (0, 3)]

    # L knee: lateral is +x. The table's normal (-1) reverses; an own normal of +1 keeps the order.
    assert reversed_slots({"sagittal": -1.0, "sagittal_nonfs": 1.0}) == [True, False], "own normal ignored"
    assert reversed_slots({}) == [True, True], "the table fallback was not applied"
    assert reversed_slots({"sagittal": float("nan"), "sagittal_nonfs": -1.0}) == [True, True], "a NaN normal was trusted"
    return "a second sagittal slot follows its own slice normal; old entries fall back to laterality.csv"


def check_laterality_dataset(cfg: Config, tmp_dir) -> str:
    """data.laterality_canonical flips exactly the bags the table says, validation and training alike."""
    from pathlib import Path

    path = Path(tmp_dir) / "laterality.csv"
    pd.DataFrame(
        {
            STUDY_ID: ["study0", "study1", "study2"],
            "side": ["R", "L", ""],
            "side_source": ["tag", "geometry", "unresolved"],
            "tag_side": ["R", "", ""],
            "geometry_side": ["R", "L", ""],
            "center_x_mm": [-80.0, 70.0, 3.0],
            "sagittal_normal_x": [-1.0, -1.0, -1.0],
        }
    ).to_csv(path, index=False)
    # study0 (R, nx<0): mirror coronal/axial, sagittal already lateral-ward; study1 (L, nx<0):
    # reverse sagittal only; study2: untouched.
    _SideAwareExpectedDataset.SIDES = {"study0": ("R", False), "study1": ("L", True), "study2": ("", False)}

    on = cfg.copy()
    on.data.series_slots = ["sagittal", "coronal", "axial"]
    on.data.laterality_canonical = True
    on.paths.laterality_csv = str(path)
    off = on.copy()
    off.data.laterality_canonical = False
    ids = ["study0", "study1", "study2"]
    for train in (False, True):
        actual = _InMemoryStudyBagDataset(on, ids, None, train=train)
        expected = _SideAwareExpectedDataset(off, ids, None, train=train)
        plain = _InMemoryStudyBagDataset(off, ids, None, train=train)
        for i, study in enumerate(ids):
            a, e, n = actual[(i, 1)], expected[(i, 1)], plain[(i, 1)]
            assert torch.equal(a["images"], e["images"]), f"{study} (train={train}) not in the canonical frame"
            assert a["meta"]["laterality_side"] == ["R", "L", ""][i]
            changed = not torch.equal(a["images"], n["images"])
            assert changed == (study != "study2"), f"{study}: flip applied={changed}"
    missing = on.copy()
    missing.paths.laterality_csv = str(Path(tmp_dir) / "absent.csv")
    try:
        _InMemoryStudyBagDataset(missing, ids, None, train=False)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("a missing laterality table was accepted")
    return "R mirrored on coronal/axial, L sagittal reversed, unresolved untouched; missing table refused"


# --------------------------------------------------------------------------------------
# Target attention over (slot, depth zone, map half) tokens
# --------------------------------------------------------------------------------------


def _target_attention_model(cfg: Config, enabled: bool, seed: int = 0):
    from .model import build_model

    built = cfg.copy()
    built.model.weights = "none"
    built.model.head_dropout = 0.0
    built.model.target_attention = enabled
    torch.manual_seed(seed)
    return build_model(built).eval()


def check_depth_zones() -> str:
    """Zones are thirds of the VALID windows in slice order; padding gets zone 0."""
    from .model import depth_zones

    valid = torch.zeros((1, 3, 6), dtype=torch.bool)
    valid[0, 0] = True  # 6 valid windows
    valid[0, 1, :5] = True  # 5 valid, 1 padded
    valid[0, 2, :1] = True  # a single window
    zones = depth_zones(valid, 3)[0].tolist()
    assert zones[0] == [0, 0, 1, 1, 2, 2], zones[0]
    assert zones[1] == [0, 0, 1, 1, 2, 0], zones[1]
    assert zones[2] == [0, 0, 0, 0, 0, 0], zones[2]
    return "6 -> 001122, 5 valid -> 00112 + padded 0, single window -> 0"


def check_target_attention_starts_as_head(cfg: Config) -> str:
    """Zero output weights: the enabled model reproduces the mean/max model exactly, and learns from there."""
    wide = wide_config(cfg)  # a map at least 2 wide, so the two halves differ
    base = _target_attention_model(wide, enabled=False, seed=3)
    attn = _target_attention_model(wide, enabled=True, seed=3)
    missing, unexpected = attn.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("readout.") for k in missing), (missing, unexpected)
    batch = _random_batch(wide, b=2, seed=4)
    batch["slice_valid_mask"][1, 0, 2:] = False  # padding in one slot
    batch["series_present_mask"][1, 2] = False  # a missing slot
    inputs = (batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
    with torch.no_grad():
        expected, actual = base(*inputs), attn(*inputs)
    assert torch.allclose(expected, actual, atol=1e-6), "the readout must add exactly 0 at initialisation"

    attn.train()
    attn(*inputs).sum().backward()
    readout = attn.readout
    assert readout is not None and float(readout.out_weight.grad.abs().max()) > 0, "output weights get no gradient"
    groups = attn.parameter_groups(encoder_lr=1e-4, head_lr=3e-4, weight_decay=0.0)
    head_params = {id(q) for q in groups[1]["params"]}
    assert all(id(q) in head_params for q in readout.parameters()), "the readout must use the head lr"
    n_params = sum(q.numel() for q in readout.parameters())
    return f"identical logits at init with padding and a missing slot; readout ({n_params / 1e3:.0f}k params) trains at the head lr"


def check_target_attention_masking(cfg: Config) -> str:
    """Weights sum to 1 over valid tokens, never touch padding, follow the position bias; empty studies stay finite."""
    from .model import TargetAttentionReadout, depth_zones

    torch.manual_seed(0)
    readout = TargetAttentionReadout(feature_dim=16, n_slots=3, n_targets=N_TARGETS, dim=8, n_zones=3).eval()
    tokens = torch.randn(2, 3, 6, 2, 16)
    valid = torch.ones((2, 3, 6), dtype=torch.bool)
    valid[0, 1, 4:] = False
    valid[1] = False  # a study without any window
    zones = depth_zones(valid, 3)
    with torch.no_grad():
        readout.position_bias.zero_()
        readout.position_bias[3, 0, 2, 1] = 30.0  # target 3: sagittal slot, last zone, second half
        logits, weights = readout(tokens, valid, zones, return_attention=True)
    assert torch.isfinite(logits).all() and torch.equal(logits[1], torch.zeros(N_TARGETS)), "empty study must give 0"
    sums = weights[0].sum(dim=(1, 2, 3))
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), f"weights do not sum to 1: {sums}"
    assert float(weights[0][:, 1, 4:].abs().max()) == 0.0, "padded windows received attention"
    assert float(weights[1].abs().max()) == 0.0
    focus = float(weights[0, 3, 0, 4:, 1].sum())  # zone 2 of slot 0 = windows 4, 5; half 1
    assert focus > 0.99, f"position bias did not steer target 3 (mass {focus:.3f})"
    return f"normalised over valid tokens, padding and empty studies safe, bias steers attention ({focus:.3f})"


def check_target_attention_checkpoints(cfg: Config, tmp_dir) -> str:
    """Inference rebuilds the readout from the checkpoint; older checkpoints stay plain; laterality is enforced."""
    from pathlib import Path

    from .constants import CHECKPOINT_VERSION
    from .evaluate import load_checkpoint_for_inference

    def save(model, tree, name):
        path = Path(tmp_dir) / name
        torch.save(
            {
                "version": CHECKPOINT_VERSION,
                "target_order": list(TARGETS),
                "config": tree,
                "model_description": model.describe(),
                "model": model.state_dict(),
            },
            path,
        )
        return path

    trained = cfg.copy()
    trained.model.weights = "none"
    trained.model.target_attention = True
    trained.data.laterality_canonical = True
    model = _target_attention_model(cfg, enabled=True, seed=8)
    with torch.no_grad():
        model.readout.out_weight.normal_()  # a trained readout that matters
    path = save(model, trained.to_dict(), "attn.pt")
    batch = _random_batch(cfg, b=1, seed=9)
    inputs = (batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
    caller = cfg.copy()
    caller.model.target_attention = False
    caller.data.laterality_canonical = True
    loaded, _ = load_checkpoint_for_inference(caller, path)
    with torch.no_grad():
        assert loaded.describe()["target_attention"] and torch.allclose(model(*inputs), loaded(*inputs), atol=1e-6)

    # A checkpoint written before these keys existed: plain model, trained without the frame.
    legacy = _target_attention_model(cfg, enabled=False, seed=8)
    tree = cfg.copy().to_dict()
    for key in ("target_attention", "attention_dim", "depth_zones"):
        tree["model"].pop(key, None)
    for key in ("laterality_canonical", "laterality_min_offset_mm"):
        tree["data"].pop(key, None)
    description = legacy.describe()
    for key in ("target_attention", "n_depth_zones"):
        description.pop(key, None)
    legacy_path = Path(tmp_dir) / "legacy.pt"
    torch.save(
        {"version": CHECKPOINT_VERSION, "target_order": list(TARGETS), "config": tree,
         "model_description": description, "model": legacy.state_dict()},
        legacy_path,
    )
    plain_caller = cfg.copy()
    plain_caller.model.target_attention = True  # must not decide an old checkpoint's architecture
    plain_caller.data.laterality_canonical = False
    assert not load_checkpoint_for_inference(plain_caller, legacy_path)[0].describe()["target_attention"]
    for checkpoint, wrong_frame in ((legacy_path, True), (path, False)):
        mismatched = cfg.copy()
        mismatched.data.laterality_canonical = wrong_frame
        try:
            load_checkpoint_for_inference(mismatched, checkpoint)
        except ValueError as error:
            assert "data.laterality_canonical" in str(error), error
        else:
            raise AssertionError(f"{checkpoint.name}: inputs in the wrong laterality frame were accepted")
    return "readout rebuilt from the checkpoint; legacy = plain + no frame; a laterality-frame mismatch is refused"


def check_target_attention_training_step(cfg: Config) -> str:
    """The full synthetic training path runs with target attention and the laterality keys off."""
    import tempfile
    from pathlib import Path

    from .train import run_synthetic

    small = cfg.copy()
    small.model.target_attention = True
    small.train.num_workers = 0
    small.train.eval_num_workers = 0
    with tempfile.TemporaryDirectory(prefix="knee_mri_tattn_") as tmp:
        small.paths.output_dir = str(Path(tmp))
        try:
            run_synthetic(small, n_studies=8, epochs=1, name="tattn")
            history = pd.read_csv(Path(tmp) / "tattn" / "history.csv")
        finally:
            remove_file_logging(tmp)
    assert np.isfinite(history["train_loss"]).all()
    return f"synthetic epoch with target attention: train loss {float(history['train_loss'].iloc[-1]):.4f}"


# --------------------------------------------------------------------------------------
# Side pooling: one mean/max per side of each slot
# --------------------------------------------------------------------------------------


def _side_pooling_model(cfg: Config, seed: int = 0, target_attention: bool = False):
    from .model import build_model

    built = cfg.copy()
    built.model.weights = "none"
    built.model.head_dropout = 0.0
    built.model.side_pooling = True
    built.model.target_attention = target_attention
    torch.manual_seed(seed)
    return build_model(built).eval()


def check_side_pooling_features(cfg: Config) -> str:
    """Head input = per-zone (sagittal) / per-half (coronal, axial) masked mean and max, by hand."""
    from .model import depth_zones, masked_max, masked_mean

    wide = wide_config(cfg)  # a map at least 2 wide: two distinct column halves
    wide.data.series_slots = ["sagittal", "coronal", "axial"]
    wide.data.centers_per_series = 5
    model = _side_pooling_model(wide, seed=5)
    f = model.feature_dim
    assert model.slot_sides == ("zones", "halves", "halves")
    assert model.head_in == (3 + 2 + 2) * 2 * f + 3, model.head_in

    batch = _random_batch(wide, b=2, seed=6)
    batch["slice_valid_mask"][1, 0, 3:] = False  # 3 sagittal windows: one per zone
    batch["slice_valid_mask"][1, 1, 1:] = False  # a single coronal window
    batch["series_present_mask"][1, 2] = False  # no axial series
    images, slice_valid, present = batch["images"], batch["slice_valid_mask"], batch["series_present_mask"]
    b, p, n = slice_valid.shape
    valid = slice_valid & present.unsqueeze(-1)
    with torch.no_grad():
        encoded, halves = model.encode_with_halves(images.reshape(b * p * n, *images.shape[3:]))
        features = encoded.view(b, p, n, f) * valid.unsqueeze(-1)
        halves = halves.view(b, p, n, 2, f) * valid[..., None, None]
        zones = depth_zones(valid, model.n_depth_zones)
        pooled = model.side_pooled(features, halves, valid, zones)
        logits = model(images, slice_valid, present)
        expected_logits = model.head(torch.cat([pooled, present.float()], dim=-1))
    assert torch.allclose(logits, expected_logits, atol=1e-5), "forward does not feed side_pooled to the head"

    for study in range(b):
        parts = []
        # sagittal: windows of each depth zone, in slice order
        for zone in range(3):
            keep = [i for i in range(n) if valid[study, 0, i] and int(zones[study, 0, i]) == zone]
            parts.append(("mean", features[study, 0, keep].mean(0) if keep else torch.zeros(f)))
        for zone in range(3):
            keep = [i for i in range(n) if valid[study, 0, i] and int(zones[study, 0, i]) == zone]
            parts.append(("max", features[study, 0, keep].max(0).values if keep else torch.zeros(f)))
        for slot in (1, 2):
            keep = [i for i in range(n) if valid[study, slot, i]]
            for stat in ("mean", "max"):
                for half in range(2):
                    x = halves[study, slot, keep, half]
                    value = (x.mean(0) if stat == "mean" else x.max(0).values) if keep else torch.zeros(f)
                    parts.append((stat, value))
        manual = torch.cat([v for _, v in parts])
        assert torch.allclose(pooled[study], manual, atol=1e-5), f"study {study}: side features differ from the manual pooling"
    assert float(pooled[1, -4 * f :].abs().max()) == 0.0, "a missing slot must give exact zeros"

    # What the head still has from the plain model: the slot mean and the sagittal max.
    blocks = pooled.view(b, -1, f)
    slot_mean = masked_mean(features, valid)
    assert torch.allclose((blocks[:, 6] + blocks[:, 7]) / 2, slot_mean[:, 1], atol=1e-5), "coronal half means must average to the slot mean"
    sag_max = torch.stack([blocks[:, 3], blocks[:, 4], blocks[:, 5]]).max(dim=0).values
    assert torch.allclose(sag_max, masked_max(features, valid)[:, 0], atol=1e-6), "zone maxes must give the sagittal max"
    return f"head_in {model.head_in}: zones x (mean, max) sagittal, halves x (mean, max) coronal/axial; padding and missing slots exact"


def check_side_pooling_checkpoints(cfg: Config, tmp_dir) -> str:
    """Inference rebuilds side pooling from the checkpoint; the config refuses it without the canonical frame."""
    from pathlib import Path

    from .config import validate_config
    from .constants import CHECKPOINT_VERSION
    from .evaluate import load_checkpoint_for_inference

    trained = cfg.copy()
    trained.model.weights = "none"
    trained.model.side_pooling = True
    trained.model.target_attention = True
    trained.data.laterality_canonical = True
    model = _side_pooling_model(cfg, seed=11, target_attention=True)
    path = Path(tmp_dir) / "side.pt"
    torch.save(
        {"version": CHECKPOINT_VERSION, "target_order": list(TARGETS), "config": trained.to_dict(),
         "model_description": model.describe(), "model": model.state_dict()},
        path,
    )
    caller = cfg.copy()
    caller.model.side_pooling = False
    caller.data.laterality_canonical = True
    loaded, _ = load_checkpoint_for_inference(caller, path)
    batch = _random_batch(cfg, b=1, seed=12)
    inputs = (batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
    with torch.no_grad():
        assert loaded.describe()["side_pooling"] and torch.allclose(model(*inputs), loaded(*inputs), atol=1e-6)

    wrong = cfg.copy()
    wrong.model.side_pooling = True
    wrong.data.laterality_canonical = False
    try:
        validate_config(wrong)
    except ValueError as error:
        assert "laterality_canonical" in str(error), error
    else:
        raise AssertionError("side pooling without the canonical laterality frame was accepted")
    return "side pooling rebuilt from the checkpoint (caller off); refused without data.laterality_canonical"


def check_side_pooling_training_step(cfg: Config) -> str:
    """The full synthetic training path runs with side pooling and target attention together."""
    import tempfile
    from pathlib import Path

    from .train import run_synthetic

    small = cfg.copy()
    small.model.side_pooling = True
    small.model.target_attention = True
    small.train.num_workers = 0
    small.train.eval_num_workers = 0
    with tempfile.TemporaryDirectory(prefix="knee_mri_side_") as tmp:
        small.paths.output_dir = str(Path(tmp))
        try:
            run_synthetic(small, n_studies=8, epochs=1, name="side")
            history = pd.read_csv(Path(tmp) / "side" / "history.csv")
        finally:
            remove_file_logging(tmp)
    assert np.isfinite(history["train_loss"]).all()
    return f"synthetic epoch with side pooling: train loss {float(history['train_loss'].iloc[-1]):.4f}"


# --------------------------------------------------------------------------------------
# Weight EMA
# --------------------------------------------------------------------------------------


def check_ema_update_formula() -> str:
    """ema_t = d_t * ema_{t-1} + (1 - d_t) * param_t with d_t = min(decay, (1+t)/(10+t)); buffers copied."""
    from .ema import ModelEMA

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
    ema = ModelEMA(model, decay=0.9)
    expected = {k: v.detach().clone().double() for k, v in model.named_parameters()}
    decays = []
    for t in range(1, 101):
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn_like(parameter))
            model[1].running_mean.add_(1.0)
        d = min(0.9, (1.0 + t) / (10.0 + t))
        decays.append(d)
        assert math.isclose(ema.current_decay(), d), f"step {t}: decay {ema.current_decay()} != {d}"
        ema.update(model)
        for name, parameter in model.named_parameters():
            expected[name] = d * expected[name] + (1 - d) * parameter.detach().double()
    got = dict(ema.module.named_parameters())
    err = max(float((got[k].double() - v).abs().max()) for k, v in expected.items())
    assert err < 1e-5, f"EMA deviates from the recursion by {err:.2e}"
    assert torch.equal(ema.module[1].running_mean, model[1].running_mean), "buffers must be copied, not averaged"
    assert ema.updates == 100 and decays[0] == 2 / 11 and decays[-1] == 0.9 and decays[78] < 0.9
    assert not any(p.requires_grad for p in ema.module.parameters()), "the EMA copy must not collect gradients"
    return f"100 updates match the warmed-up recursion (max err {err:.1e}; d: {decays[0]:.3f} -> {decays[-1]:.3f})"


def check_ema_skipped_step() -> str:
    """An overflow-skipped optimizer step must not move the EMA."""
    from types import SimpleNamespace

    from .ema import ModelEMA
    from .train import Trainer, TrainState, make_scheduler

    torch.manual_seed(0)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = load_config(resolve=False)
    cfg.train.grad_clip = 1.0
    trainer = SimpleNamespace(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=make_scheduler(optimizer, warmup_steps=2, total_steps=10),
        scaler=torch.amp.GradScaler("cpu", init_scale=2.0**8),
        state=TrainState(),
        ema=ModelEMA(model, decay=0.5),
    )
    x = torch.ones(4, 2)

    def step(loss_scale: float) -> None:
        optimizer.zero_grad(set_to_none=True)
        trainer.scaler.scale(model(x).sum() * loss_scale).backward()
        Trainer._optimizer_step(trainer)

    step(1.0)
    assert trainer.ema.updates == 1
    before = trainer.ema.module.weight.detach().clone()
    step(float("inf"))
    assert trainer.ema.updates == 1 and torch.equal(trainer.ema.module.weight, before), "a skipped step moved the EMA"
    step(1.0)
    assert trainer.ema.updates == 2
    return "the EMA is updated once per optimizer step actually taken, never on a skipped one"


def check_atomic_checkpoint_save(tmp) -> str:
    """A save that fails half-way (here: an unpicklable object late in the payload) keeps the
    previous checkpoint loadable and leaves no temporary file behind."""
    from .utils import atomic_torch_save

    path = Path(tmp) / "atomic" / "last.pt"
    atomic_torch_save({"epoch": 1, "weights": torch.ones(1000)}, path)
    try:
        atomic_torch_save({"epoch": 2, "weights": torch.zeros(1000), "broken": lambda: None}, path)
    except Exception:
        pass
    else:
        raise AssertionError("the unpicklable payload was saved")
    kept = torch.load(path, map_location="cpu", weights_only=False)
    assert kept["epoch"] == 1 and torch.equal(kept["weights"], torch.ones(1000)), "the previous checkpoint was damaged"
    leftovers = [f.name for f in path.parent.iterdir() if f.name != "last.pt"]
    assert not leftovers, f"temporary files left behind: {leftovers}"
    atomic_torch_save({"epoch": 3}, path)
    assert torch.load(path, map_location="cpu", weights_only=False)["epoch"] == 3, "a good save did not replace the file"
    return "failed save keeps the previous checkpoint, no temp file left; a good save replaces it"


def check_shard_sampler_partition() -> str:
    """Rank r takes every W-th study of the shared order: equal lengths, full cover, same windows,
    and the sampler's generator advances exactly as in the single-device run."""
    import itertools

    from torch.utils.data import RandomSampler

    from .distributed import ShardSampler

    def seeded(n: int) -> tuple[RandomSampler, torch.Generator]:
        generator = torch.Generator().manual_seed(3)
        return RandomSampler(range(n), generator=generator), generator

    for n, world, local_accumulation in ((16, 2, 4), (17, 2, 4), (7, 3, 1), (2, 3, 1), (5, 1, 8)):
        order = list(seeded(n)[0])
        shards = []
        for rank in range(world):
            sampler = ShardSampler(seeded(n)[0], rank, world)
            shard = list(sampler)
            assert len(shard) == len(sampler), "__len__ disagrees with the iteration"
            shards.append(shard)
        assert len({len(s) for s in shards}) == 1, f"ranks got unequal shares for n={n}, W={world}"
        assert set().union(*map(set, shards)) == set(range(n)), "a study is missing from every shard"
        assert sum(map(len, shards)) - n == (-n) % world, "more padding than needed for equal shares"
        # The k-th global window (local windows of every rank) is the single-device k-th window.
        window = local_accumulation * world
        for k in range(n // window):
            local = [s[k * local_accumulation : (k + 1) * local_accumulation] for s in shards]
            assert sorted(i for part in local for i in part) == sorted(order[k * window : (k + 1) * window]), (
                f"window {k} differs from the single-device window (n={n}, W={world})"
            )

    # The next epoch's order comes from the generator state: a full pass and a pass that stops
    # after exactly len() items (the window loop) must leave it where the single device does.
    for exhaust in (True, False):
        single, single_generator = seeded(16)
        shard, shard_generator = seeded(16)
        if exhaust:
            list(single), list(ShardSampler(shard, 1, 2))
        else:
            list(itertools.islice(iter(single), 16)), list(itertools.islice(iter(ShardSampler(shard, 1, 2)), 8))
        assert torch.equal(single_generator.get_state(), shard_generator.get_state()), (
            f"the shard sampler moved the generator differently (run to the end: {exhaust})"
        )
    return "equal shares, full cover, at most W-1 repeats, single-device windows and generator state"


class _PartlyUnlabelledSynthetic(SyntheticBagDataset):
    """Synthetic studies of which some carry no valid label at all."""

    def __init__(self, cfg: Config, unlabelled: set[int], **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        self.unlabelled = set(unlabelled)

    def label_weight_matrix(self) -> np.ndarray:
        weights = super().label_weight_matrix()
        weights[sorted(self.unlabelled)] = 0.0
        return weights

    def __getitem__(self, key):
        item = super().__getitem__(key)
        index = key[0] if isinstance(key, tuple) else key
        if int(index) in self.unlabelled:
            item["label_weights"] = torch.zeros_like(item["label_weights"])
        return item


def _ddp_check_worker(
    rank: int,
    world: int,
    cfg: Config,
    unlabelled: set[int],
    port: int,
    out: str,
    deterministic: bool = True,
    stop_after_epochs: int | None = None,
    complete_studies: bool = False,
) -> None:
    """One process of the DDP checks (CPU, gloo); rank 0 saves the outcome.

    `deterministic` switches dropout off; `stop_after_epochs` interrupts the run like a crash
    would (schedule and signature still built for train.max_epochs); `train.resume` resumes;
    `complete_studies` gives every training study all slots and full stacks.
    """
    import os

    from . import distributed
    from .model import build_model
    from .train import EvaluationReference, Trainer, _run_dir
    from .utils import seed_everything

    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world),
            "USE_LIBUV": "0",  # Windows builds have no libuv TCPStore
        }
    )
    # CPU only, so every process computes in the same fp32 numerics on any machine. (Hiding
    # the GPUs through CUDA_VISIBLE_DEVICES is not portable: "-1" crashes torch on Windows.)
    torch.cuda.is_available = lambda: False
    distributed.init_distributed()
    try:
        seed_everything(int(cfg.seed) + distributed.rank())
        shape = {"missing_slot_probability": 0.0, "short_stack_probability": 0.0} if complete_studies else {}
        train_ds = _PartlyUnlabelledSynthetic(cfg, unlabelled, n_studies=16, seed=int(cfg.seed), **shape)
        val_ds = SyntheticBagDataset(cfg, n_studies=9, seed=int(cfg.seed) + 1)  # odd: unequal validation shards
        items = [val_ds[i] for i in range(len(val_ds))]
        reference = EvaluationReference(
            study_ids=[item["study_id"] for item in items],
            values=np.stack([item["targets"].numpy() for item in items]).astype(np.float32),
            valid=np.stack([item["label_weights"].numpy() for item in items]) > 0,
        )
        trainer = Trainer(
            cfg, build_model(cfg), train_ds, val_ds, reference, _run_dir(cfg, "synthetic", "run"), mode="synthetic",
            provenance={"label_source": "synthetic", "prep_hash": "synthetic"},
        )
        if deterministic:
            # Dropout and stochastic depth draw per rank; comparing with one process needs them off.
            for module in trainer.model.modules():
                if isinstance(getattr(module, "p", None), float):
                    module.p = 0.0
        if cfg.train.get("resume"):
            trainer.load_checkpoint(cfg.train.resume)
        if stop_after_epochs is not None:
            trainer.cfg.train.max_epochs = stop_after_epochs
        initial_buffers = {name: b.detach().cpu().clone() for name, b in trainer.model.named_buffers()}
        trainer.fit()
        # Every rank's buffers after the run (a collective: all ranks take part).
        rank_buffers = distributed.all_gather_object(
            {
                "model": {name: b.detach().cpu() for name, b in trainer.model.named_buffers()},
                "ema": None if trainer.ema is None else {n: b.detach().cpu() for n, b in trainer.ema.module.named_buffers()},
            }
        )
        if distributed.is_main():
            torch.save(
                {
                    "state": trainer.model.state_dict(),
                    "history": trainer.state.history,
                    "steps": trainer.state.global_step,
                    "initial_buffers": initial_buffers,
                    "rank_buffers": rank_buffers,
                },
                out,
            )
    finally:
        distributed.cleanup()


def _free_port() -> int:
    import socket

    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def check_ddp_matches_single_process(cfg: Config) -> str:
    """2 DDP processes train the same model as one process, for every loss policy.

    CPU + gloo, so it runs on any machine. Rank 1's whole first local window is unlabelled
    while rank 0's is not: the ranks must still agree on every collective (no hang), and the
    weights after two epochs - steps, epoch boundary, sharded validation and checkpointing
    included - must equal the single-process ones, which skip those micro-batches.
    """
    import tempfile

    import torch.multiprocessing as mp
    from torch.utils.data import RandomSampler

    small = cfg.copy()
    small.train.num_workers = 0
    small.train.eval_num_workers = 0
    small.train.microbatch_studies = 1
    small.train.accumulation_steps = 8
    small.train.max_epochs = 2
    small.train.warmup_epochs = 1
    small.train.early_stopping_patience = 10
    small.train.ema_decay = 0.0
    small.train.resume = None
    small.train.init_weights = None
    small.train.amp = "fp32"
    small.augment.enabled = False
    small.augment.noise_std = 0.0
    small.model.head_dropout = 0.0
    # Epoch 0's order: rank 1 takes positions 1, 3, 5, 7 - its whole first local window.
    order = list(RandomSampler(range(16), generator=torch.Generator().manual_seed(int(small.seed))))
    unlabelled = {order[i] for i in (1, 3, 5, 7)}

    details = []
    with tempfile.TemporaryDirectory(prefix="knee_mri_ddp_") as tmp:
        for policy in ("global", "window", "microbatch"):
            outcome = {}
            for world in (1, 2):
                run_cfg = small.copy()
                run_cfg.train.loss_normalization = policy
                run_cfg.paths.output_dir = str(Path(tmp) / f"{policy}_w{world}")
                out = str(Path(tmp) / f"{policy}_w{world}.pt")
                mp.spawn(
                    _ddp_check_worker, args=(world, run_cfg, unlabelled, _free_port(), out), nprocs=world, join=True
                )
                outcome[world] = torch.load(out, map_location="cpu", weights_only=False)
            single, ddp = outcome[1], outcome[2]
            assert single["steps"] == ddp["steps"] > 0, f"{policy}: {single['steps']} vs {ddp['steps']} steps"
            worst = max(
                float((value - ddp["state"][key]).abs().max())
                for key, value in single["state"].items()
                if value.is_floating_point()
            )
            assert worst < 1e-6, f"{policy}: DDP weights differ from the single process by {worst:.2e}"
            assert len(single["history"]) == len(ddp["history"]), f"{policy}: epoch counts differ"
            for a, b in zip(single["history"], ddp["history"]):
                for column in ("train_loss", "val_selection_score"):
                    assert abs(a[column] - b[column]) < 1e-6, f"{policy} epoch {a['epoch']} {column}: {a[column]} vs {b[column]}"
                for column in ("train_empty_microbatches", "optimizer_steps"):
                    assert a[column] == b[column], f"{policy} epoch {a['epoch']} {column}: {a[column]} vs {b[column]}"
            details.append(f"{policy} {worst:.0e}")
    return "2 processes = 1 process (max weight diff: " + ", ".join(details) + "); half-unlabelled window, no hang"


def check_ddp_exact_resume(cfg: Config) -> str:
    """A 2-process run resumed from last.pt equals the uninterrupted one (dropout on: every rank's
    RNG streams are restored), and a 2-process checkpoint is refused by a single process."""
    import tempfile

    import torch.multiprocessing as mp

    from .train import build_synthetic_trainer

    small = cfg.copy()
    small.train.num_workers = 0
    small.train.eval_num_workers = 0
    small.train.microbatch_studies = 1
    small.train.accumulation_steps = 4
    small.train.max_epochs = 2
    small.train.warmup_epochs = 1
    small.train.early_stopping_patience = 10
    small.train.ema_decay = 0.0
    small.train.resume = None
    small.train.init_weights = None
    small.train.amp = "fp32"
    small.model.head_dropout = 0.5  # draws from each rank's own torch RNG

    with tempfile.TemporaryDirectory(prefix="knee_mri_ddp_resume_") as tmp:
        outcome = {}
        for run, resume, stop in (("full", None, None), ("interrupted", None, 1), ("resumed", "interrupted", None)):
            run_cfg = small.copy()
            run_cfg.paths.output_dir = str(Path(tmp) / run)
            if resume:
                run_cfg.train.resume = str(Path(tmp) / resume / "run" / "last.pt")
            out = str(Path(tmp) / f"{run}.pt")
            mp.spawn(
                _ddp_check_worker, args=(2, run_cfg, set(), _free_port(), out, False, stop), nprocs=2, join=True
            )
            outcome[run] = torch.load(out, map_location="cpu", weights_only=False)
        full, resumed = outcome["full"], outcome["resumed"]
        assert full["steps"] == resumed["steps"] > 0, "optimizer steps differ after resume"
        drift = max(float((value - resumed["state"][key]).abs().max()) for key, value in full["state"].items()
                    if value.is_floating_point())
        assert drift <= 1e-6, f"DDP weights differ after resume (max {drift:.2e})"
        assert [h["train_loss"] for h in full["history"]] == [h["train_loss"] for h in resumed["history"]]

        single = small.copy()
        single.paths.output_dir = str(Path(tmp) / "single")
        single.train.resume = str(Path(tmp) / "full" / "run" / "last.pt")
        try:
            build_synthetic_trainer(single, n_studies=16, name="single")
        except ValueError as error:
            assert "world_size" in str(error), f"the process-count mismatch is not named: {error}"
        else:
            raise AssertionError("a single process resumed a 2-process checkpoint")
        finally:
            remove_file_logging(tmp)  # Windows cannot delete an open run.log
    return f"resumed 2-process run = uninterrupted (max drift {drift:.0e}); 1-process resume refused"


def check_ddp_bn_buffers_in_sync(cfg: Config) -> str:
    """model.freeze_bn_running_stats=false under DDP: the running statistics move on every rank,
    and after each epoch all ranks (model and EMA) hold one averaged set - the one validation
    scores on every shard and the checkpoint stores.

    Needs a trainable encoder with BatchNorm: run it on `efficientnet_config` (B0, encoder
    trainable), not the configured backbone (DINOv2 has none, a frozen encoder never updates).
    """
    import tempfile

    import torch.multiprocessing as mp

    small = cfg.copy()
    small.train.num_workers = 0
    small.train.eval_num_workers = 0
    small.train.microbatch_studies = 1
    small.train.accumulation_steps = 4
    small.train.max_epochs = 1
    small.train.warmup_epochs = 1
    small.train.ema_decay = 0.9
    small.train.resume = None
    small.train.init_weights = None
    small.train.amp = "fp32"
    small.model.freeze_bn_running_stats = False

    with tempfile.TemporaryDirectory(prefix="knee_mri_ddp_bn_") as tmp:
        small.paths.output_dir = tmp
        out = str(Path(tmp) / "out.pt")
        # Complete studies: train-mode BatchNorm needs more than one triplet per micro-batch at
        # the tiny size's 1x1 maps (a single-device run would fail the same way).
        mp.spawn(_ddp_check_worker, args=(2, small, set(), _free_port(), out, False, None, True), nprocs=2, join=True)
        result = torch.load(out, map_location="cpu", weights_only=False)

    stats = [name for name in result["initial_buffers"] if name.endswith("running_mean")]
    assert stats, "the model has no BatchNorm running statistics to check"
    moved = sum(not torch.equal(result["initial_buffers"][n], result["rank_buffers"][0]["model"][n]) for n in stats)
    assert moved > 0, "freeze_bn_running_stats=false but no running statistic changed: the test checks nothing"
    for part in ("model", "ema"):
        reference = result["rank_buffers"][0][part]
        for rank, buffers in enumerate(result["rank_buffers"][1:], start=1):
            for name, value in reference.items():
                assert torch.equal(value, buffers[part][name]), f"{part} buffer {name} differs between rank 0 and rank {rank}"
    return f"{moved}/{len(stats)} BatchNorm layers updated; model and EMA buffers identical on both ranks"


def _ema_synthetic_trainer(cfg: Config, tmp, name: str, ema_decay: float, epochs: int = 2):
    from .train import build_synthetic_trainer

    small = cfg.copy()
    small.train.num_workers = 0
    small.train.eval_num_workers = 0
    small.train.microbatch_studies = 1
    small.train.accumulation_steps = 2
    small.train.early_stopping_patience = 10
    small.train.resume = None
    small.train.init_weights = None
    small.train.ema_decay = ema_decay
    small.paths.output_dir = str(tmp)
    return build_synthetic_trainer(small, n_studies=6, epochs=epochs, name=name)


def check_ema_checkpoint_roundtrip(cfg: Config) -> str:
    """best/last.pt store the EMA as "model": inference reproduces the trainer's EMA predictions."""
    import tempfile
    from pathlib import Path

    from .evaluate import load_checkpoint_for_inference

    with tempfile.TemporaryDirectory(prefix="knee_mri_ema_") as tmp:
        try:
            trainer = _ema_synthetic_trainer(cfg, Path(tmp), "ema", ema_decay=0.9)
            trainer.fit()
            history = pd.DataFrame(trainer.state.history)
            for column in ("val_selection_score_raw", "val_macro_roc_auc_raw"):
                assert column in history and history[column].notna().all(), f"history lacks {column}"
            payload = torch.load(Path(tmp) / "ema" / "last.pt", map_location="cpu", weights_only=False)
            assert payload["ema"]["updates"] == trainer.state.global_step > 0
            for key, value in trainer.ema.state_dict().items():
                assert torch.equal(payload["model"][key].cpu(), value.cpu()), f"'model' is not the EMA ({key})"
            for key, value in trainer.model.state_dict().items():
                assert torch.equal(payload["model_raw"][key].cpu(), value.cpu()), f"'model_raw' is not the raw model ({key})"
            differs = any(
                not torch.equal(payload["model"][k], payload["model_raw"][k])
                for k in payload["model"]
                if payload["model"][k].is_floating_point()
            )
            assert differs, "EMA and raw weights are identical - the average did nothing"

            _, expected = trainer.predict(trainer.val_loader)
            _, raw = trainer.predict(trainer.val_loader, trainer.model)
            loaded, _ = load_checkpoint_for_inference(trainer.cfg, Path(tmp) / "ema" / "last.pt")
            _, actual = trainer.predict(trainer.val_loader, loaded.to(trainer.device_spec.device))
            assert np.allclose(expected, actual, atol=1e-6), "the loaded checkpoint does not reproduce the EMA predictions"
            assert not np.allclose(raw, actual, atol=1e-6), "inference loaded the raw weights, not the EMA"
        finally:
            remove_file_logging(tmp)
    return "last.pt 'model' = EMA ('model_raw' kept for resume); inference reproduces the EMA scores; *_raw logged"


def check_epoch_subset_history(cfg: Config) -> str:
    """Every epoch: per-subset macros in history.csv and a predictions file, both from the same scores;
    a subset without validation studies gets no column; the selection stays on the full set."""
    import tempfile
    from pathlib import Path

    from .train import build_synthetic_trainer

    with tempfile.TemporaryDirectory(prefix="knee_mri_subsets_") as tmp:
        try:
            ids = [f"synthetic-{i:04d}" for i in range(8)]  # SyntheticBagDataset ids of the 8 validation studies
            subsets_csv = Path(tmp) / "subsets.csv"
            pd.DataFrame({STUDY_ID: ids + ["elsewhere"], "subset": ["first"] * 4 + ["second"] * 4 + ["absent"]}).to_csv(
                subsets_csv, index=False
            )
            small = cfg.copy()
            small.train.num_workers = 0
            small.train.eval_num_workers = 0
            small.train.accumulation_steps = 2
            small.train.early_stopping_patience = 10
            small.train.resume = None
            small.train.init_weights = None
            small.paths.output_dir = str(Path(tmp) / "runs")
            small.paths.reference_subsets_csv = str(subsets_csv)
            trainer = build_synthetic_trainer(small, n_studies=6, epochs=2, name="subsets")
            assert trainer.reference.study_ids == ids, "the synthetic validation ids changed"
            trainer.fit()
            run = Path(tmp) / "runs" / "subsets"
            history = pd.read_csv(run / "history.csv")
            for name in ("first", "second"):
                assert f"val_{name}_macro_soft_auc" in history and f"val_{name}_macro_roc_auc" in history, name
            assert not any("absent" in c for c in history.columns), "a subset without studies got a column"
            for epoch in history["epoch"]:
                frame = pd.read_csv(run / "epoch_predictions" / f"validation_predictions_ep{epoch:02d}.csv",
                                    dtype={STUDY_ID: str})
                assert (frame["epoch"] == epoch).all() and (frame["checkpoint"] == f"epoch_{epoch:02d}").all()
                wide = {c: frame.pivot(index=STUDY_ID, columns="target", values=c).loc[ids, list(TARGETS)]
                        for c in ("score", "reference", "reference_valid")}
                row = history.set_index("epoch").loc[epoch]
                _, full = evaluate_predictions(wide["score"].to_numpy(float), wide["reference"].to_numpy(float),
                                               wide["reference_valid"].to_numpy(bool))
                assert math.isclose(full["macro_soft_auc"], row["val_macro_soft_auc"], rel_tol=1e-6)
                _, first = evaluate_predictions(wide["score"].to_numpy(float)[:4], wide["reference"].to_numpy(float)[:4],
                                                wide["reference_valid"].to_numpy(bool)[:4])
                expected, logged = first["macro_soft_auc"], row["val_first_macro_soft_auc"]
                assert (np.isnan(expected) and np.isnan(logged)) or math.isclose(expected, logged, rel_tol=1e-6)
            best = int(history.loc[history["val_selection_score"].idxmax(), "epoch"])
            assert trainer.state.best_epoch == best, "the selection must stay on the full validation set"
        finally:
            remove_file_logging(tmp)
    return "per-subset macros in history.csv match the per-epoch prediction files; absent subset skipped; selection on the full set"


def check_epoch_checkpoints(cfg: Config) -> str:
    """eval.save_epoch_checkpoints: one inference-only file per epoch (no resume-only keys) that
    reproduces that epoch's saved validation predictions; the last one matches last.pt's model."""
    import tempfile
    from pathlib import Path

    from .constants import RESUME_ONLY_CHECKPOINT_KEYS
    from .evaluate import load_checkpoint_for_inference
    from .train import build_synthetic_trainer

    with tempfile.TemporaryDirectory(prefix="knee_mri_epoch_ckpt_") as tmp:
        try:
            small = cfg.copy()
            small.train.num_workers = 0
            small.train.eval_num_workers = 0
            small.train.accumulation_steps = 2
            small.train.early_stopping_patience = 10
            small.train.ema_decay = 0.9
            small.train.resume = None
            small.train.init_weights = None
            small.paths.output_dir = str(Path(tmp) / "runs")
            small.eval.save_epoch_predictions = True
            small.eval.save_epoch_checkpoints = True
            trainer = build_synthetic_trainer(small, n_studies=6, epochs=2, name="epochs")
            trainer.fit()
            run = Path(tmp) / "runs" / "epochs"
            epochs = list(pd.read_csv(run / "history.csv")["epoch"])
            files = sorted((run / "epoch_checkpoints").glob("ep*.pt"))
            assert [f.name for f in files] == [f"ep{e:02d}.pt" for e in epochs], [f.name for f in files]
            device = trainer.device_spec.device
            for epoch, path in zip(epochs, files):
                payload = torch.load(path, map_location="cpu", weights_only=False)
                assert payload["epoch"] == epoch, (payload["epoch"], epoch)
                leaked = [k for k in RESUME_ONLY_CHECKPOINT_KEYS if k in payload]
                assert not leaked, f"{path.name} keeps resume-only keys {leaked}"
                model, _ = load_checkpoint_for_inference(trainer.cfg, path)
                ids, scores = trainer.predict(trainer.val_loader, model.to(device))
                frame = pd.read_csv(run / "epoch_predictions" / f"validation_predictions_ep{epoch:02d}.csv",
                                    dtype={STUDY_ID: str})
                saved = frame.pivot(index=STUDY_ID, columns="target", values="score").loc[ids, list(TARGETS)]
                assert np.allclose(saved.to_numpy(float), scores, atol=1e-5), f"{path.name} does not reproduce its epoch"
            last = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
            final = torch.load(files[-1], map_location="cpu", weights_only=False)
            assert all(torch.equal(last["model"][k], final["model"][k]) for k in last["model"]), "last epoch != last.pt"
        finally:
            remove_file_logging(tmp)
    return f"{len(files)} inference-only epoch checkpoints reproduce their epoch's predictions"


def check_ema_off_is_identity(cfg: Config) -> str:
    """ema_decay = 0 leaves the payload, history and resume signature exactly as before."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="knee_mri_noema_") as tmp:
        try:
            trainer = _ema_synthetic_trainer(cfg, Path(tmp), "noema", ema_decay=0.0, epochs=1)
            trainer.fit()
            assert trainer.ema is None and trainer.selected_model is trainer.model
            payload = torch.load(Path(tmp) / "noema" / "last.pt", map_location="cpu", weights_only=False)
            assert "model_raw" not in payload and "ema" not in payload
            assert not any(c.endswith("_raw") for c in pd.DataFrame(trainer.state.history).columns)
            assert "train.ema_decay" not in trainer.resume_signature, "the signature would refuse older checkpoints"
        finally:
            remove_file_logging(tmp)
    return "EMA off: no model_raw/ema in the payload, no *_raw history columns, unchanged resume signature"


def check_exact_resume_with_ema(cfg: Config) -> str:
    """The exact-resume guarantee also holds for the raw weights AND the EMA."""
    ema_cfg = cfg.copy()
    ema_cfg.train.ema_decay = 0.9
    return check_exact_resume(ema_cfg)


# --------------------------------------------------------------------------------------
# Encoder adapters (encoders.py): architecture checks with RANDOM weights. Nothing here
# downloads or reads a pretrained file; the real-weight checks live in
# tests/test_pretrained_encoders.py and skip when the weights are not available.
# --------------------------------------------------------------------------------------

# Expected [C, Hf, Wf] of each adapter's map at 224 px (and DINOv2 at 336 px).
EXPECTED_MAPS_224 = {
    "efficientnet_b0": (1280, 7, 7),
    "efficientnet_v2_s": (1280, 7, 7),
    "radimagenet_resnet50": (2048, 7, 7),
    "dinov2_vits14": (384, 16, 16),
    "coatnet_0_rw": (768, 7, 7),
    "coatnet_rmlp_1_rw2": (768, 7, 7),
    "convnext_nano": (640, 7, 7),
    "convnext_tiny": (768, 7, 7),
    "convnextv2_tiny": (768, 7, 7),
}
NEW_BACKBONES = (
    "radimagenet_resnet50", "dinov2_vits14", "coatnet_0_rw", "coatnet_rmlp_1_rw2",
    "convnext_nano", "convnext_tiny", "convnextv2_tiny",
)


def _backbone_cfg(cfg: Config, backbone: str, **model_keys) -> Config:
    """`cfg` switched to `backbone` with random weights, its tiny size and its normalisation."""
    from .constants import RECOMMENDED_NORMALIZATION

    chosen = cfg.copy()
    chosen.model.backbone = backbone
    chosen.model.weights = "none"
    chosen.model.head_dropout = 0.0
    chosen.model.drop_path_rate = 0.0  # a ConvNeXt-only option; a check that wants it passes it
    chosen.data.image_size = WIDE_SIZES[backbone]  # 2-column maps, so the half split is real
    chosen.data.encoder_normalization = RECOMMENDED_NORMALIZATION.get(backbone, cfg.data.encoder_normalization)
    for key, value in model_keys.items():
        chosen.model[key] = value
    return chosen


def _backbone_model(cfg: Config, backbone: str, seed: int = 0, **model_keys):
    from .model import build_model

    torch.manual_seed(seed)
    return build_model(_backbone_cfg(cfg, backbone, **model_keys)).eval()


def _padded_batch(cfg: Config, b: int = 2, seed: int = 0) -> dict:
    """A batch with padded centres, an absent slot and one completely empty study."""
    batch = _random_batch(cfg, b=b, seed=seed)
    batch["slice_valid_mask"][0, 0, -1] = False  # padded centre
    batch["slice_valid_mask"][0, 2] = False  # absent slot
    batch["series_present_mask"][0, 2] = False
    batch["slice_valid_mask"][-1] = False  # a study without any series
    batch["series_present_mask"][-1] = False
    return batch


def check_encoder_feature_maps(cfg: Config) -> str:
    """Every backbone gives its documented map at 224 px and finite [B, 12] logits."""
    from .encoders import build_encoder_adapter
    from .model import EfficientNetMIL

    x = torch.randn((2, 3, 224, 224), generator=torch.Generator().manual_seed(0))
    shapes = {}
    for backbone, expected in EXPECTED_MAPS_224.items():
        torch.manual_seed(0)
        model = EfficientNetMIL(n_slots=3, weights="none", backbone=backbone, head_hidden=16, image_size=224).eval()
        encoder = model.features
        with torch.no_grad():
            feature_map = encoder(x)
        assert tuple(feature_map.shape) == (2, *expected), f"{backbone}: {tuple(feature_map.shape)} != {(2, *expected)}"
        assert encoder.out_channels == expected[0] and encoder.feature_map_size(224) == expected[1:], backbone
        assert torch.isfinite(feature_map).all(), backbone
        shapes[backbone] = tuple(feature_map.shape[1:])

        batch = {
            "images": torch.randn((1, 3, 2, 3, 224, 224), generator=torch.Generator().manual_seed(1)),
            "slice_valid_mask": torch.ones((1, 3, 2), dtype=torch.bool),
            "series_present_mask": torch.ones((1, 3), dtype=torch.bool),
        }
        with torch.no_grad():
            logits = model(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
        assert logits.shape == (1, N_TARGETS) and torch.isfinite(logits).all(), backbone
        assert model.feature_dim == expected[0] and model.head[0].in_features == 3 * 2 * expected[0] + 3, backbone

    torch.manual_seed(0)
    dino336 = build_encoder_adapter("dinov2_vits14", "none", image_size=336).eval()
    with torch.no_grad():
        big = dino336(torch.randn((1, 3, 336, 336)))
    assert tuple(big.shape) == (1, 384, 24, 24), tuple(big.shape)
    coat320 = build_encoder_adapter("coatnet_0_rw", "none", image_size=320).eval()
    with torch.no_grad():
        wide = coat320(torch.randn((1, 3, 320, 320)))
    assert tuple(wide.shape) == (1, 768, 10, 10), tuple(wide.shape)
    convnext320 = build_encoder_adapter("convnext_tiny", "none").eval()
    with torch.no_grad():
        convnext_map = convnext320(torch.randn((1, 3, 320, 320)))
    assert tuple(convnext_map.shape) == (1, 768, 10, 10), tuple(convnext_map.shape)
    return (
        f"maps at 224 px: {shapes}; dinov2 at 336 px: {tuple(big.shape[1:])}; coatnet and convnext at 320 px: "
        f"{tuple(wide.shape[1:])}, {tuple(convnext_map.shape[1:])}; finite [B, 12] logits"
    )


def check_new_backbone_masking(cfg: Config) -> str:
    """Padded centres and absent slots never reach the new encoders and cannot cause NaN."""
    counts = {}
    for backbone in NEW_BACKBONES:
        model = _backbone_model(cfg, backbone, seed=1)
        chosen = _backbone_cfg(cfg, backbone)
        batch = _padded_batch(chosen, b=3, seed=2)
        valid = batch["slice_valid_mask"] & batch["series_present_mask"].unsqueeze(-1)
        encoded = []
        hook = model.features.register_forward_hook(lambda _m, inputs, _o: encoded.append(inputs[0].shape[0]))
        try:
            with torch.no_grad():
                base = model(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
                assert sum(encoded) == int(valid.sum()), f"{backbone}: encoded {sum(encoded)} of {int(valid.sum())} valid"
                polluted = batch["images"].clone()
                polluted[~valid] = float("nan")  # would poison every output if padding were encoded
                other = model(polluted, batch["slice_valid_mask"], batch["series_present_mask"])
                encoded.clear()
                empty = model(batch["images"], torch.zeros_like(valid), torch.zeros_like(batch["series_present_mask"]))
                assert not encoded, f"{backbone}: an all-missing batch still called the encoder"
        finally:
            hook.remove()
        assert torch.isfinite(base).all() and torch.equal(base, other), f"{backbone}: padding changed the logits"
        assert torch.isfinite(empty).all(), f"{backbone}: all-missing batch is not finite"
        counts[backbone] = int(valid.sum())
    return f"only valid windows encoded {counts}; NaN in padding leaves the logits identical and finite"


def check_new_backbone_pooling_options(cfg: Config) -> str:
    """Spatial pooling, side pooling and target attention run on the new adapters' maps."""
    widths = {}
    for backbone in NEW_BACKBONES:
        chosen = _backbone_cfg(cfg, backbone)
        batch = _padded_batch(chosen, b=2, seed=3)
        channels = EXPECTED_MAPS_224[backbone][0]
        for pool, width in (("avg", channels), ("avgmax", 2 * channels), ("attention", channels)):
            model = _backbone_model(cfg, backbone, seed=4, spatial_pool=pool)
            with torch.no_grad():
                logits = model(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
            assert model.feature_dim == width and logits.shape == (2, N_TARGETS) and torch.isfinite(logits).all()

        model = _backbone_model(cfg, backbone, seed=5, side_pooling=True, target_attention=True)
        n_zones = int(model.n_depth_zones)
        slots = list(cfg.data.series_slots)  # sagittal slots split into depth zones, the others into 2 halves
        parts = sum(n_zones if slot_plane(slot) == "sagittal" else 2 for slot in slots)
        assert model.head_in == parts * 2 * channels + len(slots), f"{backbone}: head_in {model.head_in}"
        with torch.no_grad():
            triplets = batch["images"][0, 0, :2]
            global_avg, halves = model.encode_with_halves(triplets)
            logits = model(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
        assert global_avg.shape == (2, channels) and halves.shape == (2, 2, channels)
        assert torch.allclose(halves.mean(dim=1), global_avg, atol=1e-5), f"{backbone}: even-width halves must average to the map"
        assert logits.shape == (2, N_TARGETS) and torch.isfinite(logits).all(), backbone
        # The readout's output weights start at zero: it adds nothing until it has learned.
        assert float(model.readout.out_weight.detach().abs().max()) == 0.0
        widths[backbone] = model.head_in
    return f"avg/avgmax/attention, side pooling + target attention finite on the new maps (side-pooled head_in {widths})"


def check_dino_patch_grid(cfg: Config) -> str:
    """DINOv2 patch tokens -> map: prefix tokens dropped, row-major order, explicit sizes only."""
    import copy

    from .encoders import DinoV2Encoder, build_encoder_adapter, dinov2_grid

    assert dinov2_grid(224) == (16, 16) and dinov2_grid(336) == (24, 24)
    refused = []
    for size in (320, 384):
        for build in (lambda s: dinov2_grid(s), lambda s: DinoV2Encoder("none", s)):
            try:
                build(size)
            except ValueError as error:
                assert "multiple of the DINOv2 patch size 14" in str(error), error
            else:
                raise AssertionError(f"image size {size} was accepted by the DINOv2 adapter")
        bad = cfg.copy()
        bad.model.backbone, bad.model.weights = "dinov2_vits14", "lvd142m"
        bad.data.image_size, bad.data.encoder_normalization = size, "imagenet"
        try:
            validate_config(bad)
        except ValueError as error:
            assert "image_size" in str(error), error
            refused.append(size)
        else:
            raise AssertionError(f"validate_config accepted data.image_size={size} for DINOv2")

    torch.manual_seed(0)
    encoder = build_encoder_adapter("dinov2_vits14", "none", image_size=56).eval()
    x = torch.randn((2, 3, 56, 56), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ours = encoder(x)
        reference = encoder.model.forward_intermediates(
            x, indices=[len(encoder.model.blocks) - 1], norm=True, output_fmt="NCHW", intermediates_only=True
        )[0]
    assert ours.shape == reference.shape == (2, 384, 4, 4), (ours.shape, reference.shape)
    agreement = float((ours - reference).abs().max())
    assert agreement < 1e-5, f"patch map differs from timm's own NCHW intermediates by {agreement:.2e}"

    # Independent of timm's reshaping: no blocks, no norm, zero position embeddings, a CLS token
    # of 777 and a patch embedding that returns each patch's mean in channel 0. Patch (i, j) of
    # the input holds 10*i + j, so the map must read back exactly that grid and never 777.
    probe = copy.deepcopy(encoder)
    model = probe.model
    model.blocks, model.norm = torch.nn.Sequential(), torch.nn.Identity()
    with torch.no_grad():
        model.pos_embed.zero_()
        model.cls_token.fill_(777.0)
        model.patch_embed.proj.weight.zero_()
        model.patch_embed.proj.bias.zero_()
        model.patch_embed.proj.weight[0].fill_(1.0 / (3 * 14 * 14))
    grid = torch.arange(4).view(4, 1) * 10.0 + torch.arange(4).view(1, 4)
    image = grid.repeat_interleave(14, 0).repeat_interleave(14, 1).expand(1, 3, 56, 56).contiguous()
    with torch.no_grad():
        read_back = probe(image)[0, 0]
    assert torch.allclose(read_back, grid, atol=1e-4), f"patch order broken:\n{read_back}"
    with torch.no_grad():
        assert float(probe(image).abs().max()) < 700.0, "the CLS token leaked into the map"

    tokens = torch.zeros((1, encoder.num_prefix_tokens + 15, 384))
    try:
        encoder.patch_tokens_to_map(tokens, 56, 56)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a wrong patch-token count was reshaped silently")
    try:
        encoder(torch.zeros((1, 3, 70, 70)))
    except ValueError as error:
        assert "built for 56x56" in str(error), error
    else:
        raise AssertionError("an input of another size than the built one was accepted")
    return (
        f"CLS dropped ({encoder.num_prefix_tokens} prefix token), row-major 4x4 grid read back exactly, equals timm "
        f"NCHW intermediates (max diff {agreement:.1e}); sizes {refused} refused; 224 -> 16x16, 336 -> 24x24"
    )


def check_coatnet_input_size(cfg: Config) -> str:
    """CoAtNet is built for one size: other inputs are refused at runtime, sizes the stride 32 does
    not divide are refused by the adapter and the config, and pretrained names stay per variant."""
    from .encoders import build_encoder_adapter, coatnet_map_size

    assert coatnet_map_size(224) == (7, 7) and coatnet_map_size(320) == (10, 10)
    refused = []
    for size in (48, 300, 336):
        try:
            coatnet_map_size(size)
        except ValueError:
            refused.append(size)
        else:
            raise AssertionError(f"coatnet_map_size accepted {size}")
        bad = cfg.copy()
        bad.model.backbone, bad.model.weights, bad.data.image_size = "coatnet_0_rw", "none", size
        bad.data.encoder_normalization = "inception"
        try:
            validate_config(bad)
        except ValueError as error:
            assert "multiple of 32" in str(error) or "multiple of 8" in str(error), error
        else:
            raise AssertionError(f"validate_config accepted data.image_size={size} for CoAtNet")
    wrong_tag = cfg.copy()
    wrong_tag.model.backbone, wrong_tag.model.weights, wrong_tag.data.image_size = "coatnet_0_rw", "sw_in12k_ft_in1k", 64
    try:
        validate_config(wrong_tag)
    except ValueError as error:
        assert "not a pretrained source" in str(error), error
    else:
        raise AssertionError("coatnet_0_rw accepted the rmlp variant's pretrained tag")

    encoder = build_encoder_adapter("coatnet_0_rw", "none", image_size=64).eval()
    try:
        encoder(torch.zeros((1, 3, 96, 96)))
    except ValueError as error:
        assert "built for 64x64" in str(error), error
    else:
        raise AssertionError("an input of another size than the built one was accepted")
    return f"64 px -> 2x2, 224 -> 7x7, 320 -> 10x10; sizes {refused} refused; other input sizes refused at runtime"


def check_convnext_adapter(cfg: Config) -> str:
    """ConvNeXt: any input size (floor-rounded stride 32), the pretrained final LayerNorm on every
    map position, no BatchNorm, pretrained tags per architecture, and drop_path_rate as a
    ConvNeXt-only, training-only option."""
    import tempfile

    from .constants import CONVNEXT_WEIGHT_TAGS
    from .encoders import build_encoder_adapter, canonical_encoder_model_id, convnext_map_size

    sizes = {size: convnext_map_size(size) for size in (32, 224, 300, 320)}
    assert sizes == {32: (1, 1), 224: (7, 7), 300: (9, 9), 320: (10, 10)}, sizes
    try:
        convnext_map_size(16)
    except ValueError:
        pass
    else:
        raise AssertionError("convnext_map_size accepted 16 px")

    for backbone in CONVNEXT_WEIGHT_TAGS:
        torch.manual_seed(0)
        encoder = build_encoder_adapter(backbone, "none").eval()
        assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in encoder.modules()), backbone
        for size in (64, 96):  # one encoder, several input sizes
            x = torch.randn((2, 3, size, size), generator=torch.Generator().manual_seed(size))
            with torch.no_grad():
                feature_map = encoder(x)
                raw = encoder.model.forward_features(x)
            assert tuple(feature_map.shape[2:]) == encoder.feature_map_size(size), f"{backbone} at {size} px"
            assert torch.equal(feature_map, encoder.model.head.norm(raw)), f"{backbone}: final norm not applied"
            # Freshly initialised LayerNorm (weight 1, bias 0): every position is normalised over channels.
            assert float(feature_map.mean(dim=1).abs().max()) < 1e-4, f"{backbone}: map is not channel-normalised"

    refused = []
    for backbone, weights in (("convnext_nano", "dinov3_lvd1689m"), ("convnext_tiny", "sw_in1k"), ("convnextv2_tiny", "in12k_ft_in1k")):
        wrong = cfg.copy()
        wrong.model.backbone, wrong.model.weights, wrong.data.image_size = backbone, weights, 64
        wrong.data.encoder_normalization = "imagenet"
        try:
            validate_config(wrong)
        except ValueError as error:
            assert "not a pretrained source" in str(error), error
            refused.append(f"{backbone}/{weights}")
        else:
            raise AssertionError(f"{backbone} accepted the pretrained tag {weights}")

    for backbone, rate in (("efficientnet_b0", 0.1), ("coatnet_0_rw", 0.1), ("convnext_tiny", 1.0), ("convnext_tiny", -0.1)):
        bad = cfg.copy()
        bad.model.backbone, bad.model.weights, bad.model.drop_path_rate = backbone, "none", rate
        bad.data.image_size = 64
        bad.data.encoder_normalization = "inception" if backbone.startswith("coatnet_") else "imagenet"
        try:
            validate_config(bad)
        except ValueError as error:
            assert "drop_path_rate" in str(error), error
        else:
            raise AssertionError(f"validate_config accepted model.drop_path_rate={rate} for {backbone}")
    try:
        build_encoder_adapter("efficientnet_b0", "none", drop_path_rate=0.1)
    except ValueError as error:
        assert "ConvNeXt option" in str(error), error
    else:
        raise AssertionError("build_encoder_adapter passed drop_path_rate to an EfficientNet")

    torch.manual_seed(0)
    plain = build_encoder_adapter("convnext_tiny", "none")
    torch.manual_seed(0)
    dropping = build_encoder_adapter("convnext_tiny", "none", drop_path_rate=0.3)
    assert torch.equal(plain.model.stem[0].weight, dropping.model.stem[0].weight), "drop path changed the initialisation"
    x = torch.randn((2, 3, 64, 64), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert torch.equal(plain.eval()(x), dropping.eval()(x)), "drop path acts in eval mode"
        dropping.train()
        torch.manual_seed(2)
        first = dropping(x)
        torch.manual_seed(3)
        second = dropping(x)
    assert not torch.equal(first, second), "drop_path_rate=0.3 drops nothing in training"

    # The encoder id names the architecture only: a pretrained build and the weights=none rebuild
    # of inference must agree; the source stays in weights_info. A local file claims no tag; a hub
    # tag (download mocked with a local file) records exactly that tag. Wrong files are refused.
    from unittest import mock

    import huggingface_hub
    from safetensors.torch import load_file, save_file

    with tempfile.TemporaryDirectory(prefix="knee_mri_convnext_") as tmp:
        nano_file = _fake_convnext_weights("convnext_nano", tmp)
        tiny_file = _fake_convnext_weights("convnext_tiny", tmp)
        pretrained = build_encoder_adapter("convnext_nano", str(nano_file))
        assert torch.equal(pretrained.model.stem[0].weight, load_file(str(nano_file))["stem.0.weight"])
        local = pretrained.weights_info
        assert local["hf_hub_id"] is None and local["pretrained_tag"] is None and local["weights_used"] == str(nano_file), local
        with mock.patch.object(huggingface_hub, "hf_hub_download", lambda *_a, **_k: str(tiny_file)):
            hub = build_encoder_adapter("convnext_tiny", "dinov3_lvd1689m").weights_info
        assert hub["hf_hub_id"] == "timm/convnext_tiny.dinov3_lvd1689m" and hub["pretrained_tag"] == "dinov3_lvd1689m", hub

        state = {k: v.clone() for k, v in load_file(str(tiny_file)).items()}  # no memory map left open (Windows)
        partial = Path(tmp) / "partial.safetensors"
        save_file({k: v for k, v in state.items() if k != "head.norm.weight"}, str(partial))
        extra = Path(tmp) / "extra.safetensors"
        save_file({**state, "head.extra.weight": torch.zeros(1)}, str(extra))
        del state
        for backbone, path in (("convnext_tiny", nano_file), ("convnext_tiny", partial), ("convnext_tiny", extra), ("convnext_nano", tiny_file)):
            try:
                build_encoder_adapter(backbone, str(path))
            except ValueError as error:
                assert "not a timm-format" in str(error) or "does not fit" in str(error), error
            else:
                raise AssertionError(f"{backbone} loaded the wrong file {path.name}")
    bare = build_encoder_adapter("convnext_nano", "none")
    assert pretrained.describe()["encoder_model_id"] == bare.describe()["encoder_model_id"] == "timm/convnext_nano"
    for backbone, stored, expected in (
        ("convnext_tiny", "timm/convnext_tiny.dinov3_lvd1689m", "timm/convnext_tiny"),  # early checkpoints
        ("convnext_tiny", "timm/convnext_tiny", "timm/convnext_tiny"),
        ("convnext_tiny", "timm/convnext_tiny.sw_in1k", "timm/convnext_tiny.sw_in1k"),  # not its tag: kept, refused later
        ("convnext_tiny", "timm/convnext_nano.in12k_ft_in1k", "timm/convnext_nano.in12k_ft_in1k"),
        ("coatnet_0_rw", "timm/coatnet_0_rw_224.sw_in1k", "timm/coatnet_0_rw_224.sw_in1k"),
    ):
        assert canonical_encoder_model_id(backbone, stored) == expected, (backbone, stored)
    return (
        f"maps {sizes}; any input size, final LayerNorm per position, no BatchNorm; tags {refused} refused; "
        "drop_path_rate ConvNeXt-only, in [0, 1), inactive in eval; encoder id independent of the weights; "
        "local files claim no pretrained tag, wrong files refused"
    )


def _gradient_split(model) -> tuple[list[str], list[str], list[str]]:
    """(trainable with a non-zero gradient, trainable without one, frozen with a gradient)."""
    moving, silent, leaking = [], [], []
    for name, parameter in model.features.named_parameters():
        if parameter.requires_grad:
            (moving if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0 else silent).append(name)
        elif parameter.grad is not None:
            leaking.append(name)
    return moving, silent, leaking


def check_freeze_policies(cfg: Config) -> str:
    """After backward only the unfrozen units have gradients; frozen units stay in eval mode.

    ResNet with trainable BN statistics shows the two policies are separate: layer4's running
    statistics move, the frozen stem and layer1-3 keep theirs - also after a later train().
    """
    cases = [
        ("dinov2_vits14", "last_n", 2, lambda n: n.startswith(("model.blocks.10.", "model.blocks.11.", "model.norm."))),
        ("dinov2_vits14", "frozen", 0, lambda n: False),
        ("dinov2_vits14", "all", 0, lambda n: True),
        ("radimagenet_resnet50", "last_n", 1, lambda n: n.startswith("layer4.")),
        ("radimagenet_resnet50", "last_n", 3, lambda n: n.startswith(("layer2.", "layer3.", "layer4."))),
        ("radimagenet_resnet50", "frozen", 0, lambda n: False),
        ("efficientnet_b0", "last_n", 2, lambda n: n.startswith(("7.", "8."))),
        ("coatnet_0_rw", "last_n", 1, lambda n: n.startswith(("model.stages.3.", "model.norm."))),
        ("coatnet_0_rw", "last_n", 3, lambda n: n.startswith(("model.stages.1.", "model.stages.2.", "model.stages.3.", "model.norm."))),
        ("coatnet_rmlp_1_rw2", "frozen", 0, lambda n: False),
        ("convnext_tiny", "last_n", 1, lambda n: n.startswith(("model.stages.3.", "model.head.norm."))),
        ("convnext_nano", "last_n", 2, lambda n: n.startswith(("model.stages.2.", "model.stages.3.", "model.head.norm."))),
        ("convnextv2_tiny", "frozen", 0, lambda n: False),
    ]
    summary = []
    for backbone, mode, units, expected in cases:
        chosen = _backbone_cfg(cfg, backbone, encoder_trainable=mode, encoder_trainable_units=units)
        chosen.model.freeze_bn_running_stats = False
        torch.manual_seed(0)
        from .model import build_model

        model = build_model(chosen)
        model.eval()
        model.train()  # the policy must survive a later train() call
        for name, parameter in model.features.named_parameters():
            assert parameter.requires_grad == expected(name), f"{backbone}/{mode}: requires_grad wrong for {name}"
        frozen = model.features.frozen_modules()
        assert all(not m.training for f in frozen for m in f.modules()), f"{backbone}/{mode}: a frozen unit is in train mode"

        bn_before = {k: v.clone() for k, v in model.features.state_dict().items() if "running_mean" in k}
        batch = _random_batch(chosen, b=1, seed=6)
        logits = model(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
        logits.square().mean().backward()
        moving, silent, leaking = _gradient_split(model)
        assert not silent, f"{backbone}/{mode}: trainable parameters without gradient {silent[:4]}"
        assert not leaking, f"{backbone}/{mode}: frozen parameters received gradients {leaking[:4]}"
        assert all(p.grad is not None for p in model.head.parameters()), "the head must always train"
        if backbone == "radimagenet_resnet50":
            after = model.features.state_dict()
            for key, value in bn_before.items():
                changed = not torch.equal(value, after[key])
                assert changed == expected(key.replace("running_mean", "weight")), f"{mode}: BN statistics of {key} changed={changed}"
        summary.append(f"{backbone}/{mode}{units or ''}: {len(moving)} trained")
    return "; ".join(summary)


def check_parameter_groups_complete(cfg: Config) -> str:
    """Optimizer groups: every trainable parameter exactly once, frozen ones never, encoder first."""
    from .model import build_model

    cases = [
        ("efficientnet_b0", "all", 0, {}),
        ("dinov2_vits14", "last_n", 4, {"target_attention": True}),
        ("dinov2_vits14", "frozen", 0, {"spatial_pool": "attention"}),
        ("radimagenet_resnet50", "last_n", 1, {"side_pooling": True}),
        ("coatnet_0_rw", "last_n", 2, {"side_pooling": True, "target_attention": True}),
        ("coatnet_rmlp_1_rw2", "all", 0, {}),
        ("convnext_tiny", "last_n", 2, {"side_pooling": True, "target_attention": True}),
        ("convnextv2_tiny", "all", 0, {"drop_path_rate": 0.1}),
    ]
    out = []
    for backbone, mode, units, extra in cases:
        chosen = _backbone_cfg(cfg, backbone, encoder_trainable=mode, encoder_trainable_units=units, **extra)
        torch.manual_seed(0)
        model = build_model(chosen)
        groups = model.parameter_groups(encoder_lr=1e-4, head_lr=3e-4, weight_decay=1e-4)
        ids = [id(p) for g in groups for p in g["params"]]
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        assert len(ids) == len(set(ids)), f"{backbone}/{mode}: a parameter appears twice"
        assert set(ids) == trainable, f"{backbone}/{mode}: groups != trainable parameters"
        encoder_ids = {id(p) for p in model.features.parameters() if p.requires_grad}
        assert {id(p) for p in groups[0]["params"]} == encoder_ids and groups[0]["lr"] == 1e-4 and groups[1]["lr"] == 3e-4
        optimizer = torch.optim.AdamW(groups)  # an empty encoder group (frozen) must be accepted
        assert len(optimizer.param_groups) == 2
        out.append(f"{backbone}/{mode}: {len(groups[0]['params'])}+{len(groups[1]['params'])}")
    return "each trainable parameter once, frozen excluded - " + "; ".join(out)


def check_normalization_profiles(cfg: Config) -> str:
    """Each profile is applied exactly once, identically on the worker and the device path.

    The dataset (worker path) output, de-normalised with the profile, must reproduce the raw
    [0, 1] volume slices exactly - normalising twice (or never) would not. With identity
    augmentation parameters the batched device path must produce the same tensor; on a
    CUDA machine it runs on the GPU, otherwise the comparison is on the CPU (and says so).
    """
    from .constants import NORMALIZATION_PROFILES
    from .dataset import augment_batch

    assert normalization_stats(_profile_cfg(cfg, "radimagenet_torch")) == ((0.5,) * 3, (0.5,) * 3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    worst = {}
    for profile in NORMALIZATION_PROFILES:
        chosen = _profile_cfg(cfg, profile)
        mean, std = (torch.tensor(v).view(1, 3, 1, 1) for v in normalization_stats(chosen))

        worker = _InMemoryStudyBagDataset(chosen, ["study3"], None, train=True)  # augment off: worker path
        assert not worker.defer_augment
        item = worker[0]
        for p, slot in enumerate(chosen.data.series_slots):
            volume = torch.from_numpy(worker._load_slot("study3", slot)[0])
            raw = item["images"][p] * std + mean
            for s in range(raw.shape[0]):
                if not bool(item["slice_valid_mask"][p, s]):
                    assert float(item["images"][p, s].abs().max()) == 0.0
                    continue
                middle = raw[s, 1]
                j = int((volume - middle).abs().flatten(1).max(dim=1).values.argmin())
                assert float((volume[j] - middle).abs().max()) < 1e-5, f"{profile}: not normalised exactly once"

        # The deferred (device) path: the worker emits the RAW [0, 1] bag of the same centres,
        # and augment_batch normalises it. Forced on every machine - without a GPU the worker
        # would otherwise augment itself - so both paths start from the same raw image; the
        # augmentation is the identity, so only the normalisation is compared.
        deferred_cfg = chosen.copy()
        deferred_cfg.augment.enabled, deferred_cfg.augment.noise_std = True, 0.0
        deferred_cfg.data.cache_dtype = "float32"  # float16 transport would quantise these float32 test volumes
        deferred = _InMemoryStudyBagDataset(deferred_cfg, ["study3"], None, train=True)
        deferred.defer_augment = True
        batch = collate_studies([deferred[0]])
        assert "augment_params" in batch and float(batch["images"].min()) >= 0.0 and float(batch["images"].max()) <= 1.0
        assert torch.equal(batch["slice_valid_mask"][0], item["slice_valid_mask"]), "the two paths drew other centres"
        identity = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=torch.float64).repeat(1, len(chosen.data.series_slots), 1)
        got = augment_batch(
            batch["images"].float().to(device),
            identity.to(device),
            batch["slice_valid_mask"].to(device),
            batch["series_present_mask"].to(device),
            mean.view(1, 1, 1, 3, 1, 1).to(device),
            std.view(1, 1, 1, 3, 1, 1).to(device),
        ).cpu()[0]
        diff = float((got - item["images"]).abs().max())
        assert diff < 1e-4, f"{profile}: worker and {device} normalisation differ by {diff:.2e}"
        worst[profile] = diff
    where = "on the GPU" if device == "cuda" else "on the CPU only - NO GPU available, the GPU comparison did not run"
    return f"each profile applied exactly once; worker vs device path {where}, max diff {worst}"


def _profile_cfg(cfg: Config, profile: str) -> Config:
    chosen = cfg.copy()
    chosen.data.encoder_normalization = profile
    chosen.augment.enabled = False
    chosen.data.laterality_canonical = False
    return chosen


def check_legacy_efficientnet_state_keys(cfg: Config) -> str:
    """The EfficientNet adapters keep the exact state_dict keys and shapes of the pre-adapter model.

    The fixture was written by model.py at commit a9e5155 (before encoders.py); every existing
    EfficientNet checkpoint therefore loads strictly, without a key migration.
    """
    import json
    from pathlib import Path

    from .model import EfficientNetMIL

    fixture = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "legacy_efficientnet_state_keys.json"
    stored = json.loads(fixture.read_text(encoding="utf-8"))["models"]
    variants = {"avg": {}, "attention": {"spatial_pool": "attention"}, "target_attention": {"target_attention": True}}
    for name, keys in stored.items():
        backbone, variant = name.split("/")
        model = EfficientNetMIL(n_slots=3, weights="none", head_hidden=32, backbone=backbone, **variants[variant])
        now = {k: list(v.shape) for k, v in model.state_dict().items()}
        assert list(now) == list(keys), f"{name}: state_dict keys changed ({set(now) ^ set(keys)})"
        assert now == keys, f"{name}: a parameter shape changed"
    return f"{len(stored)} EfficientNet variants keep their pre-adapter state_dict keys and shapes"


def check_backbone_resume_signature(cfg: Config) -> str:
    """Resume refuses another freeze policy or adapter; an old B0 signature is unchanged; init_weights
    from another backbone is refused."""
    from types import SimpleNamespace

    from .train import EvaluationReference, Trainer

    dataset = SyntheticBagDataset(cfg, n_studies=2)
    reference = EvaluationReference(["a"], np.zeros((1, N_TARGETS), dtype=np.float32), np.ones((1, N_TARGETS), dtype=bool))

    def signature(run_cfg: Config) -> dict:
        trainer = SimpleNamespace(
            cfg=run_cfg, mode="fold", device_spec=SimpleNamespace(device=torch.device("cpu")),
            augment_device="cpu", ema_decay=0.0, reference=reference,
        )
        return Trainer._resume_signature(trainer, dataset)

    legacy = efficientnet_config(cfg)
    for key in ("encoder_trainable", "encoder_trainable_units", "drop_path_rate"):
        legacy.model.pop(key, None)  # a B0 config from before the keys existed
    current = efficientnet_config(cfg)
    current.model.encoder_trainable, current.model.encoder_trainable_units = "all", 4
    assert signature(legacy) == signature(current), "the default freeze policy changed the B0 resume signature"

    convnext = _backbone_cfg(cfg, "convnext_tiny")
    no_key = convnext.copy()
    no_key.model.pop("drop_path_rate", None)
    dropping = _backbone_cfg(cfg, "convnext_tiny", drop_path_rate=0.1)
    assert signature(no_key) == signature(convnext), "drop_path_rate=0 changed the resume signature"
    assert signature(dropping) != signature(convnext), "model.drop_path_rate was not detected by the resume check"

    dino = _backbone_cfg(cfg, "dinov2_vits14", encoder_trainable="last_n", encoder_trainable_units=4)
    base = signature(dino)
    assert base.get("encoder_adapter_version") == "dinov2_patch_tokens_v1"
    for key, value in (("encoder_trainable_units", 2), ("encoder_trainable", "frozen"), ("encoder_trainable", "all")):
        other = dino.copy()
        other.model[key] = value
        assert signature(other) != base, f"model.{key}={value} was not detected by the resume check"
    other_backbone = _backbone_cfg(cfg, "radimagenet_resnet50", encoder_trainable="last_n", encoder_trainable_units=4)
    assert signature(other_backbone) != base

    b0_payload = {"config": legacy.to_dict(), "model": {}}
    stub = SimpleNamespace(model=_backbone_model(cfg, "radimagenet_resnet50"), ema=None, _read_checkpoint=lambda _p: b0_payload)
    try:
        Trainer.load_weights(stub, "b0.pt")
    except ValueError as error:
        assert "new experiment" in str(error), error
    else:
        raise AssertionError("init_weights from an EfficientNet checkpoint into ResNet-50 was accepted")
    return (
        "freeze policy, adapter and drop path changes are refused on resume; old B0 signature unchanged; "
        "cross-backbone init refused"
    )


def check_new_backbone_grad_checkpointing(cfg: Config) -> str:
    """Backbone-specific checkpointing leaves logits and gradients unchanged - also with partly
    frozen encoders, where a reentrant checkpoint would silently drop the trainable gradients."""
    from .model import build_model

    worst = {}
    # ConvNeXt with stochastic depth on: the recomputation must replay the same drop masks.
    cases = (
        ("dinov2_vits14", 2, {}), ("radimagenet_resnet50", 2, {}), ("coatnet_0_rw", 2, {}), ("coatnet_rmlp_1_rw2", 2, {}),
        ("convnext_tiny", 2, {"drop_path_rate": 0.3}), ("convnextv2_tiny", 2, {}),
    )
    for backbone, units, extra in cases:
        results = []
        for enabled in (False, True):
            chosen = _backbone_cfg(
                cfg, backbone, encoder_trainable="last_n", encoder_trainable_units=units, grad_checkpointing=enabled, **extra
            )
            torch.manual_seed(0)
            model = build_model(chosen).train()
            batch = _random_batch(chosen, b=1, seed=7)
            logits = model(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
            logits.square().mean().backward()
            grads = {n: p.grad.clone() for n, p in model.features.named_parameters() if p.grad is not None}
            results.append((logits.detach(), grads, model.grad_checkpointing))
        (plain, plain_grads, off), (ckpt, ckpt_grads, on) = results
        assert not off and on
        assert torch.allclose(plain, ckpt, atol=1e-5), f"{backbone}: checkpointing changed the forward pass"
        assert plain_grads.keys() == ckpt_grads.keys() and plain_grads, f"{backbone}: gradients lost under checkpointing"
        worst[backbone] = max(float((plain_grads[k] - ckpt_grads[k]).abs().max()) for k in plain_grads)
        assert worst[backbone] < 1e-5, f"{backbone}: gradients differ by {worst[backbone]:.2e}"
    return f"identical logits and encoder gradients with checkpointing on partly frozen encoders (max diff {worst})"


def check_radimagenet_loader(cfg: Config, tmp_dir) -> str:
    """The RadImageNet loader on a file in the official format (random tensors, not the real weights).

    Accepts exactly the demo's Backbone state_dict (backbone.{0,1,4,5,6,7}.*), loads every
    parameter and BN buffer, and refuses a missing, extra, mis-shaped, foreign-named or
    non-finite entry instead of loading partially.
    """
    from pathlib import Path

    import torchvision

    from .encoders import RadImageNetResNet50Encoder

    torch.manual_seed(0)
    official = torch.nn.Module()
    official.backbone = torch.nn.Sequential(*list(torchvision.models.resnet50(weights=None).children())[:9])
    with torch.no_grad():
        for module in official.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.normal_()
                module.running_var.uniform_(0.5, 2.0)
                module.num_batches_tracked.fill_(7)
    state = official.state_dict()
    good = Path(tmp_dir) / "ResNet50_official_format.pt"
    torch.save(state, good)

    encoder = RadImageNetResNet50Encoder(str(good))
    loaded = encoder.state_dict()
    assert len(loaded) == len(state) == 318, (len(loaded), len(state))
    renamed = {
        "backbone.0.": "conv1.", "backbone.1.": "bn1.", "backbone.4.": "layer1.",
        "backbone.5.": "layer2.", "backbone.6.": "layer3.", "backbone.7.": "layer4.",
    }

    def torchvision_name(key: str) -> str:
        prefix = key[: key.index(".", len("backbone.")) + 1]
        return renamed[prefix] + key[len(prefix):]

    for key, value in state.items():
        assert torch.equal(loaded[torchvision_name(key)], value), f"{key} not loaded"
    assert encoder.weights_info["n_bn_buffers"] == 53 * 3 and encoder.weights_info["sha256"]

    broken = {
        "missing BN buffer": {k: v for k, v in state.items() if k != "backbone.7.2.bn3.running_var"},
        "extra classifier": {**state, "backbone.9.weight": torch.zeros(12, 2048)},
        "wrong shape": {**state, "backbone.0.weight": torch.zeros(64, 1, 7, 7)},
        "torchvision names": {torchvision_name(k): v for k, v in state.items()},
        "non-finite": {**state, "backbone.1.running_mean": torch.full((64,), float("nan"))},
    }
    for label, bad_state in broken.items():
        path = Path(tmp_dir) / f"ResNet50_{label.replace(' ', '_')}.pt"
        torch.save(bad_state, path)
        try:
            RadImageNetResNet50Encoder(str(path))
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError(f"RadImageNet loader accepted a file with {label}")
    try:
        RadImageNetResNet50Encoder(str(Path(tmp_dir) / "does_not_exist.pt"))
    except FileNotFoundError as error:
        assert "never substituted" in str(error), error
    else:
        raise AssertionError("a missing RadImageNet file did not raise")
    return f"official Backbone format loads all 318 tensors (159 BN buffers); {len(broken)} broken variants and a missing file refused"


def _fake_convnext_weights(backbone: str, directory) -> Path:
    """A timm-format pretrained file of `backbone` with random values (and the 1000-class classifier
    the real ImageNet files have), so the pretrained code path runs without the real download."""
    import timm
    from safetensors.torch import save_file

    torch.manual_seed(11)
    source = timm.create_model(backbone, pretrained=False, num_classes=1000)
    path = Path(directory) / f"{backbone}_fake_pretrained.safetensors"
    save_file({key: value.contiguous() for key, value in source.state_dict().items()}, str(path))
    return path


def check_new_backbone_training_roundtrip(cfg: Config) -> str:
    """Synthetic train epoch -> best.pt -> offline rebuild (no network, no pretrained file) -> same predictions.

    ConvNeXt trains from a (fake) pretrained file, so the reload compares a pretrained-built
    description with a `weights=none` rebuild; a checkpoint with the early tagged encoder id
    (`timm/convnext_tiny.<tag>`) still loads, another architecture id is still refused."""
    import socket
    import tempfile
    from pathlib import Path
    from unittest import mock

    from .constants import CONVNEXT_WEIGHT_TAGS, FIXED_INPUT_BACKBONES
    from .evaluate import load_checkpoint_for_inference
    from .train import build_synthetic_trainer

    def no_network(*_args, **_kwargs):
        raise RuntimeError("network access during an offline checkpoint load")

    out = []
    for backbone, pretrained in (
        ("dinov2_vits14", "lvd142m"),
        ("radimagenet_resnet50", "Z:/nowhere/RadImageNet/ResNet50.pt"),
        ("coatnet_0_rw", "sw_in1k"),
        ("convnext_tiny", "dinov3_lvd1689m"),
    ):
        run_cfg = _backbone_cfg(cfg, backbone, encoder_trainable="last_n", encoder_trainable_units=1, side_pooling=True)
        run_cfg.train.num_workers = run_cfg.train.eval_num_workers = 0
        run_cfg.train.microbatch_studies, run_cfg.train.accumulation_steps = 1, 2
        with tempfile.TemporaryDirectory(prefix=f"knee_mri_{backbone}_") as tmp:
            run_cfg.paths.output_dir = tmp
            if backbone in CONVNEXT_WEIGHT_TAGS:
                run_cfg.model.weights = str(_fake_convnext_weights(backbone, tmp))
            try:
                trainer = build_synthetic_trainer(run_cfg, n_studies=4, epochs=1, name="run")
                trainer.fit()
                best = Path(tmp) / "run" / "best.pt"
                payload = torch.load(best, map_location="cpu", weights_only=False)
                record = payload["encoder"]
                assert record["backbone"] == backbone and record["normalization"]["profile"] == run_cfg.data.encoder_normalization
                assert record["trainable"] == "last_n" and record["trainable_units"] == 1 and record["adapter_version"]
                assert payload["model_description"]["encoder_model_id"] == record["model_id"]
                if backbone in CONVNEXT_WEIGHT_TAGS:
                    # A local file: its pretraining is unknown, so no hub id / tag is claimed.
                    source = record["weights_source"]
                    assert record["model_id"] == f"timm/{backbone}", record["model_id"]
                    assert source["hf_hub_id"] is None and source["pretrained_tag"] is None, source
                    assert source["weights_used"] == run_cfg.model.weights and len(source["sha256"]) == 64, source

                # Pretend the run started from pretrained weights that this machine does not have.
                payload["config"]["model"]["weights"] = pretrained
                torch.save(payload, best)
                blocked = [mock.patch.object(socket.socket, "connect", no_network)]
                try:
                    import huggingface_hub

                    blocked.append(mock.patch.object(huggingface_hub, "hf_hub_download", no_network))
                except ImportError:  # pragma: no cover
                    pass
                for patcher in blocked:
                    patcher.start()
                try:
                    loaded, _ = load_checkpoint_for_inference(Config(payload["config"]), best)
                finally:
                    for patcher in blocked:
                        patcher.stop()
                assert loaded.weights_info["weights_used"] == "random", "inference must not load pretrained weights"

                trained = trainer.selected_model.cpu().eval()
                batch = _padded_batch(run_cfg, b=2, seed=9)
                with torch.no_grad():
                    expected = trained(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
                    actual = loaded.cpu()(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
                assert torch.equal(expected, actual), f"{backbone}: offline reload predictions differ"

                if backbone in CONVNEXT_WEIGHT_TAGS:
                    for stored_id, accepted in ((f"timm/{backbone}.{pretrained}", True), ("timm/convnext_nano", False)):
                        variant = dict(payload, model_description=dict(payload["model_description"], encoder_model_id=stored_id))
                        variant_path = Path(tmp) / "run" / "variant.pt"
                        torch.save(variant, variant_path)
                        try:
                            reloaded, _ = load_checkpoint_for_inference(Config(variant["config"]), variant_path)
                        except ValueError as error:
                            assert not accepted and "differs in ['encoder_model_id']" in str(error), error
                        else:
                            assert accepted, f"encoder_model_id {stored_id!r} was accepted for {backbone}"
                            with torch.no_grad():
                                again = reloaded.cpu()(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
                            assert torch.equal(expected, again), f"{backbone}: tagged-id checkpoint predicts differently"

                if backbone in FIXED_INPUT_BACKBONES:
                    other = Config(payload["config"])
                    other.data.image_size = 2 * int(run_cfg.data.image_size)
                    try:
                        load_checkpoint_for_inference(other, best, allow_data_overrides=["data.image_size"])
                    except ValueError as error:
                        assert f"trained at data.image_size={run_cfg.data.image_size}" in str(error), error
                    else:
                        raise AssertionError(f"a {backbone} checkpoint was rebuilt for another input size")
                out.append(f"{backbone} (loss {trainer.state.history[-1]['train_loss']:.3f})")
            finally:
                remove_file_logging(tmp)
    return "train -> best.pt -> offline strict reload gives identical logits: " + ", ".join(out)


# --------------------------------------------------------------------------------------


def run_all_checks(cfg: Config, quick: bool = False) -> list[tuple[str, bool, str]]:
    import tempfile

    # `small` keeps the configured backbone (at its tiny size): the generic MIL, training,
    # resume and checkpoint checks also exercise DINOv2 / ResNet-50 when those are configured.
    # `effnet` is for the checks about EfficientNet behaviour itself (they build a B0/V2-S).
    small = tiny_config(cfg)
    effnet = efficientnet_config(small)
    results: list[tuple[str, bool, str]] = []

    with tempfile.TemporaryDirectory(prefix="knee_mri_selftest_") as tmp_dir:
        checks: list[tuple[str, Callable[[], str]]] = [
            ("geometric_slice_sorting", check_geometric_sorting),
            ("plane_assignment", check_plane_assignment),
            ("inplane_transform_is_reorder", check_inplane_transform),
            ("true_neighbour_triplets", check_true_neighbour_triplets),
            ("short_stack_padding", check_short_stack_padding),
            ("masked_pooling", check_masked_pooling),
            ("nan_target_rejected", check_nan_target_rejected),
            ("class_normalized_loss", check_class_normalization),
            ("epoch_loss_accounting", check_epoch_loss_accounting),
            ("window_loss_matches_full", check_window_loss_matches_full),
            ("window_weight_scaling", check_window_weight_scaling),
            ("window_empty_cases", check_window_empty_cases),
            ("global_loss_matches_formula", check_global_loss_matches_formula),
            ("global_no_dilution", check_global_no_dilution),
            ("eval_loader_budget", lambda: check_eval_loader_budget(cfg)),
            ("explicit_run_selection", check_explicit_run_selection),
            ("attention_pool_starts_as_average", lambda: check_attention_pool_starts_as_average(effnet)),
            ("spatial_pool_shapes", lambda: check_spatial_pool_shapes(effnet)),
            ("focal_signal_survives_pooling", lambda: check_focal_signal_survives_pooling(effnet)),
            ("single_class_auc_is_na", check_single_class_auc),
            ("label_join_by_key", check_label_join),
            ("empty_numeric_is_unknown", check_empty_numeric_is_unknown),
            ("unmentioned_weight_per_target", check_unmentioned_weight_per_target),
            ("details_soft_roundtrip", lambda: check_details_soft_roundtrip(tmp_dir)),
            ("wide_soft_source", lambda: check_wide_soft_source(tmp_dir)),
            ("statuses_all_targets", lambda: check_statuses_all_targets(tmp_dir)),
            ("details_missing_values", lambda: check_details_missing_values(tmp_dir)),
            ("frozen_reference_roundtrip", lambda: check_frozen_reference_roundtrip(tmp_dir)),
            ("label_mask", lambda: check_label_mask(tmp_dir)),
            ("freeze_reference_from_csv", lambda: check_freeze_reference_from_csv(tmp_dir)),
            ("soft_auc_matches_roc_auc", check_soft_auc_matches_roc_auc),
            ("soft_auc_bruteforce", check_soft_auc_bruteforce),
            ("subset_metrics", lambda: check_subset_metrics(tmp_dir)),
            ("nonfinite_scores_rejected", check_nonfinite_scores_rejected),
            ("skipped_amp_step", check_skipped_amp_step),
            ("ema_update_formula", check_ema_update_formula),
            ("laterality_rules", check_laterality_rules),
            ("laterality_derivation", check_laterality_derivation),
            ("laterality_dataset", lambda: check_laterality_dataset(small, tmp_dir)),
            ("override_keys", check_override_keys),
            ("filtered_slot_selection", lambda: check_filtered_slot_selection(small, tmp_dir)),
            ("nonfs_slot_selection", lambda: check_nonfs_slot_selection(small, tmp_dir)),
            ("second_slot_selection", lambda: check_second_slot_selection(small, tmp_dir)),
            ("knee_consistent_selection", lambda: check_knee_consistent_selection(small, tmp_dir)),
            ("non_knee_selection", lambda: check_non_knee_selection(small, tmp_dir)),
            ("sagittal_own_normal", lambda: check_sagittal_own_normal(small, tmp_dir)),
            ("unselected_cache_removed", lambda: check_unselected_cache_removed(small, tmp_dir)),
            ("dataset_rejects_stale_cache", lambda: check_dataset_rejects_stale_cache(small, tmp_dir)),
            ("depth_zones", check_depth_zones),
            ("target_attention_masking", lambda: check_target_attention_masking(small)),
            ("ema_skipped_step", check_ema_skipped_step),
            ("soft_reference", check_soft_reference),
            ("bootstrap_keeps_soft", check_bootstrap_keeps_soft),
            ("crop_edge_fill", check_crop_edge_fill),
            ("foreground_extent_center", check_foreground_extent_center),
            ("patient_group_separation", lambda: check_group_separation(tmp_dir)),
            ("pseudonymous_patient_id", lambda: check_pseudonymous_patient_id(tmp_dir)),
            ("fixed_split", lambda: check_fixed_split(tmp_dir)),
            ("verify_fixed_split", lambda: check_verify_fixed_split(tmp_dir)),
            ("training_exclusion", lambda: check_training_exclusion(tmp_dir)),
            ("dataset_contract", lambda: check_dataset_contract(small)),
            ("augment_device_resolution", lambda: check_augment_device_resolution(small)),
            ("augment_device_equivalence", lambda: check_augment_device_equivalence(small)),
            ("shard_sampler_partition", check_shard_sampler_partition),
            ("atomic_checkpoint_save", lambda: check_atomic_checkpoint_save(tmp_dir)),
        ]
        if not quick:
            checks += [
                ("padding_invariance", lambda: check_padding_invariance(effnet)),
                ("missing_slot_is_zero", lambda: check_missing_slot(effnet)),
                ("masked_label_gradients", lambda: check_masked_label_gradients(effnet)),
                ("gradient_accumulation", lambda: check_accumulation(effnet)),
                ("sigmoid_not_softmax", lambda: check_sigmoid_outputs(effnet)),
                ("checkpoint_roundtrip", lambda: check_checkpoint_roundtrip(effnet, tmp_dir)),
                ("exact_resume", lambda: check_exact_resume(small)),
                ("exact_resume_with_ema", lambda: check_exact_resume_with_ema(small)),
                ("ema_checkpoint_roundtrip", lambda: check_ema_checkpoint_roundtrip(small)),
                ("ema_off_is_identity", lambda: check_ema_off_is_identity(small)),
                ("epoch_subset_history", lambda: check_epoch_subset_history(small)),
                ("epoch_checkpoints", lambda: check_epoch_checkpoints(small)),
                ("target_attention_starts_as_head", lambda: check_target_attention_starts_as_head(small)),
                ("target_attention_checkpoints", lambda: check_target_attention_checkpoints(small, tmp_dir)),
                ("target_attention_training_step", lambda: check_target_attention_training_step(small)),
                ("side_pooling_features", lambda: check_side_pooling_features(small)),
                ("side_pooling_checkpoints", lambda: check_side_pooling_checkpoints(small, tmp_dir)),
                ("side_pooling_training_step", lambda: check_side_pooling_training_step(small)),
                ("inference_architecture", lambda: check_inference_architecture(small, tmp_dir)),
                ("backbone_choice", lambda: check_backbone_choice(effnet)),
                ("grad_checkpointing_is_exact", lambda: check_grad_checkpointing_is_exact(effnet)),
                ("legacy_checkpoint_is_b0", lambda: check_legacy_checkpoint_is_b0(effnet, tmp_dir)),
                ("train_step_reduces_loss", lambda: check_train_step_reduces_loss(effnet)),
                ("epoch_reaches_workers", lambda: check_epoch_reaches_workers(small)),
                ("window_training_step", lambda: check_window_training_step(small)),
                ("global_training_step", lambda: check_global_training_step(small)),
                ("float16_transport", lambda: check_float16_transport(small)),
                ("spatial_pool_training_step", lambda: check_spatial_pool_training_step(small)),
                ("legacy_efficientnet_state_keys", lambda: check_legacy_efficientnet_state_keys(small)),
                ("encoder_feature_maps", lambda: check_encoder_feature_maps(small)),
                ("new_backbone_masking", lambda: check_new_backbone_masking(small)),
                ("new_backbone_pooling_options", lambda: check_new_backbone_pooling_options(small)),
                ("dino_patch_grid", lambda: check_dino_patch_grid(small)),
                ("coatnet_input_size", lambda: check_coatnet_input_size(small)),
                ("convnext_adapter", lambda: check_convnext_adapter(small)),
                ("freeze_policies", lambda: check_freeze_policies(small)),
                ("parameter_groups_complete", lambda: check_parameter_groups_complete(small)),
                ("normalization_profiles", lambda: check_normalization_profiles(small)),
                ("backbone_resume_signature", lambda: check_backbone_resume_signature(small)),
                ("new_backbone_grad_checkpointing", lambda: check_new_backbone_grad_checkpointing(small)),
                ("radimagenet_loader", lambda: check_radimagenet_loader(small, tmp_dir)),
                ("new_backbone_training_roundtrip", lambda: check_new_backbone_training_roundtrip(small)),
                ("ddp_matches_single_process", lambda: check_ddp_matches_single_process(small)),
                ("ddp_exact_resume", lambda: check_ddp_exact_resume(small)),
                ("ddp_bn_buffers_in_sync", lambda: check_ddp_bn_buffers_in_sync(effnet)),
            ]

        for name, function in checks:
            try:
                detail = function()
                results.append((name, True, detail))
            except Exception as exc:
                LOG.exception("Check %s failed", name)
                results.append((name, False, f"{exc.__class__.__name__}: {exc}"))
    return results
