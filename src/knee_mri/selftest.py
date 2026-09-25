"""Focused checks for the failure modes that actually break this pipeline.

They run on synthetic data so no real dataset is needed, and they exercise the real
model and gradient path - nothing that is being checked is mocked away.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd
import torch

from .config import Config, load_config, validate_config
from .constants import EFFICIENTNET_B0_FEATURES, ENCODER_FEATURES, N_TARGETS, STUDY_ID, TARGETS
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
from .splits import assert_group_disjoint, make_splits
from .train import window_sizes
from .utils import LOG, remove_file_logging


def tiny_config(cfg: Config) -> Config:
    """A small, fast variant of the real config: same code path, cheap tensors."""
    small = cfg.copy()
    small.data.image_size = 32
    small.data.centers_per_series = 4
    small.model.weights = "none"  # deliberate: the check is about masking, not pretraining
    small.model.head_hidden = 32
    small.train.microbatch_studies = 2
    return small


def _tiny_model(cfg: Config, seed: int = 0) -> EfficientNetB0MIL:
    torch.manual_seed(seed)
    return EfficientNetB0MIL(
        n_targets=N_TARGETS,
        n_slots=len(cfg.data.series_slots),
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
        run_synthetic(small, n_studies=8, epochs=1, name="pool_check")
        history = pd.read_csv(Path(tmp) / "pool_check" / "history.csv")
        remove_file_logging(tmp)
    assert np.isfinite(history["train_loss"]).all()

    # A map wider than 1x1 is needed: with a single position the softmax is always 1 and
    # the scorer has, correctly, no gradient. The selftest config would give exactly that.
    wide = cfg.copy()
    wide.data.image_size = 64  # EfficientNet stride 32 -> a 2x2 map
    model = _pool_model(wide, "attention").train()
    batch = _random_batch(wide, b=2, seed=5)
    logits = model(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
    logits.sum().backward()
    assert model.attention_pool is not None
    grad = model.attention_pool.score[2].weight.grad
    assert grad is not None and float(grad.abs().max()) > 0, "the attention scorer received no gradient"

    flat = _pool_model(cfg, "attention").train()  # image_size 32 -> 1x1 map
    single = _random_batch(cfg, b=2, seed=5)
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
    return "short stacks pad with invalid centres; full stacks use one centre per bin"


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
        run_synthetic(small, n_studies=8, epochs=2, name="global_check")
        history = pd.read_csv(Path(tmp) / "global_check" / "history.csv")
        remove_file_logging(tmp)
    assert history["train_objective"].notna().all() and np.isfinite(history["train_objective"]).all()
    assert int(history["optimizer_steps"].iloc[0]) <= 3
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
    assert all(torch.equal(v, full_params[k]) for k, v in tuned.model.state_dict().items()), "init_weights not loaded"
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


def run_all_checks(cfg: Config, quick: bool = False) -> list[tuple[str, bool, str]]:
    import tempfile

    small = tiny_config(cfg)
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
            ("attention_pool_starts_as_average", lambda: check_attention_pool_starts_as_average(small)),
            ("spatial_pool_shapes", lambda: check_spatial_pool_shapes(small)),
            ("focal_signal_survives_pooling", lambda: check_focal_signal_survives_pooling(small)),
            ("single_class_auc_is_na", check_single_class_auc),
            ("label_join_by_key", check_label_join),
            ("empty_numeric_is_unknown", check_empty_numeric_is_unknown),
            ("unmentioned_weight_per_target", check_unmentioned_weight_per_target),
            ("details_soft_roundtrip", lambda: check_details_soft_roundtrip(tmp_dir)),
            ("wide_soft_source", lambda: check_wide_soft_source(tmp_dir)),
            ("statuses_all_targets", lambda: check_statuses_all_targets(tmp_dir)),
            ("details_missing_values", lambda: check_details_missing_values(tmp_dir)),
            ("frozen_reference_roundtrip", lambda: check_frozen_reference_roundtrip(tmp_dir)),
            ("soft_auc_matches_roc_auc", check_soft_auc_matches_roc_auc),
            ("soft_auc_bruteforce", check_soft_auc_bruteforce),
            ("nonfinite_scores_rejected", check_nonfinite_scores_rejected),
            ("skipped_amp_step", check_skipped_amp_step),
            ("soft_reference", check_soft_reference),
            ("bootstrap_keeps_soft", check_bootstrap_keeps_soft),
            ("crop_edge_fill", check_crop_edge_fill),
            ("foreground_extent_center", check_foreground_extent_center),
            ("patient_group_separation", lambda: check_group_separation(tmp_dir)),
            ("pseudonymous_patient_id", lambda: check_pseudonymous_patient_id(tmp_dir)),
            ("dataset_contract", lambda: check_dataset_contract(small)),
            ("augment_device_resolution", lambda: check_augment_device_resolution(small)),
            ("augment_device_equivalence", lambda: check_augment_device_equivalence(small)),
        ]
        if not quick:
            checks += [
                ("padding_invariance", lambda: check_padding_invariance(small)),
                ("missing_slot_is_zero", lambda: check_missing_slot(small)),
                ("masked_label_gradients", lambda: check_masked_label_gradients(small)),
                ("gradient_accumulation", lambda: check_accumulation(small)),
                ("sigmoid_not_softmax", lambda: check_sigmoid_outputs(small)),
                ("checkpoint_roundtrip", lambda: check_checkpoint_roundtrip(small, tmp_dir)),
                ("exact_resume", lambda: check_exact_resume(small)),
                ("inference_architecture", lambda: check_inference_architecture(small, tmp_dir)),
                ("backbone_choice", lambda: check_backbone_choice(small)),
                ("grad_checkpointing_is_exact", lambda: check_grad_checkpointing_is_exact(small)),
                ("legacy_checkpoint_is_b0", lambda: check_legacy_checkpoint_is_b0(small, tmp_dir)),
                ("train_step_reduces_loss", lambda: check_train_step_reduces_loss(small)),
                ("epoch_reaches_workers", lambda: check_epoch_reaches_workers(small)),
                ("window_training_step", lambda: check_window_training_step(small)),
                ("global_training_step", lambda: check_global_training_step(small)),
                ("float16_transport", lambda: check_float16_transport(small)),
                ("spatial_pool_training_step", lambda: check_spatial_pool_training_step(small)),
            ]

        for name, function in checks:
            try:
                detail = function()
                results.append((name, True, detail))
            except Exception as exc:
                LOG.exception("Check %s failed", name)
                results.append((name, False, f"{exc.__class__.__name__}: {exc}"))
    return results
