"""2.5D bag dataset: fixed-order plane slots, adjacent-slice triplets, explicit masks.

Item contract
-------------
    images            [P, S, K, H, W]  float32, encoder-normalised
    slice_valid_mask  [P, S]           bool
    series_present_mask [P]            bool
    targets           [12]             finite, in [0, 1]
    label_weights     [12]             >= 0
    study_id / meta                    returned separately from the model inputs

Centre indices come from the geometrically sorted *original* stack, and a triplet is
`[i-1, i, i+1]` of that same stack - never neighbours taken from the sparse list of
sampled centres. Training samples one centre per bin, validation uses the fixed bin
midpoints and no augmentation at all.

Augmentation runs in one of two places, chosen by `augment.device`:

    cpu   the worker augments, normalises and masks the bag, exactly as the item
          contract above describes.
    cuda  the worker emits the raw `[0, 1]` bag plus an `augment_params` `[P, 7]` row,
          and the training loop calls `augment_batch` on the device. The parameters are
          still drawn per (seed, epoch, index, slot) in the worker, so the transform a
          given study receives in a given epoch does not depend on the placement.

Run on the same device with the noise field off, the two are bit-identical, and the
selftest asserts exactly that. Two things do differ in a real cuda run, neither of them a
property of this code: the noise field (numpy draws it on the CPU path, torch on the device
path) and float32 `grid_sample` rounding, which is ~1e-4 between PyTorch's CPU and CUDA
kernels. Over an epoch that compounds into a different training trajectory, the same way a
different seed would - not a better or worse one.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .config import Config
from .constants import N_TARGETS, NORMALIZATION_PROFILES, STUDY_ID, slot_plane
from .labels import LabelTable
from .laterality import apply_canonical, canonical_ops, laterality_path, load_laterality
from .manifest import load_selection, selection_path
from .preprocess import cache_path, preprocess_hash, read_cache_entry
from .utils import LOG


def normalization_stats(cfg: Config) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """(mean, std) of the data.encoder_normalization profile, applied once to the [0, 1] slices.

    The one place both augmentation paths (dataset worker and `augment_batch`), validation
    and inference take the encoder input statistics from.
    """
    profile = str(cfg.data.encoder_normalization)
    if profile not in NORMALIZATION_PROFILES:
        raise ValueError(f"data.encoder_normalization={profile!r}; known: {sorted(NORMALIZATION_PROFILES)}")
    return NORMALIZATION_PROFILES[profile]


def bin_centers(n_slices: int, n_centers: int, rng: np.random.Generator | None) -> tuple[np.ndarray, np.ndarray]:
    """Centre indices plus their validity.

    With at least `n_centers` slices the full index range is divided into `n_centers`
    bins: one random centre per bin while training, the deterministic bin midpoint in
    validation. Shorter stacks use every slice once and pad the remaining slots as
    invalid - padded centres must never gain pooling weight.
    """
    valid = np.zeros(n_centers, dtype=bool)
    centers = np.zeros(n_centers, dtype=np.int64)
    if n_slices <= 0:
        return centers, valid
    if n_slices <= n_centers:
        centers[:n_slices] = np.arange(n_slices, dtype=np.int64)
        valid[:n_slices] = True
        return centers, valid

    edges = np.linspace(0, n_slices, n_centers + 1)
    for i in range(n_centers):
        lo = int(np.floor(edges[i]))
        hi = max(lo + 1, int(np.ceil(edges[i + 1])))
        hi = min(hi, n_slices)
        if rng is None:
            centers[i] = min(n_slices - 1, (lo + hi - 1) // 2)
        else:
            centers[i] = int(rng.integers(lo, hi))
        valid[i] = True
    return centers, valid


def triplet_indices(center: int, n_slices: int, gap_ok: np.ndarray | None) -> tuple[int, int, int]:
    """`[i-1, i, i+1]` clipped at the original stack boundaries.

    Where the manifest flagged an unusable physical gap, the centre slice is repeated
    instead of crossing the gap: an edge repetition is acceptable, silently pretending
    two far-apart slices are neighbours is not.
    """
    prev_idx = max(0, center - 1)
    next_idx = min(n_slices - 1, center + 1)
    if gap_ok is not None and gap_ok.size:
        if center - 1 >= 0 and center - 1 < gap_ok.size and not gap_ok[center - 1]:
            prev_idx = center
        if center < gap_ok.size and not gap_ok[center]:
            next_idx = center
    return prev_idx, center, next_idx


def build_bag(
    image: np.ndarray,
    centers: np.ndarray,
    valid: np.ndarray,
    gap_ok: np.ndarray | None,
) -> tuple[np.ndarray, int]:
    """Gather `[S, 3, H, W]` triplets for one series."""
    n_centers = len(centers)
    n_slices, height, width = image.shape
    bag = np.zeros((n_centers, 3, height, width), dtype=np.float32)
    substitutions = 0
    for s in range(n_centers):
        if not valid[s]:
            continue
        i0, i1, i2 = triplet_indices(int(centers[s]), n_slices, gap_ok)
        if i0 == i1 and int(centers[s]) > 0:
            substitutions += 1
        if i2 == i1 and int(centers[s]) < n_slices - 1:
            substitutions += 1
        bag[s, 0] = image[i0]
        bag[s, 1] = image[i1]
        bag[s, 2] = image[i2]
    return bag, substitutions


# --------------------------------------------------------------------------------------
# Augmentation: one spatial transform per series, applied to every slice and channel
# --------------------------------------------------------------------------------------


@dataclass
class SpatialParams:
    angle_deg: float
    scale: float
    translate_x: float
    translate_y: float

    def theta(self) -> torch.Tensor:
        angle = np.deg2rad(self.angle_deg)
        cos, sin = float(np.cos(angle)), float(np.sin(angle))
        inv = 1.0 / max(self.scale, 1e-3)
        return torch.tensor(
            [[cos * inv, -sin * inv, self.translate_x], [sin * inv, cos * inv, self.translate_y]],
            dtype=torch.float32,
        )


@dataclass
class IntensityParams:
    gain: float
    bias: float
    gamma: float


# Column order of the flat per-series parameter vector handed to `augment_batch`.
AUGMENT_PARAM_COLUMNS = ("angle_deg", "scale", "translate_x", "translate_y", "gain", "bias", "gamma")


def sample_spatial(cfg: Config, rng: np.random.Generator) -> SpatialParams:
    aug = cfg.augment
    return SpatialParams(
        angle_deg=float(rng.uniform(-aug.rotation_deg, aug.rotation_deg)),
        scale=float(rng.uniform(aug.scale_range[0], aug.scale_range[1])),
        translate_x=float(rng.uniform(-aug.translate_frac, aug.translate_frac)) * 2.0,
        translate_y=float(rng.uniform(-aug.translate_frac, aug.translate_frac)) * 2.0,
    )


def sample_intensity(cfg: Config, rng: np.random.Generator) -> IntensityParams:
    aug = cfg.augment
    return IntensityParams(
        gain=1.0 + float(rng.uniform(-aug.intensity_gain, aug.intensity_gain)),
        bias=float(rng.uniform(-aug.intensity_bias, aug.intensity_bias)),
        gamma=float(rng.uniform(aug.gamma_range[0], aug.gamma_range[1])),
    )


def pack_augment_params(spatial: SpatialParams, intensity: IntensityParams) -> list[float]:
    """One series worth of augmentation, in `AUGMENT_PARAM_COLUMNS` order."""
    return [
        spatial.angle_deg,
        spatial.scale,
        spatial.translate_x,
        spatial.translate_y,
        intensity.gain,
        intensity.bias,
        intensity.gamma,
    ]


def apply_spatial(bag: torch.Tensor, params: SpatialParams) -> torch.Tensor:
    """Apply one affine transform to a `[S, K, H, W]` bag - identically for every slice."""
    s, k, h, w = bag.shape
    flat = bag.reshape(s * k, 1, h, w)
    theta = params.theta().unsqueeze(0).expand(flat.shape[0], 2, 3)
    grid = torch.nn.functional.affine_grid(theta, size=(flat.shape[0], 1, h, w), align_corners=False)
    out = torch.nn.functional.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out.reshape(s, k, h, w)


def apply_intensity(
    bag: torch.Tensor,
    cfg: Config,
    rng: np.random.Generator,
    params: IntensityParams | None = None,
) -> torch.Tensor:
    """Mild, series-consistent intensity jitter. No independent per-channel colour jitter:
    the three channels are neighbouring slices of the same anatomy.

    `params` may be passed in when they were already drawn from `rng`. The noise field is
    always drawn here, so the sequence taken from `rng` is the same either way.
    """
    aug = cfg.augment
    if params is None:
        params = sample_intensity(cfg, rng)
    out = torch.clamp(bag * params.gain + params.bias, 0.0, 1.0)
    out = torch.clamp(out, 1e-6, 1.0).pow(params.gamma)
    if aug.noise_std and aug.noise_std > 0:
        noise = torch.from_numpy(rng.normal(0.0, float(aug.noise_std), size=tuple(out.shape)).astype(np.float32))
        out = torch.clamp(out + noise, 0.0, 1.0)
    return out


# --------------------------------------------------------------------------------------
# Batched augmentation on the training device
# --------------------------------------------------------------------------------------


def resolve_augment_device(cfg: Config) -> str:
    """Where augmentation runs: "cpu" (in the DataLoader workers) or "cuda" (in the train loop).

    "auto" takes cuda whenever one is visible - the same rule `select_device` applies - so the
    dataset and the trainer reach the same answer without being wired to each other.
    """
    requested = str(cfg.augment.get("device", "auto")).lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            LOG.warning("augment.device=cuda requested but no CUDA device is visible; augmenting on CPU")
            return "cpu"
        return "cuda"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def affine_theta(params: torch.Tensor) -> torch.Tensor:
    """`[N, >=4]` packed parameters -> `[N, 2, 3]` in float64, matching `SpatialParams.theta`.

    `SpatialParams.theta` does this arithmetic in float64 (numpy on Python floats) and only
    then narrows to float32. Doing the same here - which is why `augment_params` is carried
    in float64 - makes the two matrices come out bit-identical, and with them the sampled
    pixels. Computing from float32 inputs instead moved ~1 % of pixels by up to 3e-5, which
    an epoch of training amplifies into a visibly different trajectory.
    """
    wide = params.to(torch.float64)
    angle = torch.deg2rad(wide[:, 0])
    cos, sin = torch.cos(angle), torch.sin(angle)
    inv = 1.0 / wide[:, 1].clamp(min=1e-3)
    row0 = torch.stack([cos * inv, -sin * inv, wide[:, 2]], dim=1)
    row1 = torch.stack([sin * inv, cos * inv, wide[:, 3]], dim=1)
    return torch.stack([row0, row1], dim=1)


def augment_batch(
    images: torch.Tensor,
    params: torch.Tensor,
    slice_valid: torch.Tensor,
    present: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    noise_std: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Augment, encoder-normalise and mask a whole micro-batch of raw `[0, 1]` bags.

    `images` is `[B, P, S, K, H, W]`, `params` is `[B, P, 7]` in `AUGMENT_PARAM_COLUMNS` order.
    The result carries the same item contract as the CPU path: encoder-normalised, with every
    padded centre and every absent slot left at exactly zero.

    One series rides through `grid_sample` with its `[S, K]` stack folded into the channel
    axis, so it costs one sampling grid instead of `S * K` identical ones. That is bit-identical
    to sampling each slice on its own: `grid_sample` applies one grid to every channel.

    With the noise field off, the result is bit-identical to the CPU path. `params` must be
    float64 for that - see `affine_theta`.
    """
    b, p, s, k, h, w = images.shape
    params = params.to(images.device)
    flat = images.reshape(b * p, s * k, h, w)
    theta = affine_theta(params.reshape(b * p, -1)).to(images.dtype)
    grid = torch.nn.functional.affine_grid(theta, size=(b * p, s * k, h, w), align_corners=False)
    out = torch.nn.functional.grid_sample(
        flat, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    ).reshape(b, p, s, k, h, w)

    # Narrowed to the image dtype first: the CPU path multiplies by a Python float, which
    # torch narrows the same way before the multiply.
    scalars = params.to(images.dtype)
    gain = scalars[..., 4].reshape(b, p, 1, 1, 1, 1)
    bias = scalars[..., 5].reshape(b, p, 1, 1, 1, 1)
    gamma = scalars[..., 6].reshape(b, p, 1, 1, 1, 1)
    out = torch.clamp(out * gain + bias, 0.0, 1.0)
    out = torch.clamp(out, 1e-6, 1.0).pow(gamma)
    if noise_std and noise_std > 0:
        noise = torch.randn(out.shape, device=out.device, dtype=out.dtype, generator=generator)
        out = torch.clamp(out + noise * float(noise_std), 0.0, 1.0)

    out = (out - mean.to(out.device)) / std.to(out.device)
    mask = (slice_valid & present.unsqueeze(-1)).to(out.dtype).reshape(b, p, s, 1, 1, 1)
    return out * mask


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------


class EpochSampler(Sampler):
    """Yield `(index, epoch)` pairs so the epoch reaches every DataLoader worker.

    With `persistent_workers=True` each worker keeps the dataset copy it received when the
    workers started; `dataset.set_epoch()` in the main process never reaches that copy. The
    sampler, however, is iterated in the main process, so the epoch it carries alongside
    each index is always current. The wrapped sampler decides the order (and keeps the
    seeded shuffle exactly as before); this class only attaches the epoch.
    """

    def __init__(self, sampler: Sampler, epoch: int = 0) -> None:
        self.sampler = sampler
        self.epoch = int(epoch)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        epoch = self.epoch
        for index in self.sampler:
            yield (int(index), epoch)

    def __len__(self) -> int:
        return len(self.sampler)


def resolve_item_key(key: int | tuple[int, int], default_epoch: int) -> tuple[int, int]:
    """`(index, epoch)` from an EpochSampler, or a plain index that uses the dataset's own epoch."""
    if isinstance(key, tuple):
        index, epoch = key
        return int(index), int(epoch)
    return int(key), int(default_epoch)


# Selected series allowed to lack a cache entry (failed decodes, e.g. truncated DICOMs) before the
# dataset refuses to start: more than this looks like a cache built for another selection.
MAX_UNBUILT_ENTRIES = 2
MAX_UNBUILT_FRACTION = 0.01


class StudyBagDataset(Dataset):
    """Study-level bags read from the deterministic preprocessing cache.

    Every cache entry read is checked against the series selection (paths.series_selection_csv):
    the file must hold the volume the selection names for that (study, slot), and a slot the
    selection leaves empty must have no file. A stale cache fails loudly instead of feeding
    another series than the selection says.
    """

    VERIFY_SELECTION = True  # the synthetic selftest datasets have no cache and no selection

    def __init__(
        self,
        cfg: Config,
        study_ids: Sequence[str],
        label_table: LabelTable | None,
        train: bool,
        epoch: int = 0,
        strict_cache: bool = True,
        allow_unbuilt_cache: bool = False,
    ) -> None:
        """`allow_unbuilt_cache`: selected series without a cache entry only warn, however many -
        for a caller that has just built the cache and reports its failures itself (inference)."""
        self.cfg = cfg
        self.allow_unbuilt_cache = bool(allow_unbuilt_cache)
        self.study_ids = [str(s) for s in study_ids]
        self.slots = list(cfg.data.series_slots)
        self.n_centers = int(cfg.data.centers_per_series)
        self.image_size = int(cfg.data.image_size)
        self.train = bool(train)
        self.epoch = int(epoch)
        self.strict_cache = strict_cache
        self.prep_hash = preprocess_hash(cfg)
        self.augment_enabled = self.train and bool(cfg.augment.enabled)
        # Deferred: emit raw bags and let the training loop augment them on the device.
        self.defer_augment = self.augment_enabled and resolve_augment_device(cfg) == "cuda"
        # A deferred bag is only gathered slices of a float16 cache, so float16 transport is
        # lossless and halves the shared memory the DataLoader workers commit. The CPU path
        # normalises and augments here, in float32, so it must stay float32.
        self.transport_dtype = (
            torch.float16 if self.defer_augment and str(cfg.data.cache_dtype) == "float16" else torch.float32
        )
        self.label_table = label_table.subset(self.study_ids) if label_table is not None else None
        mean, std = normalization_stats(cfg)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.failures: dict[str, str] = {}
        # (study, slot) -> the selected volume_id, None for a slot the selection leaves empty.
        self.expected_volumes: dict[tuple[str, str], str | None] | None = None
        if self.VERIFY_SELECTION and bool(cfg.data.get("verify_cache_selection", True)):
            self.expected_volumes = self._expected_volumes(cfg)
        # data.laterality_canonical: mirror / reorder every study into one medial-lateral frame
        # (see laterality.py). Studies without a resolved side are used as they are.
        self.laterality: dict[str, tuple[str, float]] | None = None
        if bool(cfg.data.get("laterality_canonical", False)):
            table = load_laterality(laterality_path(cfg))
            self.laterality = {s: table.get(s, ("", float("nan"))) for s in self.study_ids}
            missing = sum(1 for s in self.study_ids if s not in table)
            unresolved = sum(1 for side, _ in self.laterality.values() if side not in ("R", "L"))
            LOG.info(
                "Laterality canonical frame: %d studies, %d right / %d left, %d unresolved (%d absent from the table)",
                len(self.study_ids),
                sum(1 for side, _ in self.laterality.values() if side == "R"),
                sum(1 for side, _ in self.laterality.values() if side == "L"),
                unresolved,
                missing,
            )
            if missing > 0.01 * max(1, len(self.study_ids)):
                raise ValueError(
                    f"{missing}/{len(self.study_ids)} studies are absent from the laterality table "
                    f"{laterality_path(cfg)}; rebuild it (`build-laterality`) for this study set."
                )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.study_ids)

    def label_weight_matrix(self) -> np.ndarray:
        """[N, 12] label weights in dataset order, without touching the image cache."""
        if self.label_table is None:
            return np.zeros((len(self.study_ids), N_TARGETS), dtype=np.float32)
        return self.label_table.weights.astype(np.float32)

    def _rng(self, index: int, slot_index: int, epoch: int) -> np.random.Generator:
        return np.random.default_rng([int(self.cfg.seed), epoch, index, slot_index])

    def _expected_volumes(self, cfg: Config) -> dict[tuple[str, str], str | None]:
        """(study, slot) -> selected volume_id (None: empty slot), after checking the table covers the run."""
        path = selection_path(cfg)
        if not path.exists():
            raise FileNotFoundError(
                f"Series selection not found: {path}. The cache is checked against it; copy the "
                "series_selection.csv that the cache was built from, point paths.series_selection_csv "
                "at it, or set data.verify_cache_selection=false to skip the check."
            )
        selection = load_selection(cfg, path)
        selection = selection[selection[STUDY_ID].isin(set(self.study_ids)) & selection["slot"].isin(self.slots)]
        selected = selection["selected"].astype(str).str.lower().isin(["true", "1"])
        expected = {
            (str(study), str(slot)): (str(volume) if is_selected else None)
            for study, slot, volume, is_selected in zip(
                selection[STUDY_ID], selection["slot"], selection["volume_id"], selected
            )
        }

        # Every (study, slot) of this run needs a row: a selection made for other slots (e.g. a
        # 3-slot table under a 4-slot config) would otherwise turn a whole slot into "missing".
        uncovered = [(study, slot) for study in self.study_ids for slot in self.slots if (study, slot) not in expected]
        if uncovered:
            by_slot = dict(Counter(slot for _, slot in uncovered))
            raise ValueError(
                f"The series selection {path} has no row for {len(uncovered)} (study, slot) pairs of this run "
                f"(per slot: {by_slot}; slots {self.slots}). It was made for other slots or studies: run "
                "select-series and build-cache with this data.series_slots."
            )

        # A selected entry without a cache file is a failed series (a few truncated DICOMs) -
        # or a cache that was never built for this selection, which must not pass as "missing".
        selected_keys = [key for key, volume in expected.items() if volume is not None]
        unbuilt = [key for key in selected_keys if not cache_path(cfg, *key).exists()]
        too_many = len(unbuilt) > max(MAX_UNBUILT_ENTRIES, MAX_UNBUILT_FRACTION * len(selected_keys))
        if too_many and not self.allow_unbuilt_cache:
            by_slot = dict(Counter(slot for _, slot in unbuilt))
            raise ValueError(
                f"{len(unbuilt)} of {len(selected_keys)} selected series have no cache entry (per slot: {by_slot}). "
                "Run build-cache for this selection."
            )
        if unbuilt:
            LOG.warning("%d selected series have no cache entry and count as missing: %s", len(unbuilt), unbuilt[:5])
        return expected

    def _check_selected_volume(self, study: str, slot: str, meta: dict) -> None:
        if self.expected_volumes is None:
            return
        expected, found = self.expected_volumes[(study, slot)], meta.get("volume_id")
        if expected is None:
            raise ValueError(
                f"Stale cache entry {study}/{slot}: the series selection leaves this slot empty, but the "
                f"cache holds {found}. Run build-cache, which removes such entries."
            )
        if found != expected:
            raise ValueError(
                f"Stale cache entry {study}/{slot}: holds volume {found}, the series selection names "
                f"{expected}. Run build-cache to rebuild it."
            )

    def _load_slot(self, study: str, slot: str) -> tuple[np.ndarray | None, dict]:
        path = cache_path(self.cfg, study, slot)
        if not path.exists():
            return None, {}
        try:
            image, meta = read_cache_entry(path, expected_hash=self.prep_hash if self.strict_cache else None)
        except Exception as exc:
            self.failures[f"{study}/{slot}"] = str(exc)
            LOG.warning("Cache read failed for %s/%s: %s", study, slot, exc)
            return None, {}
        self._check_selected_volume(study, slot, meta)
        return image, meta

    def __getitem__(self, key: int | tuple[int, int]) -> dict[str, Any]:
        index, epoch = resolve_item_key(key, self.epoch)
        study = self.study_ids[index]
        n_slots = len(self.slots)
        images = torch.zeros(
            (n_slots, self.n_centers, 3, self.image_size, self.image_size), dtype=self.transport_dtype
        )
        slice_valid = torch.zeros((n_slots, self.n_centers), dtype=torch.bool)
        present = torch.zeros((n_slots,), dtype=torch.bool)
        slot_slices = []
        substitutions = 0
        # Identity rows, so an absent slot is a no-op for `augment_batch` before it is masked.
        # float64: `affine_theta` needs the undiminished values to stay bit-exact with the CPU path.
        augment_params = (
            torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=torch.float64).repeat(n_slots, 1)
            if self.defer_augment
            else None
        )

        for p, slot in enumerate(self.slots):
            image, meta = self._load_slot(study, slot)
            if image is None or image.shape[0] == 0:
                slot_slices.append(0)
                continue
            gap_ok = np.asarray(meta.get("gap_ok", []), dtype=bool) if meta.get("gap_ok") is not None else None
            if self.laterality is not None:
                side, sagittal_normal_x = self.laterality[study]
                reverse, flip = canonical_ops(side, slot_plane(slot), sagittal_normal_x)
                image, gap_ok = apply_canonical(image, gap_ok, reverse, flip)
            rng = self._rng(index, p, epoch)
            centers, valid = bin_centers(image.shape[0], self.n_centers, rng if self.train else None)
            bag, subs = build_bag(image, centers, valid, gap_ok)
            substitutions += subs
            tensor = torch.from_numpy(bag)

            if self.defer_augment:
                # Draw in the worker so the transform stays a function of (seed, epoch,
                # index, slot); apply it on the device. Leave the bag raw and unmasked -
                # `augment_batch` normalises and masks after the transform, as the CPU
                # path does.
                assert augment_params is not None
                spatial = sample_spatial(self.cfg, rng)
                intensity = sample_intensity(self.cfg, rng)
                augment_params[p] = torch.tensor(pack_augment_params(spatial, intensity), dtype=torch.float64)
            else:
                if self.augment_enabled:
                    tensor = apply_spatial(tensor, sample_spatial(self.cfg, rng))
                    tensor = apply_intensity(tensor, self.cfg, rng)
                tensor = (tensor - self.mean) / self.std
                tensor = tensor * torch.from_numpy(valid.astype(np.float32)).view(-1, 1, 1, 1)

            images[p] = tensor
            slice_valid[p] = torch.from_numpy(valid)
            present[p] = True
            slot_slices.append(int(image.shape[0]))

        if self.label_table is not None:
            targets = torch.from_numpy(self.label_table.targets[index].astype(np.float32))
            weights = torch.from_numpy(self.label_table.weights[index].astype(np.float32))
        else:
            targets = torch.zeros(N_TARGETS, dtype=torch.float32)
            weights = torch.zeros(N_TARGETS, dtype=torch.float32)

        item = {
            "images": images,
            "slice_valid_mask": slice_valid,
            "series_present_mask": present,
            "targets": targets,
            "label_weights": weights,
            "study_id": study,
            "meta": {
                "n_present_slots": int(present.sum()),
                "slot_slices": slot_slices,
                "gap_substitutions": int(substitutions),
                "n_valid_centers": int(slice_valid.sum()),
                "laterality_side": self.laterality[study][0] if self.laterality is not None else None,
            },
        }
        if augment_params is not None:
            item["augment_params"] = augment_params
        return item


def collate_studies(batch: list[dict]) -> dict[str, Any]:
    """Stack the model inputs; keep ids and metadata out of the tensor path.

    `augment_params` is present exactly when the items are raw, un-normalised bags awaiting
    `augment_batch`; its presence is what tells the training loop which contract it holds.
    """
    out = {
        "images": torch.stack([b["images"] for b in batch]),
        "slice_valid_mask": torch.stack([b["slice_valid_mask"] for b in batch]),
        "series_present_mask": torch.stack([b["series_present_mask"] for b in batch]),
        "targets": torch.stack([b["targets"] for b in batch]),
        "label_weights": torch.stack([b["label_weights"] for b in batch]),
        "study_ids": [b["study_id"] for b in batch],
        "meta": [b["meta"] for b in batch],
    }
    if "augment_params" in batch[0]:
        out["augment_params"] = torch.stack([b["augment_params"] for b in batch])
    return out


class SyntheticBagDataset(Dataset):
    """Deterministic synthetic studies for the smoke test.

    Each target gets a bright patch in one slot whenever it is positive, so a working
    training path must be able to drive the loss down. No real image is required.
    """

    def __init__(
        self,
        cfg: Config,
        n_studies: int = 16,
        missing_slot_probability: float = 0.2,
        short_stack_probability: float = 0.2,
        unknown_probability: float = 0.2,
        seed: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.slots = list(cfg.data.series_slots)
        self.n_centers = int(cfg.data.centers_per_series)
        self.image_size = int(cfg.data.image_size)
        self.n_studies = int(n_studies)
        self.missing_slot_probability = missing_slot_probability
        self.short_stack_probability = short_stack_probability
        self.unknown_probability = unknown_probability
        self.seed = int(cfg.seed if seed is None else seed)
        mean, std = normalization_stats(cfg)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

    def __len__(self) -> int:
        return self.n_studies

    def set_epoch(self, epoch: int) -> None:  # API parity with StudyBagDataset
        return None

    def label_weight_matrix(self) -> np.ndarray:
        """[N, 12] label weights, drawn exactly as __getitem__ draws them (no images built)."""
        rows = []
        for index in range(self.n_studies):
            rng = np.random.default_rng([self.seed, index])
            rng.random(N_TARGETS)  # labels
            rows.append((rng.random(N_TARGETS) >= self.unknown_probability).astype(np.float32))
        return np.stack(rows)

    def __getitem__(self, key: int | tuple[int, int]) -> dict[str, Any]:
        index, _ = resolve_item_key(key, 0)
        rng = np.random.default_rng([self.seed, index])
        n_slots = len(self.slots)
        size = self.image_size
        images = torch.zeros((n_slots, self.n_centers, 3, size, size), dtype=torch.float32)
        slice_valid = torch.zeros((n_slots, self.n_centers), dtype=torch.bool)
        present = torch.zeros((n_slots,), dtype=torch.bool)

        labels = (rng.random(N_TARGETS) < 0.35).astype(np.float32)
        weights = (rng.random(N_TARGETS) >= self.unknown_probability).astype(np.float32)

        for p in range(n_slots):
            if rng.random() < self.missing_slot_probability:
                continue
            n_valid = self.n_centers
            if rng.random() < self.short_stack_probability:
                n_valid = int(rng.integers(1, self.n_centers))
            base = rng.normal(0.35, 0.05, size=(n_valid, 3, size, size)).astype(np.float32)
            for c in range(N_TARGETS):
                if labels[c] <= 0 or c % n_slots != p:
                    continue
                row = (c * 17) % max(size - 20, 1)
                col = (c * 29) % max(size - 20, 1)
                base[:, :, row : row + 16, col : col + 16] += 0.6
            tensor = torch.from_numpy(np.clip(base, 0.0, 1.0))
            images[p, :n_valid] = (tensor - self.mean) / self.std
            slice_valid[p, :n_valid] = True
            present[p] = True

        if not present.any():  # never emit an all-missing study as if it were normal
            present[0] = True
            slice_valid[0, :] = True
            images[0] = (torch.full((self.n_centers, 3, size, size), 0.3) - self.mean) / self.std

        return {
            "images": images,
            "slice_valid_mask": slice_valid,
            "series_present_mask": present,
            "targets": torch.from_numpy(labels),
            "label_weights": torch.from_numpy(weights),
            "study_id": f"synthetic-{index:04d}",
            "meta": {"n_present_slots": int(present.sum()), "synthetic": True},
        }


def check_study_coverage(cfg: Config, study_ids: Sequence[str]) -> dict:
    """Which studies have no cached slot at all - the data-quality gate."""
    slots = list(cfg.data.series_slots)
    missing_all, partial = [], []
    for study in study_ids:
        present = [slot for slot in slots if cache_path(cfg, str(study), slot).exists()]
        if not present:
            missing_all.append(str(study))
        elif len(present) < len(slots):
            partial.append(str(study))
    return {
        "n_studies": len(study_ids),
        "n_all_missing": len(missing_all),
        "n_partial": len(partial),
        "all_missing_studies": missing_all,
        "partial_studies": partial,
    }


def enforce_coverage_gate(cfg: Config, coverage: dict, split_name: str) -> list[str]:
    """Fail the run - or apply an explicit, separately reported fallback - on all-missing studies."""
    if not coverage["n_all_missing"]:
        return []
    message = (
        f"{coverage['n_all_missing']} of {coverage['n_studies']} {split_name} studies have no usable series "
        "in any slot."
    )
    if not bool(cfg.data.allow_all_missing_studies):
        raise RuntimeError(
            message
            + " The data-quality gate fails by default: fix series selection/caching, or set "
            "data.allow_all_missing_studies=true to exclude them explicitly and report them separately. "
            "Silently dropping them from validation would flatter the metrics."
        )
    LOG.warning("%s Excluding them explicitly (data.allow_all_missing_studies=true).", message)
    return list(coverage["all_missing_studies"])
