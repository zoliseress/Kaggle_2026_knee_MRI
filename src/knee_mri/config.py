"""YAML configuration with dotted CLI overrides.

The config object is a plain nested dict wrapper: readable, picklable and easy to
serialise into a checkpoint. Paths are resolved once, relative to the repository
root, so notebooks and CLI runs agree.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from .constants import PLANES, TARGETS

SPATIAL_POOLS = ("avg", "avgmax", "attention")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "src" / "config.yaml"


class Config(dict):
    """Nested dict with attribute access and dotted get/set."""

    def __init__(self, data: dict | None = None) -> None:
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = Config(value) if isinstance(value, dict) else value

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - attribute protocol
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = Config(value) if isinstance(value, dict) else value

    def get_dotted(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_dotted(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: Any = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = Config()
            node = node[part]
        node[parts[-1]] = Config(value) if isinstance(value, dict) else value

    def to_dict(self) -> dict:
        out: dict = {}
        for key, value in self.items():
            out[key] = value.to_dict() if isinstance(value, Config) else value
        return out

    def copy(self) -> "Config":  # type: ignore[override]
        return Config(copy.deepcopy(self.to_dict()))

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)


def _parse_scalar(text: str) -> Any:
    """Parse a CLI override value with YAML semantics (ints, floats, bools, lists)."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _resolve(path_value: Any, default: Path | None = None) -> Path | None:
    if path_value in (None, "", "null"):
        return default
    path = Path(str(path_value))
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def resolve_paths(cfg: Config) -> Config:
    """Fill in every path default and turn each entry into an absolute string."""
    paths = cfg.setdefault("paths", Config())
    data_root = _resolve(paths.get("data_root"), REPO_ROOT / "data")
    work_dir = _resolve(paths.get("work_dir"), REPO_ROOT / "work")
    assert data_root is not None and work_dir is not None

    defaults = {
        "dicom_root": data_root / "train_series",
        "train_csv": data_root / "train.csv",
        "train_series_csv": data_root / "train_series.csv",
        "cache_dir": work_dir / "cache",
        "output_dir": work_dir / "runs",
    }
    optional = [
        "labels_details_csv",
        "labels_statuses_csv",
        "labels_predictions_csv",
        "labels_predictions_exclude_borderline_csv",
        "reference_csv",
        "frozen_reference_csv",
    ]

    paths["data_root"] = str(data_root)
    paths["work_dir"] = str(work_dir)
    for key, default in defaults.items():
        paths[key] = str(_resolve(paths.get(key), default))
    for key in optional:
        resolved = _resolve(paths.get(key), None)
        paths[key] = str(resolved) if resolved is not None else None
    return cfg


def validate_config(cfg: Config) -> None:
    """Fail fast on settings the architecture or the data contract cannot honour."""
    data = cfg.data
    if int(data.image_size) < 32 or int(data.image_size) % 8 != 0:
        raise ValueError(f"data.image_size must be a multiple of 8 and >= 32, got {data.image_size}")
    if int(data.adjacent_slices) != 3:
        raise ValueError(
            "data.adjacent_slices must be 3: the torchvision EfficientNet-B0 stem expects "
            f"3 input channels, got {data.adjacent_slices}"
        )
    if int(data.centers_per_series) < 1:
        raise ValueError("data.centers_per_series must be >= 1")
    slots = list(data.series_slots)
    if not slots:
        raise ValueError("data.series_slots must not be empty")
    unknown = [s for s in slots if s not in PLANES]
    if unknown:
        raise ValueError(f"data.series_slots contains unknown planes {unknown}; known: {PLANES}")
    if len(set(slots)) != len(slots):
        raise ValueError(f"data.series_slots must be unique, got {slots}")
    if float(data.fov_mm) <= 0:
        raise ValueError("data.fov_mm must be positive")
    if data.crop_center not in ("foreground", "foreground_extent", "geometric"):
        raise ValueError("data.crop_center must be 'foreground', 'foreground_extent' or 'geometric'")
    if data.encoder_normalization not in ("imagenet", "mri_scalar"):
        raise ValueError("data.encoder_normalization must be 'imagenet' or 'mri_scalar'")

    if int(cfg.train.microbatch_studies) < 1 or int(cfg.train.accumulation_steps) < 1:
        raise ValueError("train.microbatch_studies and train.accumulation_steps must be >= 1")
    if str(cfg.train.get("loss_normalization", "microbatch")) not in ("microbatch", "window", "global"):
        raise ValueError("train.loss_normalization must be 'microbatch', 'window' or 'global'")
    for key in ("eval_num_workers", "eval_prefetch_factor"):
        value = cfg.train.get(key)
        if value is not None and int(value) < (0 if key == "eval_num_workers" else 1):
            raise ValueError(f"train.{key} must be null or a non-negative integer")
    if cfg.train.amp not in ("auto", "bf16", "fp16", "fp32"):
        raise ValueError("train.amp must be one of auto|bf16|fp16|fp32")
    if str(cfg.augment.get("device", "auto")).lower() not in ("auto", "cpu", "cuda"):
        raise ValueError(f"augment.device must be one of auto|cpu|cuda, got {cfg.augment.get('device')!r}")
    if int(cfg.train.warmup_epochs) >= int(cfg.train.max_epochs):
        raise ValueError("train.warmup_epochs must be smaller than train.max_epochs")

    if str(cfg.model.get("spatial_pool", "avg")) not in SPATIAL_POOLS:
        raise ValueError(f"model.spatial_pool must be one of {sorted(SPATIAL_POOLS)}")

    if cfg.labels.borderline_policy not in ("exclude", "as_negative"):
        raise ValueError("labels.borderline_policy must be 'exclude' or 'as_negative'")
    if not 0.0 <= float(cfg.labels.unmentioned_weight) <= 1.0:
        raise ValueError("labels.unmentioned_weight must be in [0, 1]")
    per_target = cfg.labels.get("unmentioned_weight_per_target") or {}
    if not isinstance(per_target, dict):
        raise ValueError("labels.unmentioned_weight_per_target must be a mapping target -> weight")
    unknown_targets = [t for t in per_target if t not in TARGETS]
    if unknown_targets:
        raise ValueError(f"labels.unmentioned_weight_per_target names unknown targets {unknown_targets}")
    for target, weight in per_target.items():
        if not 0.0 <= float(weight) <= 1.0:
            raise ValueError(f"labels.unmentioned_weight_per_target[{target!r}] must be in [0, 1], got {weight}")
    if cfg.labels.source not in ("auto", "details", "wide", "train_csv"):
        raise ValueError("labels.source must be one of auto|details|wide|train_csv")

    if int(cfg.split.n_folds) < 2:
        raise ValueError("split.n_folds must be >= 2")
    if not 0 <= int(cfg.split.fold) < int(cfg.split.n_folds):
        raise ValueError("split.fold must be in [0, n_folds)")
    if cfg.split.grouping not in ("auto", "patient", "study"):
        raise ValueError("split.grouping must be one of auto|patient|study")


def load_config(
    path: str | Path | None = None,
    overrides: Iterable[str] | None = None,
    resolve: bool = True,
) -> Config:
    """Load the YAML config and apply dotted key=value overrides."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = Config(yaml.safe_load(handle) or {})

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.subkey=value, got: {item!r}")
        key, raw = item.split("=", 1)
        cfg.set_dotted(key.strip(), _parse_scalar(raw.strip()))

    cfg["_config_path"] = str(config_path)
    if resolve:
        resolve_paths(cfg)
    validate_config(cfg)
    return cfg


def save_config(cfg: Config, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, sort_keys=False, allow_unicode=True)
    return path
