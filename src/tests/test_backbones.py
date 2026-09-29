"""Encoder adapters (DINOv2 ViT-S/14, RadImageNet ResNet-50, EfficientNet) with RANDOM weights.

    cd src && python -m pytest tests/test_backbones.py -q

No network and no pretrained file is needed here; tests/test_pretrained_encoders.py checks
the real weights and skips when they are not available.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from knee_mri import selftest  # noqa: E402
from knee_mri.config import Config, load_config  # noqa: E402
from knee_mri.constants import CHECKPOINT_VERSION, TARGETS  # noqa: E402

# The last commit before encoders.py: its model.py wrote every EfficientNet checkpoint so far.
LEGACY_COMMIT = "a9e5155"


@pytest.fixture(scope="module")
def cfg():
    return selftest.tiny_config(load_config())


def test_module_import_loads_no_timm():
    code = "import sys; sys.path.insert(0, r'%s'); import knee_mri.model, knee_mri.train, knee_mri.evaluate; print('timm' in sys.modules)" % SRC
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False", "importing the package must not import (or download) timm"


def test_legacy_efficientnet_state_keys(cfg):
    selftest.check_legacy_efficientnet_state_keys(cfg)


def test_encoder_feature_maps(cfg):
    selftest.check_encoder_feature_maps(cfg)


def test_new_backbone_masking(cfg):
    selftest.check_new_backbone_masking(cfg)


def test_new_backbone_pooling_options(cfg):
    selftest.check_new_backbone_pooling_options(cfg)


def test_dino_patch_grid(cfg):
    selftest.check_dino_patch_grid(cfg)


def test_freeze_policies(cfg):
    selftest.check_freeze_policies(cfg)


def test_parameter_groups_complete(cfg):
    selftest.check_parameter_groups_complete(cfg)


def test_normalization_profiles(cfg):
    detail = selftest.check_normalization_profiles(cfg)
    if not torch.cuda.is_available():
        pytest.skip(f"worker path verified, but NO GPU: the CPU-vs-GPU comparison did not run ({detail})")


def test_backbone_resume_signature(cfg):
    selftest.check_backbone_resume_signature(cfg)


def test_new_backbone_grad_checkpointing(cfg):
    selftest.check_new_backbone_grad_checkpointing(cfg)


def test_radimagenet_loader(cfg, tmp_path):
    selftest.check_radimagenet_loader(cfg, tmp_path)


def test_new_backbone_training_roundtrip(cfg):
    selftest.check_new_backbone_training_roundtrip(cfg)


def test_example_configs_validate():
    for name, backbone in (("config.dinov2_vits14.yaml", "dinov2_vits14"), ("config.radimagenet_resnet50.yaml", "radimagenet_resnet50")):
        example = load_config(SRC / name)
        assert example.model.backbone == backbone and example.model.encoder_trainable == "last_n"
    with pytest.raises(ValueError, match="multiple of the DINOv2 patch size"):
        load_config(SRC / "config.dinov2_vits14.yaml", ["data.image_size=320"])


def test_image_size_rule_is_per_backbone():
    """DINOv2 needs multiples of 14 only (not of 56); the CNNs keep the multiple-of-8 rule."""
    for size in (42, 224, 336, 378):  # 42 and 378 are not multiples of 8
        assert int(load_config(SRC / "config.dinov2_vits14.yaml", [f"data.image_size={size}"]).data.image_size) == size
    for config in ("config.yaml", "config.radimagenet_resnet50.yaml"):
        with pytest.raises(ValueError, match="multiple of 8"):
            load_config(SRC / config, ["data.image_size=378"])
        load_config(SRC / config, ["data.image_size=320"])


@pytest.mark.parametrize("config_name", ["config.dinov2_vits14.yaml", "config.radimagenet_resnet50.yaml"])
def test_older_checks_run_under_new_backbone_configs(config_name, tmp_path):
    """The selftest sizes follow the configured backbone (DINOv2 refuses 32/64 px); the checks
    about EfficientNet itself get an explicit EfficientNet config."""
    small = selftest.tiny_config(load_config(SRC / config_name))
    backbone = small.model.backbone
    assert small.data.image_size == selftest.TINY_SIZES[backbone]
    assert selftest.wide_config(small).data.image_size == selftest.WIDE_SIZES[backbone]
    effnet = selftest.efficientnet_config(small)
    assert effnet.model.backbone == "efficientnet_b0" and effnet.data.image_size == 32
    assert effnet.data.encoder_normalization in ("mri_scalar", "imagenet")
    selftest.check_side_pooling_features(small)
    selftest.check_target_attention_starts_as_head(small)
    selftest.check_inference_architecture(small, tmp_path)
    selftest.check_spatial_pool_training_step(small)
    selftest.check_augment_device_equivalence(small)
    selftest.check_backbone_resume_signature(small)
    selftest.check_padding_invariance(effnet)


def test_wrong_backbone_weights_refused():
    with pytest.raises(ValueError, match="not a pretrained source"):
        load_config(SRC / "config.yaml", ["model.backbone=radimagenet_resnet50"])


# --------------------------------------------------------------------------------------
# Reproducible compatibility with the pre-adapter code (git history)
# --------------------------------------------------------------------------------------


def _legacy_model_module(tmp_path: Path):
    """model.py as of LEGACY_COMMIT, imported as a submodule of today's package."""
    try:
        source = subprocess.run(
            ["git", "show", f"{LEGACY_COMMIT}:src/knee_mri/model.py"],
            cwd=SRC, capture_output=True, text=True, check=True, encoding="utf-8",
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git history with commit {LEGACY_COMMIT} is not available: {exc}")
    path = tmp_path / "legacy_model.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("knee_mri._legacy_model_a9e5155", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "backbone,options",
    [
        ("efficientnet_b0", {}),
        ("efficientnet_b0", {"spatial_pool": "attention"}),
        ("efficientnet_b0", {"target_attention": True, "side_pooling": True}),
        ("efficientnet_v2_s", {}),
    ],
)
def test_checkpoint_from_pre_adapter_code(cfg, tmp_path, backbone, options):
    """A checkpoint written by the pre-adapter model.py loads strictly and predicts identically."""
    from knee_mri.evaluate import load_checkpoint_for_inference

    legacy = _legacy_model_module(tmp_path)
    trained = cfg.copy()
    trained.model.backbone = backbone
    trained.model.weights = "none"
    for key, value in options.items():
        trained.model[key] = value
    for key in ("encoder_trainable", "encoder_trainable_units"):  # keys the old code never wrote
        trained.model.pop(key, None)
    torch.manual_seed(3)
    old = legacy.build_model(Config(trained.to_dict())).eval()
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "target_order": list(TARGETS),
            "config": trained.to_dict(),
            "model_description": old.describe(),
            "model": old.state_dict(),
        },
        path,
    )
    batch = selftest._random_batch(cfg, b=2, seed=8)
    batch["slice_valid_mask"][0, 1, 1:] = False
    with torch.no_grad():
        expected = old(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])

    caller = cfg.copy()  # today's config, with the new keys at their defaults
    loaded, _ = load_checkpoint_for_inference(caller, path)
    with torch.no_grad():
        actual = loaded(batch["images"], batch["slice_valid_mask"], batch["series_present_mask"])
    assert torch.equal(expected, actual)
    assert list(loaded.state_dict()) == list(old.state_dict())
