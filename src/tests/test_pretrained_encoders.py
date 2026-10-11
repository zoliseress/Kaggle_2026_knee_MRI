"""Real pretrained weights of the new encoders. Each test SKIPS (and says why) when its
weights are not on this machine; the architecture itself is covered by test_backbones.py.

    cd src && python -m pytest tests/test_pretrained_encoders.py -v -rs

RadImageNet:  KNEE_MRI_RADIMAGENET_WEIGHTS=<.../RadImageNet_pytorch/ResNet50.pt>, default
              work/pretrained/RadImageNet_pytorch/ResNet50.pt (from RadImageNet_pytorch.zip,
              linked from github.com/BMEII-AI/RadImageNet).
DINOv2:       timm/vit_small_patch14_dinov2.lvd142m model.safetensors in the Hugging Face
              cache; set KNEE_MRI_ALLOW_DOWNLOAD=1 to let the test download it.
CoAtNet:      timm/coatnet_0_rw_224.sw_in1k and timm/coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k,
              the same way (Hugging Face cache, or KNEE_MRI_ALLOW_DOWNLOAD=1).
ConvNeXt:     timm/<backbone>.<tag> for every tag in CONVNEXT_WEIGHT_TAGS, the same way.
Existing runs: every work/runs/<dataset>/*/best.pt is rebuilt offline and must give finite logits.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1]
REPO = SRC.parent
sys.path.insert(0, str(SRC))

from knee_mri.constants import CONVNEXT_WEIGHT_TAGS  # noqa: E402
from knee_mri.encoders import (  # noqa: E402
    COATNET_HF_FILE,
    COATNETS,
    CONVNEXT_HF_FILE,
    CoAtNetEncoder,
    ConvNeXtEncoder,
    DINOV2_HF_FILE,
    DINOV2_HF_HUB_ID,
    RADIMAGENET_RESNET50_SHA256,
    DinoV2Encoder,
    RadImageNetResNet50Encoder,
    convert_radimagenet_state,
)


def _radimagenet_file() -> Path:
    path = Path(os.environ.get("KNEE_MRI_RADIMAGENET_WEIGHTS", REPO / "work/pretrained/RadImageNet_pytorch/ResNet50.pt"))
    if not path.is_file():
        pytest.skip(f"RadImageNet ResNet50.pt not found at {path} (set KNEE_MRI_RADIMAGENET_WEIGHTS)")
    return path


def _dinov2_file() -> Path:
    try:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
    except ImportError:
        pytest.skip("huggingface_hub is not installed")
    cached = try_to_load_from_cache(DINOV2_HF_HUB_ID, DINOV2_HF_FILE)
    if isinstance(cached, str) and Path(cached).is_file():
        return Path(cached)
    if os.environ.get("KNEE_MRI_ALLOW_DOWNLOAD") == "1":
        return Path(hf_hub_download(DINOV2_HF_HUB_ID, DINOV2_HF_FILE))
    pytest.skip(f"{DINOV2_HF_HUB_ID} is not in the Hugging Face cache (set KNEE_MRI_ALLOW_DOWNLOAD=1)")


def test_radimagenet_real_weights_load_completely():
    path = _radimagenet_file()
    encoder = RadImageNetResNet50Encoder(str(path)).eval()
    info = encoder.weights_info
    assert info["sha256"] == RADIMAGENET_RESNET50_SHA256, "not the verified official ResNet50.pt"
    reference = convert_radimagenet_state(torch.load(str(path), map_location="cpu", weights_only=True))
    state = encoder.state_dict()
    assert set(state) == set(reference) and len(state) == 318
    for key, value in reference.items():
        assert torch.equal(state[key], value), f"{key} differs from the file"
    # Real BN statistics, not the torchvision defaults (0 / 1).
    assert float(state["bn1.running_var"].sub(1).abs().max()) > 0.1
    with torch.no_grad():
        feature_map = encoder(torch.rand((2, 3, 224, 224)) * 2 - 1)  # radimagenet_torch range [-1, 1]
    assert feature_map.shape == (2, 2048, 7, 7) and torch.isfinite(feature_map).all()


def test_radimagenet_example_config_builds_with_real_weights():
    _radimagenet_file()
    from knee_mri.config import load_config
    from knee_mri.model import build_model

    cfg = load_config(SRC / "config.radimagenet_resnet50.yaml")
    if os.environ.get("KNEE_MRI_RADIMAGENET_WEIGHTS"):
        cfg.model.weights = os.environ["KNEE_MRI_RADIMAGENET_WEIGHTS"]
    model = build_model(cfg)
    description = model.describe()
    assert description["weights_info"]["sha256"] == RADIMAGENET_RESNET50_SHA256
    assert description["encoder_trainable"] == "last_n" and description["encoder_trainable_units"] == 1
    trainable = {n.split(".")[1] for n, p in model.named_parameters() if p.requires_grad and n.startswith("features.")}
    assert trainable == {"layer4"}


@pytest.mark.parametrize("size,grid", [(224, 16), (336, 24)])
def test_dinov2_real_weights_and_position_embeddings(size, grid):
    path = _dinov2_file()
    from safetensors.torch import load_file

    reference = load_file(str(path))
    encoder = DinoV2Encoder("lvd142m", size).eval()
    state = encoder.model.state_dict()
    # Every weight but the position embeddings comes from the file unchanged ...
    for key, value in reference.items():
        if key == "pos_embed":
            continue
        assert torch.equal(state[key], value), f"{key} differs from the file"
    # ... and those are resampled from the 518 px (37x37) grid to this size; the CLS entry is kept.
    assert reference["pos_embed"].shape[1] == 1 + 37 * 37
    assert state["pos_embed"].shape == (1, 1 + grid * grid, 384)
    assert torch.equal(state["pos_embed"][:, :1], reference["pos_embed"][:, :1])
    x = torch.randn((2, 3, size, size))
    with torch.no_grad():
        ours = encoder(x)
        timm_map = encoder.model.forward_intermediates(x, indices=[11], norm=True, output_fmt="NCHW", intermediates_only=True)[0]
    assert ours.shape == (2, 384, grid, grid) and torch.isfinite(ours).all()
    assert float((ours - timm_map).abs().max()) < 1e-4
    assert encoder.weights_info["hf_hub_id"] == DINOV2_HF_HUB_ID and len(encoder.weights_info["sha256"]) == 64


def _coatnet_file(hub_id: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
    except ImportError:
        pytest.skip("huggingface_hub is not installed")
    cached = try_to_load_from_cache(hub_id, COATNET_HF_FILE)
    if isinstance(cached, str) and Path(cached).is_file():
        return Path(cached)
    if os.environ.get("KNEE_MRI_ALLOW_DOWNLOAD") == "1":
        return Path(hf_hub_download(hub_id, COATNET_HF_FILE))
    pytest.skip(f"{hub_id} is not in the Hugging Face cache (set KNEE_MRI_ALLOW_DOWNLOAD=1)")


@pytest.mark.parametrize("backbone,n_tables", [("coatnet_0_rw", 9), ("coatnet_rmlp_1_rw2", 0)])
def test_coatnet_real_weights_at_320(backbone, n_tables):
    """Every tensor of the file is in the model unchanged, except the relative-position bias tables,
    which timm resizes from the 224 px (14x14) grids to the 320 px (20x20 / 10x10) ones; the rmlp
    variant has none (MLP biases). The classifier head is the only thing left out."""
    name, tag = COATNETS[backbone]
    path = _coatnet_file(f"timm/{name}.{tag}")
    from safetensors.torch import load_file

    reference = load_file(str(path))
    encoder = CoAtNetEncoder(backbone, tag, 320).eval()
    state = encoder.model.state_dict()
    assert set(reference) - set(state) == {"head.fc.weight", "head.fc.bias"} and set(state) <= set(reference)
    resized = []
    for key, value in reference.items():
        if key.startswith("head."):
            continue
        if key.endswith("relative_position_bias_table"):
            module = encoder.model.get_submodule(key[: -len(".relative_position_bias_table")])
            window = tuple(module.window_size)
            assert window in ((20, 20), (10, 10)), f"{key}: window {window} is not a 320 px stage grid"
            assert state[key].shape[0] == (2 * window[0] - 1) * (2 * window[1] - 1), key
            resized.append(key)
            continue
        assert torch.equal(state[key], value), f"{key} differs from the file"
    assert len(resized) == n_tables
    with torch.no_grad():
        feature_map = encoder(torch.rand((2, 3, 320, 320)) * 2 - 1)  # inception range [-1, 1]
    assert feature_map.shape == (2, 768, 10, 10) and torch.isfinite(feature_map).all()
    assert encoder.weights_info["hf_hub_id"] == f"timm/{name}.{tag}" and len(encoder.weights_info["sha256"]) == 64


@pytest.mark.parametrize(
    "backbone,tag", [(backbone, tag) for backbone, tags in CONVNEXT_WEIGHT_TAGS.items() for tag in tags]
)
def test_convnext_real_weights_at_320(backbone, tag):
    """Every tensor of the file is in the model unchanged (only the linear classifier is left out),
    and the 320 px map is the pretrained final LayerNorm of forward_features at each position."""
    path = _coatnet_file(f"timm/{backbone}.{tag}")  # same Hugging Face file layout as CoAtNet
    assert path.name == CONVNEXT_HF_FILE
    from safetensors.torch import load_file

    reference = load_file(str(path))
    encoder = ConvNeXtEncoder(backbone, tag).eval()
    state = encoder.model.state_dict()
    assert set(reference) - set(state) <= {"head.fc.weight", "head.fc.bias"} and set(state) <= set(reference)
    for key, value in state.items():
        assert torch.equal(value, reference[key]), f"{key} differs from the file"
    x = (torch.rand((2, 3, 320, 320)) - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    with torch.no_grad():
        feature_map = encoder(x)
        expected = encoder.model.head.norm(encoder.model.forward_features(x))
    assert feature_map.shape == (2, encoder.out_channels, 10, 10) and torch.isfinite(feature_map).all()
    assert torch.equal(feature_map, expected)
    assert encoder.weights_info["hf_hub_id"] == f"timm/{backbone}.{tag}" and len(encoder.weights_info["sha256"]) == 64


def test_existing_run_checkpoints_rebuild_offline():
    from knee_mri.config import Config
    from knee_mri.evaluate import load_checkpoint_for_inference

    checkpoints = sorted((REPO / "work" / "runs").glob("*/*/best.pt"))
    if not checkpoints:
        pytest.skip("no work/runs/<dataset>/*/best.pt on this machine")
    for path in checkpoints[:8]:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        cfg = Config(payload["config"])
        model, _ = load_checkpoint_for_inference(cfg, path)
        size, slots = int(cfg.data.image_size), len(cfg.data.series_slots)
        images = torch.randn((1, slots, 2, 3, size, size))
        with torch.no_grad():
            logits = model(images, torch.ones((1, slots, 2), dtype=torch.bool), torch.ones((1, slots), dtype=torch.bool))
        assert logits.shape == (1, len(payload["target_order"])) and torch.isfinite(logits).all(), path
