"""Encoder adapters: one slice triplet `[N, 3, H, W]` -> one spatial feature map `[N, C, Hf, Wf]`.

The MIL model (model.py) only needs a map it can average, split into column halves and
attend over. Every backbone is wrapped so that it delivers exactly that and nothing else:

    backbone              model id                              map at 224 px     C
    efficientnet_b0       torchvision efficientnet_b0            [N, 1280,  7,  7]  1280
    efficientnet_v2_s     torchvision efficientnet_v2_s          [N, 1280,  7,  7]  1280
    radimagenet_resnet50  torchvision resnet50, RadImageNet      [N, 2048,  7,  7]  2048
    dinov2_vits14         timm vit_small_patch14_dinov2.lvd142m  [N,  384, 16, 16]   384

The head is sized from `out_channels`; nothing is projected to a common width.

Interface (every adapter):
    forward(x)                   [N, 3, H, W] -> [N, C, Hf, Wf]
    out_channels                 C
    feature_map_size(size)       (Hf, Wf) for a square input of `size` px
    units()                      ordered trainable units for the `last_n` freeze policy
    apply_freeze(mode, n)        sets requires_grad; `train()` keeps frozen units in eval mode
    grad_checkpointing           backbone-specific activation recomputation (training only)
    describe() / weights_info    model id, adapter version, weight provenance

Pretrained sources are explicit (`model.weights`); a missing or wrong file raises. Nothing
is downloaded or even imported (timm) at module import time - only when an encoder is built.
`weights="none"` builds the bare architecture: that is how inference rebuilds a trained
model before loading the full project checkpoint, with no network and no pretrained file.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

from .constants import DEFAULT_BACKBONE, DINOV2_PATCH_SIZE, ENCODER_FEATURES, ENCODER_UNITS
from .utils import LOG

# Bump when an adapter's output semantics change (token handling, which layer is returned,
# key layout). Stored in every checkpoint's model description and compared at inference.
ADAPTER_VERSIONS = {
    "efficientnet_b0": "efficientnet_features_v1",
    "efficientnet_v2_s": "efficientnet_features_v1",
    "radimagenet_resnet50": "resnet50_layer4_v1",
    "dinov2_vits14": "dinov2_patch_tokens_v1",
}
FREEZE_MODES = ("all", "frozen", "last_n")
RANDOM_WEIGHTS = ("none", "random", "null")

DINOV2_TIMM_NAME = "vit_small_patch14_dinov2"
DINOV2_TIMM_TAG = "lvd142m"
DINOV2_HF_HUB_ID = f"timm/{DINOV2_TIMM_NAME}.{DINOV2_TIMM_TAG}"
DINOV2_HF_FILE = "model.safetensors"

# The official RadImageNet PyTorch release (README of github.com/BMEII-AI/RadImageNet,
# Google Drive file RadImageNet_pytorch.zip -> RadImageNet_pytorch/ResNet50.pt). It is the
# state_dict of the demo's `Backbone` wrapper,
#     self.backbone = nn.Sequential(*list(torchvision.models.resnet50().children())[:9])
# so its keys are backbone.<i>.<rest> with i the index among resnet50's children.
RADIMAGENET_KEY_PREFIXES = {
    "backbone.0.": "conv1.",
    "backbone.1.": "bn1.",
    "backbone.4.": "layer1.",
    "backbone.5.": "layer2.",
    "backbone.6.": "layer3.",
    "backbone.7.": "layer4.",
}
# sha256 of RadImageNet_pytorch/ResNet50.pt as downloaded and verified on 2026-09-28.
RADIMAGENET_RESNET50_SHA256 = "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734"


def _is_random(spec: str) -> bool:
    return str(spec).lower() in RANDOM_WEIGHTS


def resolve_weight_path(spec: str) -> Path | None:
    """An existing file named by `spec`: as given (cwd), else relative to the repository root."""
    from .config import REPO_ROOT

    path = Path(str(spec)).expanduser()
    if path.is_file():
        return path
    if not path.is_absolute() and (REPO_ROOT / path).is_file():
        return REPO_ROOT / path
    return None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _random_warning(backbone: str, spec: str) -> None:
    LOG.warning(
        "model.weights=%s: the %s encoder starts from RANDOM initialisation. This is an intentional "
        "ablation (or an architecture rebuild before a checkpoint load), not a fallback.",
        spec,
        backbone,
    )


# --------------------------------------------------------------------------------------
# Common behaviour
# --------------------------------------------------------------------------------------


class EncoderMixin:
    """Freeze policy, train-mode policy and description shared by all adapters.

    Subclasses set `backbone`, `model_id`, `out_channels`, `weights_info`, implement `units()`
    (the `last_n` units, input to output) and may extend `last_n_trainable_modules(n)` with
    a tail that trains together with the last units (DINOv2's final norm).
    """

    backbone: str
    model_id: str
    out_channels: int
    weights_info: dict
    input_size: int | None = None
    patch_size: int | None = None

    def _init_policy(self) -> None:
        self.freeze_mode = "all"
        self.trainable_units = 0
        self.grad_checkpointing = False

    # -- freezing ----------------------------------------------------------------------

    def units(self) -> list[nn.Module]:  # pragma: no cover - abstract
        raise NotImplementedError

    def last_n_trainable_modules(self, n: int) -> list[nn.Module]:
        """The modules trained by `last_n` with `n` units; everything else is frozen."""
        return self.units()[-n:]

    def apply_freeze(self, mode: str = "all", n: int = 0) -> None:
        """Set requires_grad for the whole encoder. `all` trains everything, `frozen` nothing,
        `last_n` the last `n` units (plus the adapter's tail, e.g. DINOv2's final norm)."""
        if mode not in FREEZE_MODES:
            raise ValueError(f"model.encoder_trainable must be one of {FREEZE_MODES}, got {mode!r}")
        n_units = len(self.units())
        if mode == "last_n" and not 1 <= int(n) <= n_units:
            raise ValueError(
                f"model.encoder_trainable_units must be in [1, {n_units}] for {self.backbone} "
                f"(its units: {self.unit_names()}), got {n}"
            )
        self.freeze_mode = mode
        self.trainable_units = int(n) if mode == "last_n" else (n_units if mode == "all" else 0)
        train_all = mode == "all"
        for parameter in self.parameters():
            parameter.requires_grad_(train_all)
        if mode == "last_n":
            for module in self.last_n_trainable_modules(int(n)):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

    def unit_names(self) -> list[str]:
        names = {id(m): name for name, m in self.named_modules()}
        return [names.get(id(u), "?") for u in self.units()]

    def frozen_modules(self) -> list[nn.Module]:
        """Maximal submodules without a single trainable parameter (kept in eval mode)."""
        out: list[nn.Module] = []

        def visit(module: nn.Module) -> None:
            params = list(module.parameters())
            if params and not any(p.requires_grad for p in params):
                out.append(module)
                return
            for child in module.children():
                visit(child)

        for child in self.children():
            visit(child)
        return out

    def apply_train_policy(self, freeze_bn_running_stats: bool) -> None:
        """Called after `train(True)`.

        Frozen parts run in eval mode: their BatchNorm uses (and never updates) the
        pretrained running statistics, and dropout / stochastic depth is off there.
        Trainable BatchNorm layers keep their running statistics frozen too when
        `freeze_bn_running_stats` (their affine parameters still train).
        """
        for module in self.frozen_modules():
            module.eval()
        if freeze_bn_running_stats:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()

    # -- description -------------------------------------------------------------------

    def feature_map_size(self, image_size: int) -> tuple[int, int]:  # pragma: no cover - abstract
        raise NotImplementedError

    def describe(self) -> dict:
        return {
            "encoder_model_id": self.model_id,
            "encoder_adapter_version": ADAPTER_VERSIONS[self.backbone],
            "encoder_out_channels": int(self.out_channels),
            "encoder_input_size": self.input_size,
            "encoder_trainable": self.freeze_mode,
            "encoder_trainable_units": self.trainable_units if self.freeze_mode == "last_n" else None,
            "encoder_units": len(self.units()),
        }


# --------------------------------------------------------------------------------------
# EfficientNet (torchvision) - the original encoder
# --------------------------------------------------------------------------------------


# model.backbone -> (torchvision builder, weights enum) names.
_EFFICIENTNETS = {
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights"),
    "efficientnet_v2_s": ("efficientnet_v2_s", "EfficientNet_V2_S_Weights"),
}


def build_encoder(weights: str = "IMAGENET1K_V1", backbone: str = DEFAULT_BACKBONE) -> tuple[nn.Module, dict]:
    """Create the torchvision EfficientNet with explicit weight provenance (the original loader)."""
    import torchvision
    import torchvision.models as tvm

    if backbone not in _EFFICIENTNETS:
        raise ValueError(f"build_encoder builds EfficientNets only ({sorted(_EFFICIENTNETS)}), got {backbone!r}")
    builder_name, weights_name = _EFFICIENTNETS[backbone]
    builder, weights_enum = getattr(tvm, builder_name), getattr(tvm, weights_name)

    info: dict[str, Any] = {
        "torchvision": torchvision.__version__,
        "backbone": backbone,
        "weights_request": str(weights),
    }
    spec = str(weights)
    local = None if _is_random(spec) else resolve_weight_path(spec)

    if _is_random(spec):
        _random_warning(backbone, spec)
        model = builder(weights=None)
        info["weights_used"] = "random"
    elif local is not None:
        model = builder(weights=None)
        state = torch.load(str(local), map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(
                f"Local encoder checkpoint {spec} is missing {len(missing)} parameters (e.g. {missing[:5]}) "
                f"for backbone {backbone}. Refusing to continue with a partially initialised encoder."
            )
        info["weights_used"] = str(local)
        info["unexpected_keys"] = len(unexpected)
        LOG.info("Loaded local encoder weights from %s (%d unexpected keys ignored)", local, len(unexpected))
    else:
        try:
            enum_value = getattr(weights_enum, spec)
        except AttributeError as exc:
            raise ValueError(
                f"Unknown model.weights={spec!r}. Use an {weights_name} name "
                f"(e.g. IMAGENET1K_V1), an existing local checkpoint path, or 'none'."
            ) from exc
        try:
            model = builder(weights=enum_value)
        except Exception as exc:  # download failure, offline machine, proxy, ...
            raise RuntimeError(
                f"Could not obtain pretrained weights {weights_name}.{spec}: {exc}. "
                "Never train silently from random initialisation. Either pre-download the weights "
                f"(TORCH_HOME=<dir> python -c \"from torchvision.models import {builder_name}, "
                f'{weights_name}; {builder_name}(weights={weights_name}.{spec})"), '
                "point model.weights at a local checkpoint, or set model.weights=none deliberately."
            ) from exc
        info["weights_used"] = spec
        info["weights_meta"] = {
            "num_params": getattr(enum_value, "meta", {}).get("num_params"),
            "categories": len(getattr(enum_value, "meta", {}).get("categories", []) or []),
        }
    return model, info


class EfficientNetEncoder(EncoderMixin, nn.Sequential):
    """torchvision EfficientNet `features` (stem, MBConv stages, 1x1 head conv), stride 32.

    It *is* the original `features` Sequential - same children, same order - so as the
    model's `features` attribute its state_dict keys stay `features.<i>.…` and every
    checkpoint written before the adapters existed loads unchanged. Units are the
    Sequential's stages.
    """

    def __init__(self, backbone: str = DEFAULT_BACKBONE, weights: str = "IMAGENET1K_V1") -> None:
        model, info = build_encoder(weights, backbone)
        nn.Sequential.__init__(self, *model.features.children())
        self._init_policy()
        self.backbone = backbone
        self.model_id = f"torchvision/{_EFFICIENTNETS[backbone][0]}"
        self.out_channels = ENCODER_FEATURES[backbone]
        self.weights_info = info

    def units(self) -> list[nn.Module]:
        return list(nn.Sequential.children(self))

    def feature_map_size(self, image_size: int) -> tuple[int, int]:
        return stride32_map_size(image_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.grad_checkpointing and self.training and torch.is_grad_enabled():
            # One segment per stage: only stage boundaries are kept for backward. Stochastic
            # depth replays identically, the RNG state is preserved by the checkpoint.
            return checkpoint_sequential(self, len(self), x, use_reentrant=False)
        return nn.Sequential.forward(self, x)


def stride32_map_size(image_size: int) -> tuple[int, int]:
    """Map size after the five stride-2 steps of EfficientNet / ResNet (each rounds up)."""
    side = int(image_size)
    for _ in range(5):
        side = -(-side // 2)
    return side, side


# --------------------------------------------------------------------------------------
# RadImageNet ResNet-50
# --------------------------------------------------------------------------------------


def convert_radimagenet_state(state: dict) -> dict:
    """Official RadImageNet `Backbone` state_dict -> torchvision ResNet-50 trunk names.

    Explicit prefix map, no pattern guessing: every key must start with one of
    RADIMAGENET_KEY_PREFIXES, otherwise the file is not the documented format.
    """
    if not isinstance(state, dict) or not state:
        raise ValueError("RadImageNet weights: expected a non-empty state_dict (mapping name -> tensor)")
    converted: dict[str, torch.Tensor] = {}
    foreign: list[str] = []
    for key, value in state.items():
        for prefix, target in RADIMAGENET_KEY_PREFIXES.items():
            if key.startswith(prefix):
                converted[target + key[len(prefix):]] = value
                break
        else:
            foreign.append(key)
    if foreign:
        raise ValueError(
            f"RadImageNet weights: {len(foreign)} keys outside the documented Backbone format "
            f"(backbone.0/1/4/5/6/7.*), e.g. {foreign[:5]}. Expected RadImageNet_pytorch/ResNet50.pt "
            "from the official RadImageNet PyTorch release."
        )
    return converted


def load_radimagenet_resnet50(module: nn.Module, path: str | Path) -> dict:
    """Load the official RadImageNet ResNet50.pt into `module` (torchvision trunk names), strictly.

    Every parameter AND every BatchNorm buffer (running_mean/var, num_batches_tracked) must
    come from the file with the right shape; a missing, extra or mis-shaped entry raises.
    """
    raw = torch.load(str(path), map_location="cpu", weights_only=True)
    state = convert_radimagenet_state(raw)
    expected = module.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = sorted(k for k in set(expected) & set(state) if tuple(expected[k].shape) != tuple(state[k].shape))
    if missing or unexpected or mismatched:
        raise ValueError(
            f"RadImageNet weights {path} do not match the ResNet-50 trunk: "
            f"{len(missing)} missing (e.g. {missing[:4]}), {len(unexpected)} unexpected (e.g. {unexpected[:4]}), "
            f"{len(mismatched)} with another shape (e.g. {mismatched[:4]}). Refusing a partial load."
        )
    non_finite = [k for k, v in state.items() if v.is_floating_point() and not torch.isfinite(v).all()]
    if non_finite:
        raise ValueError(f"RadImageNet weights {path} contain non-finite values in {non_finite[:4]}")
    module.load_state_dict(state, strict=True)
    n_buffers = sum(1 for k in expected if k.endswith(("running_mean", "running_var", "num_batches_tracked")))
    return {"n_tensors": len(state), "n_bn_buffers": n_buffers}


class RadImageNetResNet50Encoder(EncoderMixin, nn.Module):
    """torchvision ResNet-50 trunk up to `layer4` (before global pooling), stride 32.

    Keys are torchvision's (`conv1`, `bn1`, `layer1`…`layer4`); fc and avgpool are dropped.
    Units for `last_n` are the four residual stages; the stem only trains under `all`.
    """

    UNIT_NAMES = ("layer1", "layer2", "layer3", "layer4")

    def __init__(self, weights: str) -> None:
        nn.Module.__init__(self)
        import torchvision

        self._init_policy()
        self.backbone = "radimagenet_resnet50"
        self.model_id = "torchvision/resnet50+radimagenet"
        self.out_channels = ENCODER_FEATURES[self.backbone]
        base = torchvision.models.resnet50(weights=None)
        self.conv1, self.bn1, self.relu, self.maxpool = base.conv1, base.bn1, base.relu, base.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4

        spec = str(weights)
        info: dict[str, Any] = {"torchvision": torchvision.__version__, "backbone": self.backbone, "weights_request": spec}
        if _is_random(spec):
            _random_warning(self.backbone, spec)
            info["weights_used"] = "random"
        else:
            path = resolve_weight_path(spec)
            if path is None:
                raise FileNotFoundError(
                    f"model.weights={spec!r}: no such file. radimagenet_resnet50 needs the RadImageNet "
                    "PyTorch weights RadImageNet_pytorch/ResNet50.pt (official release linked from "
                    "github.com/BMEII-AI/RadImageNet), or model.weights=none for a deliberate random start. "
                    "ImageNet weights are never substituted."
                )
            digest = sha256_file(path)
            counts = load_radimagenet_resnet50(self, path)
            info.update({"weights_used": str(path), "sha256": digest, "format": "radimagenet_torch_backbone", **counts})
            if digest != RADIMAGENET_RESNET50_SHA256:
                LOG.warning(
                    "RadImageNet weights %s loaded strictly, but their sha256 %s is not the verified official "
                    "ResNet50.pt (%s). Make sure this is the file you intend.",
                    path,
                    digest[:16],
                    RADIMAGENET_RESNET50_SHA256[:16],
                )
            LOG.info("Loaded RadImageNet ResNet-50 weights from %s (%d tensors, sha256 %s)", path, counts["n_tensors"], digest[:16])
        self.weights_info = info

    def units(self) -> list[nn.Module]:
        return [getattr(self, name) for name in self.UNIT_NAMES]

    def feature_map_size(self, image_size: int) -> tuple[int, int]:
        return stride32_map_size(image_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        use_ckpt = self.grad_checkpointing and self.training and torch.is_grad_enabled()
        for stage in self.units():
            trains = any(p.requires_grad for p in stage.parameters())
            if use_ckpt and trains:
                # Non-reentrant: gradients reach the stage parameters even when its input
                # comes out of a frozen stage and does not require grad.
                x = checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
        return x


# --------------------------------------------------------------------------------------
# DINOv2 ViT-S/14 (timm)
# --------------------------------------------------------------------------------------


def dinov2_grid(image_size: int, patch_size: int = DINOV2_PATCH_SIZE) -> tuple[int, int]:
    """Patch grid of a square input; refuses sizes the patch does not divide (no hidden padding)."""
    size = int(image_size)
    if size <= 0 or size % patch_size != 0:
        raise ValueError(
            f"data.image_size={size} is not a multiple of the DINOv2 patch size {patch_size}. The "
            "dinov2_vits14 adapter does no padding, cropping or resizing; use e.g. 224 (16x16 patches) "
            "or 336 (24x24 patches)."
        )
    return size // patch_size, size // patch_size


class DinoV2Encoder(EncoderMixin, nn.Module):
    """timm DINOv2 ViT-S/14: final-norm patch tokens rearranged into a `[N, 384, H/14, W/14]` map.

    `forward_features()` returns `[N, prefix + gh*gw, C]` after the final LayerNorm. The CLS
    token (and any register tokens, `num_prefix_tokens` in total) are dropped - they are not
    image positions. timm's patch embedding flattens the grid row-major (`flatten(2)` of
    `[N, C, gh, gw]`), so token `i*gw + j` is patch row i, column j; the selftest checks this
    against timm's own `forward_intermediates` and on a constructed input.

    The model is created for exactly `image_size` px: timm resamples the pretrained 518 px
    (37x37) position embeddings to this grid when the weights are loaded, and refuses any
    other input size at runtime. Units for `last_n` are the 12 transformer blocks; the final
    norm trains with them; patch embedding, CLS token and position embeddings stay frozen.
    """

    def __init__(self, weights: str, image_size: int) -> None:
        nn.Module.__init__(self)
        self._init_policy()
        self.backbone = "dinov2_vits14"
        self.model_id = DINOV2_HF_HUB_ID
        self.patch_size = DINOV2_PATCH_SIZE
        self.input_size = int(image_size)
        self.grid = dinov2_grid(self.input_size, self.patch_size)

        try:
            import timm
        except ImportError as exc:  # pragma: no cover - environment
            raise ImportError("model.backbone=dinov2_vits14 needs timm (pip install 'timm>=1.0')") from exc

        spec = str(weights)
        info: dict[str, Any] = {"timm": timm.__version__, "backbone": self.backbone, "weights_request": spec}
        kwargs = {"img_size": self.input_size, "num_classes": 0}
        if _is_random(spec):
            _random_warning(self.backbone, spec)
            self.model = timm.create_model(DINOV2_TIMM_NAME, pretrained=False, **kwargs)
            info["weights_used"] = "random"
        else:
            path = self._pretrained_file(spec)
            self.model = timm.create_model(
                f"{DINOV2_TIMM_NAME}.{DINOV2_TIMM_TAG}", pretrained=True, pretrained_cfg_overlay={"file": str(path)}, **kwargs
            )
            self._verify_loaded(path)
            info.update({"weights_used": str(path), "sha256": sha256_file(path), "hf_hub_id": DINOV2_HF_HUB_ID})
            LOG.info("Loaded DINOv2 ViT-S/14 weights from %s at %d px (%dx%d patches)", path, self.input_size, *self.grid)
        self.weights_info = info

        embed = self.model.patch_embed
        if tuple(embed.patch_size) != (self.patch_size, self.patch_size):
            raise RuntimeError(f"unexpected DINOv2 patch size {embed.patch_size}")
        if tuple(embed.grid_size) != self.grid:
            raise RuntimeError(f"timm built a {embed.grid_size} grid for {self.input_size} px, expected {self.grid}")
        self.num_prefix_tokens = int(self.model.num_prefix_tokens)
        self.out_channels = int(self.model.embed_dim)
        if self.out_channels != ENCODER_FEATURES[self.backbone]:
            raise RuntimeError(f"DINOv2 embed_dim {self.out_channels} != {ENCODER_FEATURES[self.backbone]}")
        expected_tokens = self.num_prefix_tokens + self.grid[0] * self.grid[1]
        if int(self.model.pos_embed.shape[1]) != expected_tokens:
            raise RuntimeError(
                f"DINOv2 position embeddings cover {self.model.pos_embed.shape[1]} tokens, expected "
                f"{expected_tokens} for {self.input_size} px - they were not resampled to the configured size"
            )

    @staticmethod
    def _pretrained_file(spec: str) -> Path:
        """`lvd142m` = the timm/HF hub file (cached after the first download); else a local timm-format file."""
        if spec == DINOV2_TIMM_TAG:
            try:
                from huggingface_hub import hf_hub_download

                return Path(hf_hub_download(DINOV2_HF_HUB_ID, DINOV2_HF_FILE))
            except Exception as exc:
                raise RuntimeError(
                    f"Could not obtain {DINOV2_HF_HUB_ID}/{DINOV2_HF_FILE}: {exc}. Never train silently from "
                    "random initialisation. Download it once with network access (it is then cached, "
                    "HF_HUB_OFFLINE=1 works afterwards), point model.weights at a local copy of that file, "
                    "or set model.weights=none deliberately."
                ) from exc
        path = resolve_weight_path(spec)
        if path is None:
            raise FileNotFoundError(
                f"model.weights={spec!r}: use 'lvd142m' (timm/HF hub {DINOV2_HF_HUB_ID}), a local copy of "
                f"its {DINOV2_HF_FILE}, or 'none'."
            )
        return path

    def _verify_loaded(self, path: Path) -> None:
        """The file's last block really is in the model (timm loads strictly; this is a cheap cross-check)."""
        key = f"blocks.{len(self.model.blocks) - 1}.mlp.fc2.weight"
        if path.suffix == ".safetensors":
            from safetensors import safe_open

            with safe_open(str(path), framework="pt") as handle:
                reference = handle.get_tensor(key) if key in handle.keys() else None
        else:
            state = torch.load(str(path), map_location="cpu", weights_only=True)
            reference = state.get(key)
        if reference is None:
            raise ValueError(f"{path} is not a timm-format DINOv2 checkpoint (no {key}); use the timm/HF hub file")
        if not torch.equal(reference.to(self.model.state_dict()[key].dtype), self.model.state_dict()[key]):
            raise RuntimeError(f"DINOv2 weights from {path} did not end up in the model ({key} differs)")

    def units(self) -> list[nn.Module]:
        return list(self.model.blocks)

    def last_n_trainable_modules(self, n: int) -> list[nn.Module]:
        return list(self.model.blocks)[-n:] + [self.model.norm]

    def feature_map_size(self, image_size: int) -> tuple[int, int]:
        return dinov2_grid(image_size, self.patch_size)

    def patch_tokens_to_map(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """`[N, prefix + gh*gw, C]` -> `[N, C, gh, gw]` (row-major token order)."""
        gh, gw = height // self.patch_size, width // self.patch_size
        n, count, channels = tokens.shape
        if count != self.num_prefix_tokens + gh * gw:
            raise RuntimeError(
                f"DINOv2 returned {count} tokens for a {height}x{width} input; expected "
                f"{self.num_prefix_tokens} prefix + {gh}x{gw} patch tokens"
            )
        patches = tokens[:, self.num_prefix_tokens :]
        return patches.reshape(n, gh, gw, channels).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = int(x.shape[-2]), int(x.shape[-1])
        if (height, width) != (self.input_size, self.input_size):
            raise ValueError(
                f"the dinov2_vits14 encoder was built for {self.input_size}x{self.input_size} inputs "
                f"(data.image_size at build time), got {height}x{width}"
            )
        use_ckpt = self.grad_checkpointing and self.training and torch.is_grad_enabled()
        if use_ckpt:
            from timm.layers.config import use_reentrant_ckpt

            if use_reentrant_ckpt():
                raise RuntimeError(
                    "timm is set to reentrant checkpointing (TIMM_REENTRANT_CKPT): with a partially frozen "
                    "encoder the trainable blocks would silently get no gradients. Unset it."
                )
        self.model.grad_checkpointing = use_ckpt
        return self.patch_tokens_to_map(self.model.forward_features(x), height, width)


# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------


def build_encoder_adapter(
    backbone: str = DEFAULT_BACKBONE,
    weights: str = "IMAGENET1K_V1",
    image_size: int | None = None,
    freeze: str = "all",
    trainable_units: int = 0,
    grad_checkpointing: bool = False,
) -> nn.Module:
    """Build the adapter for `backbone` with its pretrained source and freeze policy applied."""
    if backbone in _EFFICIENTNETS:
        encoder: nn.Module = EfficientNetEncoder(backbone, weights)
    elif backbone == "radimagenet_resnet50":
        encoder = RadImageNetResNet50Encoder(weights)
    elif backbone == "dinov2_vits14":
        if image_size is None:
            raise ValueError("dinov2_vits14 needs the input size (data.image_size) at build time")
        encoder = DinoV2Encoder(weights, int(image_size))
    else:
        raise ValueError(f"Unknown model.backbone={backbone!r}; known: {sorted(ENCODER_FEATURES)}")
    if len(encoder.units()) != ENCODER_UNITS[backbone]:
        raise RuntimeError(f"{backbone}: {len(encoder.units())} units, constants say {ENCODER_UNITS[backbone]}")
    encoder.apply_freeze(freeze, trainable_units)
    encoder.grad_checkpointing = bool(grad_checkpointing)
    return encoder
