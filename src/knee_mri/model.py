"""EfficientNet-B0 encoder + masked multiple-instance pooling over slice bags.

Tensor flow
-----------
    [B, P, S, 3, H, W] -> gather valid triplets -> [N_valid, 3, H, W]
    -> EfficientNet-B0 features -> spatial pooling (model.spatial_pool) -> [N_valid, F]
       avg: global average (1280) | avgmax: average and maximum (2560)
       attention: softmax-weighted average over the map (1280), zero-initialised to avg
    -> scatter back to [B, P, S, F] (differentiable)
    -> masked mean and feature-wise masked max over S -> [B, P, 2*F]
    -> concat fixed-order slots + P presence flags -> [B, P*2*F + P]
    -> Linear(.., 256) -> ReLU -> Dropout -> Linear(256, 12) -> logits

Padding never enters the mean denominator or the max, a fully missing slot produces
an exactly zero finite feature vector, and no fake image is ever encoded (that would
let BatchNorm learn from padding). The model returns logits; sigmoid is applied only
for metrics and exported predictions. There is no softmax across targets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .constants import EFFICIENTNET_B0_FEATURES, N_TARGETS
from .utils import LOG

# A finite sentinel for the masked maximum: -inf would make an all-masked max
# non-finite and poison the head.
NEG_SENTINEL = -1.0e30
# How one slice's H x W feature map becomes a vector. avg = the original global average.
SPATIAL_POOLS = ("avg", "avgmax", "attention")


def build_encoder(weights: str = "IMAGENET1K_V1") -> tuple[nn.Module, dict]:
    """Create the EfficientNet-B0 feature extractor with explicit weight provenance."""
    import torchvision
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    info: dict[str, Any] = {"torchvision": torchvision.__version__, "weights_request": str(weights)}
    spec = str(weights)

    if spec.lower() in ("none", "random", "null"):
        LOG.warning(
            "model.weights=%s: the encoder starts from RANDOM initialisation. This is an intentional "
            "ablation, not a fallback.",
            spec,
        )
        model = efficientnet_b0(weights=None)
        info["weights_used"] = "random"
    elif Path(spec).expanduser().exists():
        model = efficientnet_b0(weights=None)
        state = torch.load(str(Path(spec).expanduser()), map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(
                f"Local encoder checkpoint {spec} is missing {len(missing)} parameters (e.g. {missing[:5]}). "
                "Refusing to continue with a partially initialised encoder."
            )
        info["weights_used"] = str(spec)
        info["unexpected_keys"] = len(unexpected)
        LOG.info("Loaded local encoder weights from %s (%d unexpected keys ignored)", spec, len(unexpected))
    else:
        try:
            enum_value = getattr(EfficientNet_B0_Weights, spec)
        except AttributeError as exc:
            raise ValueError(
                f"Unknown model.weights={spec!r}. Use an EfficientNet_B0_Weights name "
                f"(e.g. IMAGENET1K_V1), an existing local checkpoint path, or 'none'."
            ) from exc
        try:
            model = efficientnet_b0(weights=enum_value)
        except Exception as exc:  # download failure, offline machine, proxy, ...
            raise RuntimeError(
                f"Could not obtain pretrained weights {spec}: {exc}. "
                "Never train silently from random initialisation. Either pre-download the weights "
                "(TORCH_HOME=<dir> python -c \"from torchvision.models import efficientnet_b0, "
                'EfficientNet_B0_Weights; efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)"), '
                "point model.weights at a local checkpoint, or set model.weights=none deliberately."
            ) from exc
        info["weights_used"] = spec
        info["weights_meta"] = {
            "num_params": getattr(enum_value, "meta", {}).get("num_params"),
            "categories": len(getattr(enum_value, "meta", {}).get("categories", []) or []),
        }
    return model, info


def masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over S using only valid entries: `[B, P, S, F] x [B, P, S] -> [B, P, F]`."""
    weights = mask.to(features.dtype).unsqueeze(-1)
    total = (features * weights).sum(dim=2)
    count = weights.sum(dim=2).clamp_min(1.0)
    return total / count


def masked_max(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Feature-wise maximum over S using only valid entries; all-masked -> exact zeros."""
    keep = mask.unsqueeze(-1)
    filled = features.masked_fill(~keep, NEG_SENTINEL)
    maxed = filled.max(dim=2).values
    any_valid = mask.any(dim=2, keepdim=True)
    return torch.where(any_valid, maxed, torch.zeros_like(maxed))


class SpatialAttentionPool(nn.Module):
    """Learned weighted average over the H x W feature map of one slice.

    A 3 mm finding covers a few cells of the 7x7 (224 px) or 10x10 (320 px) map, so a plain
    average dilutes it by the map size. The scores come from a two-layer 1x1 convolution and
    the softmax runs in float32, because bf16 would quantise near-uniform weights.

    The last layer is zero-initialised, so training STARTS from exactly uniform weights, i.e.
    from average pooling; the module can only deviate from it by learning to.
    """

    def __init__(self, channels: int, hidden: int = 128) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.Tanh(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        nn.init.zeros_(self.score[2].weight)
        nn.init.zeros_(self.score[2].bias)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        flat = feature_map.flatten(2).float()  # [N, C, H*W]
        weights = torch.softmax(self.score(feature_map).flatten(2).float(), dim=-1)  # [N, 1, H*W]
        return (flat * weights).sum(dim=-1)

    def attention_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        """[N, H, W] weights for QC; they sum to 1 per slice."""
        n, _, h, w = feature_map.shape
        return torch.softmax(self.score(feature_map).flatten(2).float(), dim=-1).view(n, h, w)


class EfficientNetB0MIL(nn.Module):
    """Study-level multi-label classifier over per-slot slice bags."""

    def __init__(
        self,
        n_targets: int = N_TARGETS,
        n_slots: int = 3,
        weights: str = "IMAGENET1K_V1",
        head_hidden: int = 256,
        head_dropout: float = 0.2,
        freeze_bn_running_stats: bool = True,
        encoder_chunk_size: int = 0,
        spatial_pool: str = "avg",
    ) -> None:
        super().__init__()
        if spatial_pool not in SPATIAL_POOLS:
            raise ValueError(f"spatial_pool must be one of {sorted(SPATIAL_POOLS)}, got {spatial_pool!r}")
        encoder, self.weights_info = build_encoder(weights)
        self.features = encoder.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.spatial_pool = str(spatial_pool)
        self.attention_pool = (
            SpatialAttentionPool(EFFICIENTNET_B0_FEATURES) if self.spatial_pool == "attention" else None
        )
        self.n_slots = int(n_slots)
        self.n_targets = int(n_targets)
        # `avgmax` keeps an average and a maximum descriptor per slice, so the slice vector doubles.
        self.feature_dim = EFFICIENTNET_B0_FEATURES * (2 if self.spatial_pool == "avgmax" else 1)
        self.freeze_bn_running_stats = bool(freeze_bn_running_stats)
        self.encoder_chunk_size = int(encoder_chunk_size)

        head_in = self.n_slots * 2 * self.feature_dim + self.n_slots
        self.head_in = head_in
        self.head = nn.Sequential(
            nn.Linear(head_in, int(head_hidden)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(head_dropout)),
            nn.Linear(int(head_hidden), self.n_targets),
        )

    # -- BatchNorm policy -------------------------------------------------------------

    def _apply_bn_policy(self) -> None:
        if not self.freeze_bn_running_stats:
            return
        for module in self.features.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()  # use the pretrained running statistics; affine params stay trainable

    def train(self, mode: bool = True) -> "EfficientNetB0MIL":
        super().train(mode)
        if mode:
            self._apply_bn_policy()
        return self

    # -- Encoding ---------------------------------------------------------------------

    def _pool_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        """`[N, C, H, W] -> [N, feature_dim]` under the configured spatial policy."""
        if self.spatial_pool == "avg":
            return self.pool(feature_map).flatten(1)
        if self.spatial_pool == "avgmax":
            averaged = self.pool(feature_map).flatten(1)
            maxed = feature_map.flatten(2).max(dim=-1).values
            return torch.cat([averaged, maxed], dim=-1)
        assert self.attention_pool is not None
        return self.attention_pool(feature_map)

    def encode(self, triplets: torch.Tensor) -> torch.Tensor:
        """`[N, 3, H, W] -> [N, feature_dim]`.

        Chunking only limits the size of a single encoder call. During training the
        autograd graph of every chunk is retained until backward, so chunking is not a
        guaranteed memory saving there - reduce train.microbatch_studies or
        data.centers_per_series instead.
        """
        if triplets.shape[0] == 0:
            return triplets.new_zeros((0, self.feature_dim))
        if self.encoder_chunk_size and self.encoder_chunk_size > 0:
            outputs = [
                self._pool_map(self.features(chunk)) for chunk in triplets.split(self.encoder_chunk_size, dim=0)
            ]
            return torch.cat(outputs, dim=0)
        return self._pool_map(self.features(triplets))

    # -- Forward ----------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,  # [B, P, S, 3, H, W]
        slice_valid_mask: torch.Tensor,  # [B, P, S]
        series_present_mask: torch.Tensor,  # [B, P]
    ) -> torch.Tensor:
        if images.dim() != 6:
            raise ValueError(f"images must be [B, P, S, 3, H, W], got {tuple(images.shape)}")
        b, p, s, k, h, w = images.shape
        if p != self.n_slots:
            raise ValueError(f"model was built for {self.n_slots} slots but received P={p}")
        if k != 3:
            raise ValueError(f"expected 3 adjacent-slice channels, got {k}")
        if slice_valid_mask.shape != (b, p, s):
            raise ValueError(f"slice_valid_mask must be [B, P, S] = {(b, p, s)}, got {tuple(slice_valid_mask.shape)}")
        if series_present_mask.shape != (b, p):
            raise ValueError(
                f"series_present_mask must be [B, P] = {(b, p)}, got {tuple(series_present_mask.shape)}"
            )

        # A slot that is absent cannot contribute valid slices.
        valid = slice_valid_mask & series_present_mask.unsqueeze(-1)
        flat_images = images.reshape(b * p * s, k, h, w)
        flat_valid = valid.reshape(b * p * s)
        indices = flat_valid.nonzero(as_tuple=False).squeeze(1)

        features = flat_images.new_zeros((b * p * s, self.feature_dim))
        if indices.numel() > 0:
            encoded = self.encode(flat_images.index_select(0, indices))
            # index_copy keeps this differentiable and leaves padded rows at exact zero.
            features = features.index_copy(0, indices, encoded.to(features.dtype))
        features = features.view(b, p, s, self.feature_dim).float()

        mean_features = masked_mean(features, valid)
        max_features = masked_max(features, valid)
        slot_features = torch.cat([mean_features, max_features], dim=-1)  # [B, P, 2*1280]
        flat_slots = slot_features.reshape(b, p * 2 * self.feature_dim)
        presence = series_present_mask.to(flat_slots.dtype)
        head_input = torch.cat([flat_slots, presence], dim=-1)
        return self.head(head_input)

    # -- Convenience ------------------------------------------------------------------

    def parameter_groups(self, encoder_lr: float, head_lr: float, weight_decay: float) -> list[dict]:
        return [
            {"params": list(self.features.parameters()), "lr": float(encoder_lr), "weight_decay": float(weight_decay)},
            {
                # The attention scorer is new, like the head: same (higher) learning rate.
                "params": list(self.head.parameters())
                + (list(self.attention_pool.parameters()) if self.attention_pool is not None else []),
                "lr": float(head_lr),
                "weight_decay": float(weight_decay),
            },
        ]

    def describe(self) -> dict:
        return {
            "architecture": "efficientnet_b0_2p5d_mil",
            "n_slots": self.n_slots,
            "n_targets": self.n_targets,
            "spatial_pool": self.spatial_pool,
            "feature_dim": self.feature_dim,
            "head_in": self.head_in,
            "freeze_bn_running_stats": self.freeze_bn_running_stats,
            "encoder_chunk_size": self.encoder_chunk_size,
            "weights_info": self.weights_info,
            "n_parameters": int(sum(p.numel() for p in self.parameters())),
            "n_trainable": int(sum(p.numel() for p in self.parameters() if p.requires_grad)),
        }


def build_model(cfg, n_slots: int | None = None) -> EfficientNetB0MIL:
    slots = n_slots if n_slots is not None else len(cfg.data.series_slots)
    model = EfficientNetB0MIL(
        n_targets=N_TARGETS,
        n_slots=slots,
        weights=str(cfg.model.weights),
        head_hidden=int(cfg.model.head_hidden),
        head_dropout=float(cfg.model.head_dropout),
        freeze_bn_running_stats=bool(cfg.model.freeze_bn_running_stats),
        encoder_chunk_size=int(cfg.model.encoder_chunk_size),
        spatial_pool=str(cfg.model.get("spatial_pool", "avg")),
    )
    LOG.info("Model: %s", model.describe())
    return model
