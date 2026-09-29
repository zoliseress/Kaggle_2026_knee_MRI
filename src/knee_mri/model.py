"""2D slice encoder (model.backbone) + masked multiple-instance pooling over slice bags.

Tensor flow
-----------
    [B, P, S, 3, H, W] -> gather valid triplets -> [N_valid, 3, H, W]
    -> encoder adapter (encoders.py) -> feature map [N_valid, C, Hf, Wf]
       efficientnet_b0 / _v2_s: C=1280, 7x7 at 224 px | radimagenet_resnet50: C=2048, 7x7
       dinov2_vits14: C=384, 16x16 patch-token map at 224 px
    -> spatial pooling (model.spatial_pool) -> [N_valid, F]
       avg: global average (C) | avgmax: average and maximum (2C)
       attention: softmax-weighted average over the map (C), zero-initialised to avg
    -> scatter back to [B, P, S, F] (differentiable)
    -> masked mean and feature-wise masked max over S -> [B, P, 2*F]
    -> concat fixed-order slots + P presence flags -> [B, P*2*F + P]
    -> Linear(.., 256) -> ReLU -> Dropout -> Linear(256, 12) -> logits

Every width downstream of the encoder comes from the adapter's `out_channels`.

Optional target attention (model.target_attention, needs spatial_pool=avg): every window
also yields two half-map vectors (column halves of its feature map). With
data.laterality_canonical these are medial|lateral on coronal/axial and posterior|anterior
on sagittal, and a window's depth zone (thirds of the valid windows) is medial/central/lateral
on sagittal. Each target has its own query over all (slot, zone, half) tokens of the study;
its pooled vector adds a per-target logit to the head's. The per-target output weights start
at zero, so training starts from exactly the mean/max model.

Optional side pooling (model.side_pooling, needs spatial_pool=avg): the head no longer sees
one mean/max per slot but one per SIDE of it - the two column halves of coronal/axial
(medial | lateral in the canonical frame) and the depth zones of sagittal (medial / central /
lateral). The slot mean is the mean of the half means (even map width) and the sagittal max
is the max of the zone maxes, so little is lost (only the coronal/axial max of whole-slice
averages); what is gained is that a linear head can tell a medial from a lateral finding:
    coronal/axial [B, 2 halves * (mean, max) * F], sagittal [B, zones * (mean, max) * F]

Padding never enters the mean denominator or the max, a fully missing slot produces
an exactly zero finite feature vector, and no fake image is ever encoded (that would
let BatchNorm learn from padding). The model returns logits; sigmoid is applied only
for metrics and exported predictions. There is no softmax across targets.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .constants import DEFAULT_BACKBONE, ENCODER_FEATURES, N_TARGETS, PLANES
from .encoders import build_encoder, build_encoder_adapter  # noqa: F401  (build_encoder: public, EfficientNet only)
from .utils import LOG

# A finite sentinel for the masked maximum: -inf would make an all-masked max
# non-finite and poison the head.
NEG_SENTINEL = -1.0e30
# How one slice's H x W feature map becomes a vector. avg = the original global average.
SPATIAL_POOLS = ("avg", "avgmax", "attention")


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


def depth_zones(valid: torch.Tensor, n_zones: int) -> torch.Tensor:
    """`[B, P, S]` zone index of each window: its rank among the slot's VALID windows, in thirds (for 3).

    Windows are ordered by slice index, so with the canonical laterality frame the sagittal
    zones run medial -> lateral. Padded windows get zone 0; they are masked anyway.
    """
    rank = valid.long().cumsum(dim=-1) - 1
    count = valid.long().sum(dim=-1, keepdim=True).clamp_min(1)
    zone = torch.div(rank.clamp_min(0) * n_zones, count, rounding_mode="floor")
    return zone.clamp(0, n_zones - 1).masked_fill(~valid, 0)


class TargetAttentionReadout(nn.Module):
    """Per-target attention over a study's (slot, depth zone, map half) window tokens.

    score[t, token] = <key(h), q_t> / sqrt(d) + bias[t, slot, zone, half]; softmax over the
    study's valid tokens; logit_t = w_t . LayerNorm(sum a * value(h)) + b_t. h carries learned
    slot, zone and half embeddings, so a target can prefer e.g. lateral sagittal windows.
    w and b start at zero: the readout adds nothing until it has learned something.
    """

    def __init__(self, feature_dim: int, n_slots: int, n_targets: int, dim: int = 256, n_zones: int = 3, n_halves: int = 2) -> None:
        super().__init__()
        self.dim, self.n_zones, self.n_halves = int(dim), int(n_zones), int(n_halves)
        self.proj = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, self.dim), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slots, self.dim) * 0.02)
        self.zone_emb = nn.Parameter(torch.randn(self.n_zones, self.dim) * 0.02)
        self.half_emb = nn.Parameter(torch.randn(self.n_halves, self.dim) * 0.02)
        self.key = nn.Linear(self.dim, self.dim)
        self.value = nn.Linear(self.dim, self.dim)
        self.query = nn.Parameter(torch.randn(n_targets, self.dim) * 0.02)
        self.position_bias = nn.Parameter(torch.zeros(n_targets, n_slots, self.n_zones, self.n_halves))
        self.norm = nn.LayerNorm(self.dim)
        self.out_weight = nn.Parameter(torch.zeros(n_targets, self.dim))
        self.out_bias = nn.Parameter(torch.zeros(n_targets))

    def forward(
        self, tokens: torch.Tensor, valid: torch.Tensor, zones: torch.Tensor, return_attention: bool = False
    ):
        """tokens `[B, P, S, H, F]`, valid `[B, P, S]`, zones `[B, P, S]` -> logits `[B, T]`."""
        b, p, s, halves, _ = tokens.shape
        h = self.proj(tokens.float())
        h = h + self.slot_emb.view(1, p, 1, 1, self.dim) + self.zone_emb[zones].unsqueeze(3) + self.half_emb.view(1, 1, 1, halves, self.dim)
        keys = self.key(h).reshape(b, p * s * halves, self.dim)
        values = self.value(h).reshape(b, p * s * halves, self.dim)

        scores = torch.einsum("bnd,td->btn", keys, self.query) / math.sqrt(self.dim)
        slot_index = torch.arange(p, device=tokens.device).view(1, p, 1, 1).expand(b, p, s, halves)
        zone_index = zones.unsqueeze(-1).expand(b, p, s, halves)
        half_index = torch.arange(halves, device=tokens.device).view(1, 1, 1, halves).expand(b, p, s, halves)
        bias = self.position_bias[:, slot_index, zone_index, half_index]  # [T, B, P, S, H]
        scores = scores + bias.permute(1, 0, 2, 3, 4).reshape(b, -1, p * s * halves)

        token_valid = valid.unsqueeze(-1).expand(b, p, s, halves).reshape(b, 1, p * s * halves)
        scores = scores.float().masked_fill(~token_valid, float("-inf"))
        any_valid = token_valid.any(dim=-1, keepdim=True)  # [B, 1, 1]
        weights = torch.softmax(torch.where(any_valid, scores, torch.zeros_like(scores)), dim=-1)
        weights = torch.where(any_valid & token_valid, weights, torch.zeros_like(weights))
        pooled = torch.einsum("btn,bnd->btd", weights, values.float())
        logits = (self.norm(pooled) * self.out_weight).sum(dim=-1) + self.out_bias
        logits = torch.where(any_valid.view(b, 1), logits, torch.zeros_like(logits))
        if return_attention:
            return logits, weights.view(b, -1, p, s, halves)
        return logits


class EfficientNetMIL(nn.Module):
    """Study-level multi-label classifier over per-slot slice bags.

    The name is historical: the encoder is any adapter from encoders.py (model.backbone).
    It lives in `self.features`, so EfficientNet checkpoints keep their `features.<i>.…` keys.
    """

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
        backbone: str = DEFAULT_BACKBONE,
        grad_checkpointing: bool = False,
        target_attention: bool = False,
        attention_dim: int = 256,
        n_depth_zones: int = 3,
        side_pooling: bool = False,
        slot_names: list[str] | tuple[str, ...] | None = None,
        image_size: int | None = None,
        encoder_trainable: str = "all",
        encoder_trainable_units: int = 0,
    ) -> None:
        super().__init__()
        if spatial_pool not in SPATIAL_POOLS:
            raise ValueError(f"spatial_pool must be one of {sorted(SPATIAL_POOLS)}, got {spatial_pool!r}")
        if target_attention and spatial_pool != "avg":
            raise ValueError("model.target_attention needs model.spatial_pool=avg (it pools map halves itself)")
        if side_pooling and spatial_pool != "avg":
            raise ValueError("model.side_pooling needs model.spatial_pool=avg (it pools map halves itself)")
        self.backbone = str(backbone)
        # The adapter maps [N, 3, H, W] -> [N, C, Hf, Wf]; image_size only matters for DINOv2,
        # whose position embeddings are built for exactly that input.
        self.features = build_encoder_adapter(
            self.backbone,
            weights,
            image_size=image_size,
            freeze=str(encoder_trainable),
            trainable_units=int(encoder_trainable_units),
            grad_checkpointing=bool(grad_checkpointing),
        )
        self.weights_info = self.features.weights_info
        encoder_features = int(self.features.out_channels)
        if encoder_features != ENCODER_FEATURES[self.backbone]:
            raise RuntimeError(
                f"{self.backbone} adapter gives {encoder_features} channels, expected {ENCODER_FEATURES[self.backbone]}"
            )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.spatial_pool = str(spatial_pool)
        self.attention_pool = (
            SpatialAttentionPool(encoder_features) if self.spatial_pool == "attention" else None
        )
        self.n_slots = int(n_slots)
        self.n_targets = int(n_targets)
        # `avgmax` keeps an average and a maximum descriptor per slice, so the slice vector doubles.
        self.feature_dim = encoder_features * (2 if self.spatial_pool == "avgmax" else 1)
        self.freeze_bn_running_stats = bool(freeze_bn_running_stats)
        self.encoder_chunk_size = int(encoder_chunk_size)

        self.n_depth_zones = int(n_depth_zones)
        self.side_pooling = bool(side_pooling)
        names = list(slot_names) if slot_names is not None else list(PLANES[: self.n_slots])
        if len(names) != self.n_slots:
            raise ValueError(f"slot_names {names} do not match n_slots={self.n_slots}")
        # How each slot is split into sides: sagittal by depth zone (slice order), the others by map half.
        self.slot_sides = tuple("zones" if str(n).lower().startswith("sag") else "halves" for n in names)
        if self.side_pooling:
            parts_per_slot = [self.n_depth_zones if mode == "zones" else 2 for mode in self.slot_sides]
        else:
            parts_per_slot = [1] * self.n_slots
        head_in = sum(parts_per_slot) * 2 * self.feature_dim + self.n_slots
        self.head_in = head_in
        self.head = nn.Sequential(
            nn.Linear(head_in, int(head_hidden)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(head_dropout)),
            nn.Linear(int(head_hidden), self.n_targets),
        )
        self.target_attention = bool(target_attention)
        self.readout = (
            TargetAttentionReadout(encoder_features, self.n_slots, self.n_targets, int(attention_dim), self.n_depth_zones)
            if self.target_attention
            else None
        )

    # -- Encoder policies -------------------------------------------------------------

    @property
    def encoder(self) -> nn.Module:
        """The encoder adapter (the same module as `features`)."""
        return self.features

    @property
    def grad_checkpointing(self) -> bool:
        return bool(self.features.grad_checkpointing)

    @grad_checkpointing.setter
    def grad_checkpointing(self, enabled: bool) -> None:
        self.features.grad_checkpointing = bool(enabled)

    def _apply_bn_policy(self) -> None:
        # Frozen encoder units run in eval mode (pretrained BN statistics, no dropout); with
        # freeze_bn_running_stats every encoder BN keeps its running statistics, while the
        # affine parameters of trainable units still train.
        self.features.apply_train_policy(self.freeze_bn_running_stats)

    def train(self, mode: bool = True) -> "EfficientNetMIL":
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

    def encode_with_halves(self, triplets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`[N, 3, H, W] -> ([N, F] global average, [N, 2, F] column-half averages)`.

        The two halves share the middle column when the map width is odd; their mean is then
        not exactly the global average, which is why both are returned.
        """
        def pooled(feature_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            width = feature_map.shape[-1]
            half = (width + 1) // 2
            left = feature_map[..., :half].mean(dim=(-2, -1))
            right = feature_map[..., width - half :].mean(dim=(-2, -1))
            return self.pool(feature_map).flatten(1), torch.stack([left, right], dim=1)

        if triplets.shape[0] == 0:
            empty = triplets.new_zeros((0, self.feature_dim))
            return empty, triplets.new_zeros((0, 2, self.feature_dim))
        chunks = (
            triplets.split(self.encoder_chunk_size, dim=0)
            if self.encoder_chunk_size and self.encoder_chunk_size > 0
            else (triplets,)
        )
        outputs = [pooled(self._run_features(chunk)) for chunk in chunks]
        return torch.cat([o[0] for o in outputs], dim=0), torch.cat([o[1] for o in outputs], dim=0)

    def encode(self, triplets: torch.Tensor) -> torch.Tensor:
        """`[N, 3, H, W] -> [N, feature_dim]`.

        Chunking only limits the size of a single encoder call. During training the
        autograd graph of every chunk is retained until backward, so chunking is not a
        guaranteed memory saving there. model.grad_checkpointing lowers it (each encoder stage
        or transformer block is recomputed in backward instead of keeping its activations);
        freezing encoder units lowers it too, since frozen units keep no autograd graph.
        """
        if triplets.shape[0] == 0:
            return triplets.new_zeros((0, self.feature_dim))
        if self.encoder_chunk_size and self.encoder_chunk_size > 0:
            outputs = [
                self._pool_map(self._run_features(chunk)) for chunk in triplets.split(self.encoder_chunk_size, dim=0)
            ]
            return torch.cat(outputs, dim=0)
        return self._pool_map(self._run_features(triplets))

    def _run_features(self, x: torch.Tensor) -> torch.Tensor:
        # Gradient checkpointing is the adapter's own (per stage / per transformer block).
        return self.features(x)

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
        needs_halves = self.readout is not None or self.side_pooling
        half_features = flat_images.new_zeros((b * p * s, 2, self.feature_dim)) if needs_halves else None
        if indices.numel() > 0:
            selected = flat_images.index_select(0, indices)
            if half_features is not None:
                encoded, halves = self.encode_with_halves(selected)
                half_features = half_features.index_copy(0, indices, halves.to(half_features.dtype))
            else:
                encoded = self.encode(selected)
            # index_copy keeps this differentiable and leaves padded rows at exact zero.
            features = features.index_copy(0, indices, encoded.to(features.dtype))
        features = features.view(b, p, s, self.feature_dim).float()
        tokens = half_features.view(b, p, s, 2, self.feature_dim).float() if half_features is not None else None
        zones = depth_zones(valid, self.n_depth_zones) if needs_halves else None

        if self.side_pooling:
            assert tokens is not None and zones is not None
            flat_slots = self.side_pooled(features, tokens, valid, zones)
        else:
            mean_features = masked_mean(features, valid)
            max_features = masked_max(features, valid)
            slot_features = torch.cat([mean_features, max_features], dim=-1)  # [B, P, 2*1280]
            flat_slots = slot_features.reshape(b, p * 2 * self.feature_dim)
        presence = series_present_mask.to(flat_slots.dtype)
        head_input = torch.cat([flat_slots, presence], dim=-1)
        logits = self.head(head_input)
        if self.readout is not None:
            assert tokens is not None and zones is not None
            logits = logits + self.readout(tokens, valid, zones)
        return logits

    def side_pooled(
        self, features: torch.Tensor, halves: torch.Tensor, valid: torch.Tensor, zones: torch.Tensor
    ) -> torch.Tensor:
        """`[B, P, S, F]`, `[B, P, S, 2, F]` -> `[B, sum(parts) * 2 * F]`: mean and max per side of each slot.

        Per slot in order: [mean part 0, .., mean part k, max part 0, .., max part k]; a part
        is a depth zone (sagittal) or a column half (coronal/axial). An empty part (a missing
        slot, or a zone without windows in a very short series) is exactly zero.
        """
        b = features.shape[0]
        out = []
        for slot, mode in enumerate(self.slot_sides):
            slot_valid = valid[:, slot]  # [B, S]
            if mode == "zones":
                parts = [
                    (features[:, slot], slot_valid & (zones[:, slot] == zone)) for zone in range(self.n_depth_zones)
                ]
            else:
                parts = [(halves[:, slot, :, half], slot_valid) for half in range(2)]
            # masked_mean / masked_max take [B, P, S, F] and [B, P, S]; one "slot" per part here.
            stacked = torch.stack([f for f, _ in parts], dim=1)
            masks = torch.stack([m for _, m in parts], dim=1)
            out.append(masked_mean(stacked, masks).reshape(b, -1))
            out.append(masked_max(stacked, masks).reshape(b, -1))
        return torch.cat(out, dim=-1)

    # -- Convenience ------------------------------------------------------------------

    def parameter_groups(self, encoder_lr: float, head_lr: float, weight_decay: float) -> list[dict]:
        """[encoder, head] AdamW groups with every trainable parameter exactly once.

        Frozen encoder parameters are left out. The encoder group stays first even when it
        is empty (encoder_trainable=frozen), so param_groups[0]/[1] keep meaning encoder/head.
        """
        encoder = [p for p in self.features.parameters() if p.requires_grad]
        # The attention scorer and readout are new, like the head: same (higher) learning rate.
        head_modules = [self.head] + [m for m in (self.attention_pool, self.readout) if m is not None]
        head = [p for m in head_modules for p in m.parameters() if p.requires_grad]
        grouped = {id(p) for p in encoder + head}
        if len(grouped) != len(encoder) + len(head):
            raise RuntimeError("a parameter is in both the encoder and the head group")
        ungrouped = [n for n, p in self.named_parameters() if p.requires_grad and id(p) not in grouped]
        if ungrouped:
            raise RuntimeError(f"trainable parameters missing from the optimizer groups: {ungrouped[:5]}")
        return [
            {"params": encoder, "lr": float(encoder_lr), "weight_decay": float(weight_decay)},
            {"params": head, "lr": float(head_lr), "weight_decay": float(weight_decay)},
        ]

    def describe(self) -> dict:
        return {
            "architecture": f"{self.backbone}_2p5d_mil",
            "backbone": self.backbone,
            "n_slots": self.n_slots,
            "n_targets": self.n_targets,
            "spatial_pool": self.spatial_pool,
            "target_attention": self.target_attention,
            "n_depth_zones": self.n_depth_zones if (self.target_attention or self.side_pooling) else None,
            "side_pooling": self.side_pooling,
            "slot_sides": list(self.slot_sides) if self.side_pooling else None,
            "feature_dim": self.feature_dim,
            "head_in": self.head_in,
            "freeze_bn_running_stats": self.freeze_bn_running_stats,
            "encoder_chunk_size": self.encoder_chunk_size,
            "grad_checkpointing": self.grad_checkpointing,
            **self.features.describe(),
            "weights_info": self.weights_info,
            "n_parameters": int(sum(p.numel() for p in self.parameters())),
            "n_trainable": int(sum(p.numel() for p in self.parameters() if p.requires_grad)),
        }


# The original name; selftests and older notebooks import it.
EfficientNetB0MIL = EfficientNetMIL


def build_model(cfg, n_slots: int | None = None) -> EfficientNetMIL:
    slots = n_slots if n_slots is not None else len(cfg.data.series_slots)
    model = EfficientNetMIL(
        n_targets=N_TARGETS,
        n_slots=slots,
        weights=str(cfg.model.weights),
        head_hidden=int(cfg.model.head_hidden),
        head_dropout=float(cfg.model.head_dropout),
        freeze_bn_running_stats=bool(cfg.model.freeze_bn_running_stats),
        encoder_chunk_size=int(cfg.model.encoder_chunk_size),
        spatial_pool=str(cfg.model.get("spatial_pool", "avg")),
        backbone=str(cfg.model.get("backbone", DEFAULT_BACKBONE)),
        grad_checkpointing=bool(cfg.model.get("grad_checkpointing", False)),
        target_attention=bool(cfg.model.get("target_attention", False)),
        attention_dim=int(cfg.model.get("attention_dim", 256)),
        n_depth_zones=int(cfg.model.get("depth_zones", 3)),
        side_pooling=bool(cfg.model.get("side_pooling", False)),
        slot_names=list(cfg.data.series_slots)[:slots] if len(cfg.data.series_slots) >= slots else None,
        image_size=int(cfg.data.image_size),
        encoder_trainable=str(cfg.model.get("encoder_trainable", "all")),
        encoder_trainable_units=int(cfg.model.get("encoder_trainable_units", 0) or 0),
    )
    LOG.info("Model: %s", model.describe())
    return model
