"""Project-wide constants that must stay stable across manifests, caches and checkpoints."""

from __future__ import annotations

# Confirmed project target order. It is persisted in every checkpoint and every
# exported prediction file; never reorder it without bumping LABELS_VERSION.
TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
N_TARGETS = len(TARGETS)

# Canonical plane names used for the fixed-order series slots.
PLANES = ["sagittal", "coronal", "axial"]

# A slot is a plane ("coronal") or a plane plus a sequence filter ("coronal_t1"). A plain
# plane slot takes the best-scoring volume of that plane; a filtered slot takes the best
# volume of that plane that passes the filter and is not already used by the plane's own
# slot (manifest.select_series). Filters:
#   t1 - non fat-suppressed spin-echo T1: 0 < TR < 1000 ms, 0 < TE < 30 ms, no GR/IR in
#        ScanningSequence, no inversion time; missing headers never pass.
SLOT_FILTERS = ("t1",)


def slot_plane(slot: str) -> str:
    """The anatomical plane of a slot name: 'coronal_t1' -> 'coronal'."""
    return str(slot).split("_", 1)[0]


def slot_filter(slot: str) -> str | None:
    """The sequence filter of a slot name: 'coronal_t1' -> 't1', 'coronal' -> None."""
    parts = str(slot).split("_", 1)
    return parts[1] if len(parts) == 2 else None


STUDY_ID = "StudyInstanceUID"
SERIES_ID = "SeriesInstanceUID"

# Versions. Bump them when the corresponding artefact format or semantics change;
# they are written into manifests/caches/checkpoints and checked on reload.
MANIFEST_VERSION = "manifest_v1"
PREPROCESS_VERSION = "prep_v1"
LABELS_VERSION = "labels_v1"
SPLITS_VERSION = "splits_v1"
CHECKPOINT_VERSION = "ckpt_v1"

# Encoder input normalisation profiles (data.encoder_normalization), applied exactly once to the
# robust-scaled [0, 1] slices - in the dataset (augment.device=cpu) or in augment_batch (cuda).
# NOTE: our three channels are adjacent MRI slices, not RGB, so the default is a
# channel-independent `mri_scalar` normalisation (mean 0.5, std 0.25 on the [0, 1]
# robust-scaled input). The pretrained ImageNet statistics remain available as `imagenet`
# (torchvision EfficientNet IMAGENET1K_V1 and timm DINOv2 lvd142m both use them).
# `radimagenet_torch` reproduces the official RadImageNet PyTorch demo, which feeds
# (uint8 - 127.5) * 2 / 255, i.e. [0, 1] -> [-1, 1]: mean 0.5, std 0.5 per channel.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MRI_SCALAR_MEAN = (0.5, 0.5, 0.5)
MRI_SCALAR_STD = (0.25, 0.25, 0.25)
RADIMAGENET_TORCH_MEAN = (0.5, 0.5, 0.5)
RADIMAGENET_TORCH_STD = (0.5, 0.5, 0.5)
NORMALIZATION_PROFILES = {
    "mri_scalar": (MRI_SCALAR_MEAN, MRI_SCALAR_STD),
    "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
    "radimagenet_torch": (RADIMAGENET_TORCH_MEAN, RADIMAGENET_TORCH_STD),
}

EFFICIENTNET_B0_FEATURES = 1280
# Encoder backbones (model.backbone) -> channels of their last feature map (encoders.py).
# The CNNs have an output stride of 32 (224 px -> 7x7, 320 px -> 10x10); DINOv2 has 14 px
# patches (224 px -> 16x16, 336 px -> 24x24).
ENCODER_FEATURES = {
    "efficientnet_b0": 1280,
    "efficientnet_v2_s": 1280,
    "radimagenet_resnet50": 2048,
    "dinov2_vits14": 384,
}
BACKBONES = tuple(ENCODER_FEATURES)
DEFAULT_BACKBONE = "efficientnet_b0"  # also what checkpoints without model.backbone were trained with
# Units the `last_n` freeze policy counts: EfficientNet feature stages, ResNet residual
# stages (layer1..layer4), DINOv2 transformer blocks.
ENCODER_UNITS = {
    "efficientnet_b0": 9,
    "efficientnet_v2_s": 8,
    "radimagenet_resnet50": 4,
    "dinov2_vits14": 12,
}
# The normalisation each backbone's pretrained weights expect. Another profile is allowed
# (with a warning) as an explicit ablation; EfficientNet keeps the project default mri_scalar.
RECOMMENDED_NORMALIZATION = {
    "radimagenet_resnet50": "radimagenet_torch",
    "dinov2_vits14": "imagenet",
}
DINOV2_PATCH_SIZE = 14

# Label status vocabulary of the report-extraction export.
STATUS_POSITIVE = "positive"
STATUS_NEGATIVE = "negative"
STATUS_UNCERTAIN = "uncertain"
STATUS_NOT_MENTIONED = "not_mentioned"
