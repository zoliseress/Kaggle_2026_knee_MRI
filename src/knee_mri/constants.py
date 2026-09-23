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

STUDY_ID = "StudyInstanceUID"
SERIES_ID = "SeriesInstanceUID"

# Versions. Bump them when the corresponding artefact format or semantics change;
# they are written into manifests/caches/checkpoints and checked on reload.
MANIFEST_VERSION = "manifest_v1"
PREPROCESS_VERSION = "prep_v1"
LABELS_VERSION = "labels_v1"
SPLITS_VERSION = "splits_v1"
CHECKPOINT_VERSION = "ckpt_v1"

# torchvision EfficientNet_B0_Weights.IMAGENET1K_V1 preprocessing statistics.
# NOTE: our three channels are adjacent MRI slices, not RGB. We still default to
# the pretrained normalisation so the encoder sees inputs in the range it was
# trained on; `mri_scalar` is offered as a documented ablation.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MRI_SCALAR_MEAN = (0.5, 0.5, 0.5)
MRI_SCALAR_STD = (0.25, 0.25, 0.25)

EFFICIENTNET_B0_FEATURES = 1280

# Label status vocabulary of the report-extraction export.
STATUS_POSITIVE = "positive"
STATUS_NEGATIVE = "negative"
STATUS_UNCERTAIN = "uncertain"
STATUS_NOT_MENTIONED = "not_mentioned"
