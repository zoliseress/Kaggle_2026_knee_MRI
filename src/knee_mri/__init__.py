"""RSNA knee MRI: study-level multi-label classification with a 2.5D MIL model
(EfficientNet-B0 by default; EfficientNet-V2-S, RadImageNet ResNet-50 or DINOv2 ViT-S/14 encoders).

Modules
-------
    config      YAML configuration, CLI overrides, validation
    schema      input schema and coverage checks
    labels      extraction statuses -> (target, weight) contract
    geometry    DICOM geometry: normals, ordering, plane, canonical in-plane orientation
    dicom_io    header inventory, acquisition grouping, decoding
    manifest    volume inventory and per-slot series selection
    preprocess  deterministic preprocessing and the full-series cache
    splits      immutable, group-disjoint folds
    dataset     2.5D bags, masks and coherent augmentation
    encoders    encoder adapters: [N, 3, H, W] -> [N, C, Hf, Wf], weights, freezing
    model       encoder adapter + masked mean/max MIL pooling
    loss        masked, class-normalised BCE
    metrics     fixed-reference validation metrics
    train       training loop and run modes
    evaluate    checkpoint evaluation, prediction export, OOF merge
    qc          QC gallery
    report      plots and the offline HTML report
    selftest    focused failure-mode checks
    cli         command line entry points
"""

from .constants import PLANES, TARGETS  # noqa: F401

__version__ = "0.1.0"
