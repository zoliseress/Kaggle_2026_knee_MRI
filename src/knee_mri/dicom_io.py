"""DICOM reading: header inventory, acquisition grouping, geometric ordering, decoding.

Design rules enforced here:
  * slices are ordered by projected `ImagePositionPatient`, never by filename or
    `InstanceNumber`;
  * different echoes / time points / orientations inside one SeriesInstanceUID become
    separate candidate volumes instead of being mixed or blindly de-duplicated;
  * a multiframe object is expanded with its real per-frame geometry, or explicitly
    flagged as unsupported - it is never silently treated as a single slice;
  * a missing decoder dependency fails loudly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .geometry import (
    InPlaneTransform,
    PlaneAssignment,
    StackOrder,
    assign_plane,
    canonical_inplane_transform,
    common_normal,
    order_by_position,
    orientations_compatible,
)
from .utils import LOG

try:  # pydicom >= 3
    from pydicom.pixels import apply_modality_lut as _apply_modality_lut
except ImportError:  # pragma: no cover - pydicom 2.x
    try:
        from pydicom.pixel_data_handlers.util import apply_modality_lut as _apply_modality_lut
    except ImportError:  # pragma: no cover
        _apply_modality_lut = None  # type: ignore[assignment]


DICOM_SUFFIXES = {".dcm", ".dicom", ""}


@dataclass
class SliceRef:
    """One 2D image: a single-frame file, or one frame of a multiframe object."""

    path: str
    frame_index: int | None
    ipp: tuple[float, float, float]
    iop: tuple[float, float, float, float, float, float]
    pixel_spacing: tuple[float, float]  # (row spacing, column spacing) in mm
    rows: int
    cols: int
    instance_number: int | None
    group_key: tuple

    @property
    def sort_hint(self) -> tuple:
        return (self.path, self.frame_index if self.frame_index is not None else -1)


@dataclass
class SeriesHeader:
    """Series-level metadata shared by the candidate volumes of one series directory."""

    study_uid: str
    series_uid: str
    series_description: str = ""
    scanning_sequence: str = ""
    sequence_variant: str = ""
    scan_options: str = ""
    mr_acquisition_type: str = ""
    echo_time: float | None = None
    repetition_time: float | None = None
    inversion_time: float | None = None
    magnetic_field_strength: float | None = None
    patient_id: str = ""
    laterality: str = ""
    body_part: str = ""
    photometric: str = "MONOCHROME2"
    transfer_syntax: str = ""
    modality: str = ""
    n_files: int = 0


@dataclass
class VolumeCandidate:
    """A geometrically coherent stack: the unit that series selection ranks."""

    study_uid: str
    series_uid: str
    volume_key: str
    slices: list[SliceRef]
    header: SeriesHeader
    plane: PlaneAssignment
    order: StackOrder
    inplane: InPlaneTransform
    normal: np.ndarray
    flags: list[str] = field(default_factory=list)

    @property
    def volume_id(self) -> str:
        return f"{self.series_uid}#{self.volume_key}"

    @property
    def n_slices(self) -> int:
        return len(self.slices)

    @property
    def row_spacing(self) -> float:
        return float(self.slices[0].pixel_spacing[0])

    @property
    def col_spacing(self) -> float:
        return float(self.slices[0].pixel_spacing[1])

    def usable(self, min_slices: int) -> bool:
        blocking = {"decode_failed", "no_geometry", "multiframe_unsupported", "inconsistent_orientation", "empty"}
        return self.n_slices >= min_slices and not (blocking & set(self.flags))


def _float_tuple(value: Any, length: int, default: tuple | None = None) -> tuple | None:
    if value is None:
        return default
    try:
        seq = [float(v) for v in value]
    except (TypeError, ValueError):
        return default
    if len(seq) != length:
        return default
    return tuple(seq)


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\\".join(str(v) for v in value)
    return str(value).strip()


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def iter_dicom_files(series_dir: Path) -> list[Path]:
    files = [p for p in sorted(series_dir.iterdir()) if p.is_file() and p.suffix.lower() in DICOM_SUFFIXES]
    return files


def _shared_and_perframe(ds: Any) -> tuple[Any, Any]:
    return getattr(ds, "SharedFunctionalGroupsSequence", None), getattr(ds, "PerFrameFunctionalGroupsSequence", None)


def _functional_group_value(shared: Any, per_frame_item: Any, sequence_name: str, tag: str) -> Any:
    for source in (per_frame_item, shared[0] if shared else None):
        if source is None:
            continue
        seq = getattr(source, sequence_name, None)
        if seq:
            value = getattr(seq[0], tag, None)
            if value is not None:
                return value
    return None


def _expand_multiframe(ds: Any, path: Path, n_frames: int) -> tuple[list[SliceRef], list[str]]:
    """Expand an enhanced/multiframe object using its real per-frame geometry."""
    flags: list[str] = []
    shared, per_frame = _shared_and_perframe(ds)
    if not per_frame or len(per_frame) != n_frames:
        return [], ["multiframe_unsupported"]

    rows, cols = int(getattr(ds, "Rows", 0)), int(getattr(ds, "Columns", 0))
    refs: list[SliceRef] = []
    for frame_index, item in enumerate(per_frame):
        ipp = _float_tuple(_functional_group_value(shared, item, "PlanePositionSequence", "ImagePositionPatient"), 3)
        iop = _float_tuple(
            _functional_group_value(shared, item, "PlaneOrientationSequence", "ImageOrientationPatient"), 6
        )
        spacing = _float_tuple(
            _functional_group_value(shared, item, "PixelMeasuresSequence", "PixelSpacing"), 2, (1.0, 1.0)
        )
        if ipp is None or iop is None:
            return [], ["multiframe_unsupported"]
        echo = _functional_group_value(shared, item, "MREchoSequence", "EffectiveEchoTime")
        stack_id = _functional_group_value(shared, item, "FrameContentSequence", "StackID")
        temporal = _functional_group_value(shared, item, "FrameContentSequence", "TemporalPositionIndex")
        group_key = (
            tuple(round(v, 3) for v in iop),
            rows,
            cols,
            round(float(echo), 2) if echo is not None else None,
            _as_str(stack_id),
            int(temporal) if temporal is not None else None,
        )
        refs.append(
            SliceRef(
                path=str(path),
                frame_index=frame_index,
                ipp=ipp,
                iop=iop,
                pixel_spacing=(float(spacing[0]), float(spacing[1])),
                rows=rows,
                cols=cols,
                instance_number=frame_index,
                group_key=group_key,
            )
        )
    return refs, flags


def read_series_slices(series_dir: Path) -> tuple[list[SliceRef], SeriesHeader, list[str]]:
    """Read every header in a series directory and return the slice references."""
    import pydicom

    files = iter_dicom_files(series_dir)
    header = SeriesHeader(
        study_uid=series_dir.parent.name,
        series_uid=series_dir.name,
        n_files=len(files),
    )
    if not files:
        return [], header, ["empty"]

    refs: list[SliceRef] = []
    flags: set[str] = set()
    first = True
    for path in files:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
        except Exception as exc:
            LOG.debug("Header read failed for %s: %s", path, exc)
            flags.add("header_read_failed")
            continue
        if first:
            header.series_description = _as_str(getattr(ds, "SeriesDescription", ""))
            header.scanning_sequence = _as_str(getattr(ds, "ScanningSequence", ""))
            header.sequence_variant = _as_str(getattr(ds, "SequenceVariant", ""))
            header.scan_options = _as_str(getattr(ds, "ScanOptions", ""))
            header.mr_acquisition_type = _as_str(getattr(ds, "MRAcquisitionType", ""))
            header.echo_time = _as_float(getattr(ds, "EchoTime", None))
            header.repetition_time = _as_float(getattr(ds, "RepetitionTime", None))
            header.inversion_time = _as_float(getattr(ds, "InversionTime", None))
            header.magnetic_field_strength = _as_float(getattr(ds, "MagneticFieldStrength", None))
            header.patient_id = _as_str(getattr(ds, "PatientID", ""))
            header.laterality = _as_str(getattr(ds, "Laterality", "")) or _as_str(
                getattr(ds, "ImageLaterality", "")
            )
            header.body_part = _as_str(getattr(ds, "BodyPartExamined", ""))
            header.photometric = _as_str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
            header.modality = _as_str(getattr(ds, "Modality", ""))
            meta = getattr(ds, "file_meta", None)
            header.transfer_syntax = _as_str(getattr(meta, "TransferSyntaxUID", "")) if meta else ""
            first = False

        n_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
        if n_frames > 1:
            frame_refs, frame_flags = _expand_multiframe(ds, path, n_frames)
            flags.update(frame_flags)
            if frame_refs:
                refs.extend(frame_refs)
            else:
                LOG.warning(
                    "Multiframe object without usable per-frame geometry: %s (%d frames). "
                    "It is flagged, not counted as one slice.",
                    path,
                    n_frames,
                )
            continue

        ipp = _float_tuple(getattr(ds, "ImagePositionPatient", None), 3)
        iop = _float_tuple(getattr(ds, "ImageOrientationPatient", None), 6)
        spacing = _float_tuple(getattr(ds, "PixelSpacing", None), 2)
        if ipp is None or iop is None:
            flags.add("no_geometry")
            continue
        if spacing is None:
            spacing = (1.0, 1.0)
            flags.add("missing_pixel_spacing")
        rows, cols = int(getattr(ds, "Rows", 0)), int(getattr(ds, "Columns", 0))
        echo_number = getattr(ds, "EchoNumbers", None)
        echo_time = _as_float(getattr(ds, "EchoTime", None))
        temporal = getattr(ds, "TemporalPositionIdentifier", None)
        image_type = getattr(ds, "ImageType", None)
        value_kind = _as_str(image_type[2]) if image_type is not None and len(image_type) > 2 else ""
        group_key = (
            tuple(round(v, 3) for v in iop),
            rows,
            cols,
            round(echo_time, 2) if echo_time is not None else None,
            int(echo_number) if echo_number is not None else None,
            int(temporal) if temporal is not None else None,
            value_kind,
        )
        refs.append(
            SliceRef(
                path=str(path),
                frame_index=None,
                ipp=ipp,
                iop=iop,
                pixel_spacing=(float(spacing[0]), float(spacing[1])),
                rows=rows,
                cols=cols,
                instance_number=int(getattr(ds, "InstanceNumber", 0) or 0),
                group_key=group_key,
            )
        )
    return refs, header, sorted(flags)


def group_slices(refs: Iterable[SliceRef]) -> dict[str, list[SliceRef]]:
    """Split a series into compatible acquisition groups (echo / time point / geometry)."""
    groups: dict[tuple, list[SliceRef]] = {}
    for ref in refs:
        groups.setdefault(ref.group_key, []).append(ref)
    # Stable, human-readable keys: order groups by (size desc, key repr) and index them.
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), repr(kv[0])))
    return {f"g{i}": sorted(items, key=lambda r: r.sort_hint) for i, (_, items) in enumerate(ordered)}


def build_volume_candidates(
    series_dir: Path,
    obliquity_tolerance_deg: float = 35.0,
    max_gap_ratio: float = 1.75,
    min_slices: int = 4,
) -> tuple[list[VolumeCandidate], SeriesHeader, list[str]]:
    """Turn one series directory into geometrically ordered volume candidates."""
    refs, header, series_flags = read_series_slices(series_dir)
    if not refs:
        return [], header, series_flags or ["empty"]

    candidates: list[VolumeCandidate] = []
    for key, group in group_slices(refs).items():
        flags = list(series_flags)
        iops = [np.asarray(r.iop) for r in group]
        compatible, max_angle = orientations_compatible(iops)
        if not compatible:
            flags.append("inconsistent_orientation")
        normal = common_normal(iops)
        plane = assign_plane(normal, tolerance_deg=obliquity_tolerance_deg)
        if plane.ambiguous:
            flags.append("ambiguous_plane")
            LOG.debug(
                "Ambiguous plane for %s/%s: %.1f deg off %s (runner-up %s)",
                header.series_uid,
                key,
                plane.angle_deg,
                plane.plane,
                plane.runner_up,
            )
        stack = order_by_position(np.stack([np.asarray(r.ipp) for r in group]), normal)
        ordered = [group[i] for i in stack.order]
        if stack.duplicate_positions:
            flags.append("duplicate_positions")
        if stack.irregular:
            flags.append("irregular_spacing")
        if stack.max_gap_ratio > max_gap_ratio:
            flags.append("large_gap")
        if len({(r.rows, r.cols) for r in ordered}) > 1:
            flags.append("mixed_matrix_size")
        if len(ordered) < min_slices:
            flags.append("short_stack")
        inplane = canonical_inplane_transform(np.asarray(ordered[0].iop), plane.plane)
        if inplane.oblique:
            flags.append("oblique_inplane")

        candidates.append(
            VolumeCandidate(
                study_uid=header.study_uid,
                series_uid=header.series_uid,
                volume_key=key,
                slices=ordered,
                header=header,
                plane=plane,
                order=stack,
                inplane=inplane,
                normal=normal,
                flags=sorted(set(flags)),
            )
        )
    return candidates, header, series_flags


def _apply_photometric(array: np.ndarray, photometric: str, policy: str) -> np.ndarray:
    """MONOCHROME1 means high value = dark. Invert once, explicitly, after the modality LUT."""
    if photometric.upper() == "MONOCHROME1" and policy == "invert":
        finite = array[np.isfinite(array)]
        if finite.size:
            return float(finite.max()) + float(finite.min()) - array
    return array


def decode_slice(ref: SliceRef, photometric: str, monochrome1_policy: str = "invert") -> np.ndarray:
    """Decode one slice to float32 in stored physical units, with padding set to NaN."""
    import pydicom

    if _apply_modality_lut is None:  # pragma: no cover - depends on pydicom version
        raise ImportError(
            "pydicom does not expose apply_modality_lut. Install pydicom>=2.3 "
            "(and a pixel-data decoder such as pylibjpeg/gdcm for compressed transfer syntaxes)."
        )
    ds = pydicom.dcmread(ref.path, force=False)
    raw = ds.pixel_array
    if ref.frame_index is not None:
        raw = raw[ref.frame_index]
    array = np.asarray(_apply_modality_lut(raw, ds), dtype=np.float32)

    padding = getattr(ds, "PixelPaddingValue", None)
    if padding is not None:
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        padded_value = float(padding) * slope + intercept
        array = np.where(np.isclose(array, padded_value), np.nan, array)

    array = _apply_photometric(array, photometric or _as_str(getattr(ds, "PhotometricInterpretation", "")), monochrome1_policy)
    return array


def decode_volume(volume: VolumeCandidate, monochrome1_policy: str = "invert") -> np.ndarray:
    """Decode a whole volume to float32 `[Z, H, W]` in geometric slice order.

    Padding pixels are NaN so that intensity statistics can exclude them; the caller
    replaces them after normalisation.
    """
    arrays = []
    target_shape: tuple[int, int] | None = None
    for ref in volume.slices:
        array = decode_slice(ref, volume.header.photometric, monochrome1_policy)
        if array.ndim != 2:
            raise ValueError(f"Expected a 2D frame from {ref.path}, got shape {array.shape}")
        if target_shape is None:
            target_shape = array.shape
        elif array.shape != target_shape:
            raise ValueError(
                f"Mixed matrix sizes inside volume {volume.volume_id}: {array.shape} vs {target_shape}. "
                "This volume should have been split into separate acquisition groups."
            )
        arrays.append(array)
    stacked = np.stack(arrays, axis=0)
    return volume.inplane.apply(stacked)


def probe_decode(volume: VolumeCandidate, monochrome1_policy: str = "invert") -> tuple[bool, str]:
    """Try to decode the middle slice; report a clear reason on failure."""
    if not volume.slices:
        return False, "empty volume"
    ref = volume.slices[len(volume.slices) // 2]
    try:
        array = decode_slice(ref, volume.header.photometric, monochrome1_policy)
    except Exception as exc:
        return False, f"{exc.__class__.__name__}: {exc}"
    if array.size == 0:
        return False, "decoded array is empty"
    if not np.isfinite(array).any():
        return False, "decoded array has no finite value"
    return True, ""
