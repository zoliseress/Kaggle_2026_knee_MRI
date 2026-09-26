"""Which knee (right/left) each study shows, and how to bring every study into one
medial/lateral frame.

The canonical in-plane orientation (`geometry.CANONICAL_INPLANE`) follows patient
coordinates, so the medial compartment of a right knee sits on the opposite image side
from a left knee's, and the sagittal slice order runs medial->lateral for one side and
lateral->medial for the other. Pooling cannot undo that; a canonical frame can.

Side of the knee, in this order:
  1. the DICOM `Laterality` / `ImageLaterality` tag (R/L; ~50% of the training studies);
  2. otherwise geometry: the median x of the coronal/axial image centres in LPS patient
     coordinates. The knee lies off the midline, x < 0 is the patient's right. Within
     `min_offset_mm` of the midline the side stays unresolved. On the tagged training
     studies this rule agrees with the tag in 97.4%.
  3. unresolved studies are left as they are (no flip).

Canonical frame: "lateral" is towards increasing column index (coronal, axial) and
towards increasing slice index (sagittal). Coronal/axial columns point to the patient's
left, which is lateral for a left knee, so right knees are mirrored. Sagittal slices are
sorted along the stored slice normal; the order is reversed when that normal does not
point laterally.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .constants import STUDY_ID
from .utils import LOG, atomic_write_dataframe, atomic_write_json

LATERALITY_COLUMNS = [
    STUDY_ID, "side", "side_source", "tag_side", "geometry_side", "center_x_mm", "sagittal_normal_x",
]
# Planes whose image centre locates the knee left/right of the midline. A sagittal image
# centre is one slice's x, not the knee's, so it is not used.
GEOMETRY_PLANES = ("coronal", "axial")


def laterality_path(cfg: Config) -> Path:
    configured = cfg.paths.get("laterality_csv")
    return Path(configured) if configured else Path(cfg.paths.work_dir) / "manifest" / "laterality.csv"


def side_from_tag(values) -> str:
    """The first R/L found in the tag values ('R', 'RIGHT', 'L', 'LEFT'); '' when none."""
    for value in values:
        text = "" if value is None or pd.isna(value) else str(value).strip().upper()
        if text in ("R", "RIGHT"):
            return "R"
        if text in ("L", "LEFT"):
            return "L"
    return ""


def image_center_x(ipp, iop, pixel_spacing, rows: int, cols: int) -> float:
    """x (LPS, mm) of the image centre: IPP + row_dir * cols*col_spacing/2 + col_dir * rows*row_spacing/2."""
    ipp = np.asarray(ipp, dtype=np.float64)
    iop = np.asarray(iop, dtype=np.float64)
    row_spacing, col_spacing = float(pixel_spacing[0]), float(pixel_spacing[1])
    return float(ipp[0] + iop[0] * cols * col_spacing / 2.0 + iop[3] * rows * row_spacing / 2.0)


def side_from_center_x(center_x: float, min_offset_mm: float) -> str:
    if center_x is None or not np.isfinite(center_x) or abs(center_x) < min_offset_mm:
        return ""
    return "R" if center_x < 0 else "L"


def _series_center_x(series_dir: str) -> float:
    """Image-centre x of the first DICOM file in a series directory (header only)."""
    import pydicom

    directory = Path(series_dir)
    files = sorted(p for p in directory.iterdir() if p.is_file())
    for path in files:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            return image_center_x(
                ds.ImagePositionPatient, ds.ImageOrientationPatient, ds.PixelSpacing, int(ds.Rows), int(ds.Columns)
            )
        except Exception:  # a broken file: try the next one
            continue
    return float("nan")


def derive_laterality(
    manifest: pd.DataFrame,
    selection: pd.DataFrame,
    min_offset_mm: float = 20.0,
    workers: int = 16,
    center_x_reader=_series_center_x,
) -> pd.DataFrame:
    """One row per study: side, where it came from, and the sagittal slice-normal x."""
    manifest = manifest.copy()
    manifest[STUDY_ID] = manifest[STUDY_ID].astype(str)
    selected = selection[selection["selected"].fillna(False).astype(bool)].copy()
    selected[STUDY_ID] = selected[STUDY_ID].astype(str)

    tags = manifest.groupby(STUDY_ID)["laterality"].agg(lambda s: side_from_tag(s.tolist())) if "laterality" in manifest else {}

    geometry_rows = selected[selected["slot"].isin(GEOMETRY_PLANES)]
    paths = list(geometry_rows["path"].astype(str))
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        centers = list(pool.map(center_x_reader, paths))
    geometry_rows = geometry_rows.assign(center_x_mm=centers)
    center_by_study = geometry_rows.groupby(STUDY_ID)["center_x_mm"].median()

    normal_x = pd.Series(dtype=float)
    if "normal_x" in manifest.columns:
        sagittal = selected[selected["slot"] == "sagittal"][[STUDY_ID, "volume_id"]]
        merged = sagittal.merge(manifest[[STUDY_ID, "volume_id", "normal_x"]], on=[STUDY_ID, "volume_id"], how="left")
        normal_x = merged.set_index(STUDY_ID)["normal_x"]

    rows = []
    for study in sorted(set(selected[STUDY_ID])):
        tag_side = str(tags.get(study, "")) if len(tags) else ""
        center_x = float(center_by_study.get(study, np.nan))
        geometry_side = side_from_center_x(center_x, min_offset_mm)
        if tag_side:
            side, source = tag_side, "tag"
        elif geometry_side:
            side, source = geometry_side, "geometry"
        else:
            side, source = "", "unresolved"
        rows.append(
            {
                STUDY_ID: study,
                "side": side,
                "side_source": source,
                "tag_side": tag_side,
                "geometry_side": geometry_side,
                "center_x_mm": round(center_x, 2) if np.isfinite(center_x) else np.nan,
                "sagittal_normal_x": float(normal_x.get(study, np.nan)),
            }
        )
    return pd.DataFrame(rows, columns=LATERALITY_COLUMNS)


def summarize_laterality(table: pd.DataFrame) -> dict:
    both = table[(table["tag_side"] != "") & (table["geometry_side"] != "")]
    return {
        "studies": int(len(table)),
        "by_source": table["side_source"].value_counts().to_dict(),
        "by_side": table["side"].replace("", "unresolved").value_counts().to_dict(),
        "tag_and_geometry": int(len(both)),
        "tag_geometry_agreement": float((both["tag_side"] == both["geometry_side"]).mean()) if len(both) else None,
    }


def build_laterality(
    cfg: Config,
    manifest: pd.DataFrame,
    selection: pd.DataFrame,
    out_path: str | Path | None = None,
    workers: int = 16,
) -> pd.DataFrame:
    table = derive_laterality(
        manifest, selection, min_offset_mm=float(cfg.data.get("laterality_min_offset_mm", 20.0)), workers=workers
    )
    out_path = Path(out_path) if out_path else laterality_path(cfg)
    atomic_write_dataframe(table, out_path)
    summary = summarize_laterality(table)
    atomic_write_json(out_path.with_name(out_path.stem + "_summary.json"), summary)
    LOG.info("Laterality written: %s | %s", out_path, summary)
    return table


def load_laterality(path: str | Path) -> dict[str, tuple[str, float]]:
    """StudyInstanceUID -> (side 'R'/'L'/'' , sagittal normal x)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Laterality table not found: {path}. Run `python -m knee_mri.cli build-laterality` "
            "(needs the manifest and the series selection)."
        )
    frame = pd.read_csv(path, dtype={STUDY_ID: "string", "side": "string"}, keep_default_na=False, na_values=[""])
    sides = frame["side"].fillna("").astype(str)
    normals = pd.to_numeric(frame["sagittal_normal_x"], errors="coerce")
    return {str(s): (side, float(n)) for s, side, n in zip(frame[STUDY_ID], sides, normals)}


def canonical_ops(side: str, plane: str, sagittal_normal_x: float = float("nan")) -> tuple[bool, bool]:
    """(reverse_slices, flip_columns) that put `plane` of a `side` knee into the canonical frame."""
    if side not in ("R", "L"):
        return False, False
    if plane in ("coronal", "axial"):
        return False, side == "R"  # columns -> patient left = lateral only for a left knee
    if plane == "sagittal":
        if not np.isfinite(sagittal_normal_x) or sagittal_normal_x == 0:
            return False, False
        lateral_sign = 1.0 if side == "L" else -1.0  # lateral is +x (patient left) for a left knee
        return bool(np.sign(sagittal_normal_x) != lateral_sign), False
    return False, False


def apply_canonical(image: np.ndarray, gap_ok: np.ndarray | None, reverse_slices: bool, flip_columns: bool):
    """Reorder a cached `[Z, H, W]` stack; `gap_ok[i]` (gap between slices i and i+1) follows the reversal."""
    if reverse_slices:
        image = image[::-1]
        if gap_ok is not None:
            gap_ok = gap_ok[::-1]
    if flip_columns:
        image = image[..., ::-1]
    if reverse_slices or flip_columns:
        image = np.ascontiguousarray(image)
        gap_ok = np.ascontiguousarray(gap_ok) if gap_ok is not None else None
    return image, gap_ok
