"""DICOM geometry: slice normals, geometric ordering, plane assignment, canonical
in-plane array orientation.

Patient coordinates are LPS: +x = left, +y = posterior, +z = superior.
Nothing in this module resamples data. Plane assignment and the canonical in-plane
orientation are decided from `ImageOrientationPatient`/`ImagePositionPatient`;
filenames and `InstanceNumber` are never the primary slice order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AXES = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}

# Slice-normal axis -> plane name.
NORMAL_AXIS_TO_PLANE = {"x": "sagittal", "y": "coronal", "z": "axial"}

# Canonical in-plane display convention per plane, as (row_axis, row_sign, col_axis, col_sign):
# the direction in patient space that the array's increasing row / column index should follow.
CANONICAL_INPLANE = {
    "sagittal": ("z", -1.0, "y", -1.0),  # rows -> inferior, columns -> anterior
    "coronal": ("z", -1.0, "x", +1.0),  # rows -> inferior, columns -> patient left
    "axial": ("y", +1.0, "x", +1.0),  # rows -> posterior, columns -> patient left
}


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8 or not np.isfinite(norm):
        raise ValueError(f"Cannot normalise a degenerate vector: {vector}")
    return vector / norm


def slice_normal(iop: np.ndarray) -> np.ndarray:
    """Slice normal from ImageOrientationPatient = [row_dir(3), col_dir(3)]."""
    iop = np.asarray(iop, dtype=np.float64).reshape(6)
    return unit(np.cross(unit(iop[0:3]), unit(iop[3:6])))


def orientations_compatible(iops: list[np.ndarray], tol_deg: float = 5.0) -> tuple[bool, float]:
    """True when every ImageOrientationPatient points the same way within `tol_deg`."""
    normals = [slice_normal(iop) for iop in iops]
    reference = normals[0]
    angles = [np.degrees(np.arccos(np.clip(abs(float(np.dot(reference, n))), -1.0, 1.0))) for n in normals]
    max_angle = float(max(angles)) if angles else 0.0
    return max_angle <= tol_deg, max_angle


def common_normal(iops: list[np.ndarray]) -> np.ndarray:
    """A single normal for a stack: the mean of sign-aligned per-slice normals."""
    normals = [slice_normal(iop) for iop in iops]
    reference = normals[0]
    aligned = [n if float(np.dot(reference, n)) >= 0 else -n for n in normals]
    return unit(np.mean(np.stack(aligned, axis=0), axis=0))


@dataclass
class PlaneAssignment:
    plane: str
    angle_deg: float  # angle between the normal and the closest principal axis
    ambiguous: bool
    runner_up: str


def assign_plane(normal: np.ndarray, tolerance_deg: float = 35.0) -> PlaneAssignment:
    """Assign a plane from the slice normal with an explicit obliquity tolerance."""
    normal = unit(normal)
    dots = {axis: abs(float(np.dot(normal, vec))) for axis, vec in AXES.items()}
    ordered = sorted(dots.items(), key=lambda kv: (-kv[1], kv[0]))
    best_axis, best_dot = ordered[0]
    runner_axis, runner_dot = ordered[1]
    angle = float(np.degrees(np.arccos(np.clip(best_dot, -1.0, 1.0))))
    ambiguous = angle > tolerance_deg or (best_dot - runner_dot) < 0.15
    return PlaneAssignment(
        plane=NORMAL_AXIS_TO_PLANE[best_axis],
        angle_deg=angle,
        ambiguous=bool(ambiguous),
        runner_up=NORMAL_AXIS_TO_PLANE[runner_axis],
    )


@dataclass
class StackOrder:
    order: np.ndarray  # indices that sort the slices along the normal
    projections: np.ndarray  # projection of each ImagePositionPatient onto the normal, sorted
    spacings: np.ndarray  # consecutive centre-to-centre distances, sorted order
    median_spacing: float
    duplicate_positions: int
    irregular: bool
    max_gap_ratio: float


def order_by_position(
    positions: np.ndarray,
    normal: np.ndarray,
    duplicate_tol_mm: float = 1e-3,
    irregular_tol: float = 0.2,
) -> StackOrder:
    """Sort slices geometrically along the slice normal and describe the spacing.

    `SliceThickness` is never substituted for centre-to-centre spacing: the spacing
    reported here is derived from the projected positions only.
    """
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    normal = unit(normal)
    raw = positions @ normal
    order = np.argsort(raw, kind="stable")
    projections = raw[order]
    spacings = np.diff(projections) if len(projections) > 1 else np.zeros(0)
    duplicates = int(np.sum(np.abs(spacings) <= duplicate_tol_mm)) if spacings.size else 0
    positive = spacings[spacings > duplicate_tol_mm]
    median = float(np.median(positive)) if positive.size else 0.0
    if positive.size and median > 0:
        deviation = float(np.max(np.abs(positive - median)) / median)
        max_ratio = float(np.max(positive) / median)
    else:
        deviation, max_ratio = 0.0, 1.0
    return StackOrder(
        order=order,
        projections=projections,
        spacings=spacings,
        median_spacing=median,
        duplicate_positions=duplicates,
        irregular=bool(deviation > irregular_tol),
        max_gap_ratio=max_ratio,
    )


@dataclass
class InPlaneTransform:
    """A pure array reordering (transpose/flips) - never a physical resampling."""

    transpose: bool
    flip_rows: bool
    flip_cols: bool
    row_axis: str
    col_axis: str
    oblique: bool

    def apply(self, array: np.ndarray) -> np.ndarray:
        """Apply to an array whose last two dimensions are (rows, columns)."""
        out = array
        if self.transpose:
            out = np.swapaxes(out, -1, -2)
        if self.flip_rows:
            out = np.flip(out, axis=-2)
        if self.flip_cols:
            out = np.flip(out, axis=-1)
        return np.ascontiguousarray(out)

    def apply_spacing(self, row_spacing: float, col_spacing: float) -> tuple[float, float]:
        """Row/column spacing follow the transpose; flips do not change them."""
        return (col_spacing, row_spacing) if self.transpose else (row_spacing, col_spacing)

    def describe(self) -> str:
        ops = []
        if self.transpose:
            ops.append("transpose")
        if self.flip_rows:
            ops.append("flip_rows")
        if self.flip_cols:
            ops.append("flip_cols")
        return "+".join(ops) if ops else "identity"


def canonical_inplane_transform(iop: np.ndarray, plane: str, oblique_tol_deg: float = 20.0) -> InPlaneTransform:
    """Bring a series into a consistent series-local in-plane orientation.

    Only axis swaps and flips are used, so no interpolation happens and oblique data
    are not silently straightened; `oblique` records that the in-plane axes are not
    close to the principal patient axes.
    """
    iop = np.asarray(iop, dtype=np.float64).reshape(6)
    row_dir, col_dir = unit(iop[0:3]), unit(iop[3:6])
    # DICOM convention: iop[0:3] is the direction of increasing *column* index,
    # iop[3:6] the direction of increasing *row* index.
    col_vec, row_vec = row_dir, col_dir

    def dominant(vec: np.ndarray) -> tuple[str, float, float]:
        dots = {axis: float(np.dot(vec, ax)) for axis, ax in AXES.items()}
        axis = max(dots, key=lambda a: abs(dots[a]))
        angle = float(np.degrees(np.arccos(np.clip(abs(dots[axis]), -1.0, 1.0))))
        return axis, float(np.sign(dots[axis]) or 1.0), angle

    row_axis, row_sign, row_angle = dominant(row_vec)
    col_axis, col_sign, col_angle = dominant(col_vec)

    want_row_axis, want_row_sign, want_col_axis, want_col_sign = CANONICAL_INPLANE[plane]
    transpose = row_axis != want_row_axis and col_axis == want_row_axis
    if transpose:
        row_axis, col_axis = col_axis, row_axis
        row_sign, col_sign = col_sign, row_sign

    flip_rows = row_axis == want_row_axis and row_sign != want_row_sign
    flip_cols = col_axis == want_col_axis and col_sign != want_col_sign

    return InPlaneTransform(
        transpose=bool(transpose),
        flip_rows=bool(flip_rows),
        flip_cols=bool(flip_cols),
        row_axis=row_axis,
        col_axis=col_axis,
        oblique=bool(max(row_angle, col_angle) > oblique_tol_deg),
    )
