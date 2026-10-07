"""Generate visual, synthetic examples of each deterministic preprocessing step.

Run from ``src`` with ``python tests2\\visual_preprocessing_examples.py``.  The
assertions make this usable as a small regression test as well as a visual aid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knee_mri.geometry import canonical_inplane_transform
from knee_mri.dicom_io import build_volume_candidates, decode_volume
from knee_mri.preprocess import estimate_center, physical_crop, resample_square, robust_scale


def _synthetic_stack(depth: int = 7, height: int = 120, width: int = 160) -> np.ndarray:
    """Build a ``[depth, height, width]`` float32 MRI-like stack for all examples.

    The volume contains asymmetric body, bone, and patella-shaped regions whose
    positions vary slightly by slice. A single extreme-bright pixel is added to
    demonstrate why percentile-based intensity scaling is needed.
    """
    rows, cols = np.mgrid[:height, :width]
    stack = np.zeros((depth, height, width), dtype=np.float32)
    for index in range(depth):
        body = ((rows - 62) / 46) ** 2 + ((cols - 91) / 59) ** 2 < 1
        bone = ((rows - (54 + index)) / 17) ** 2 + ((cols - 98) / 23) ** 2 < 1
        patella = ((rows - 52) / 10) ** 2 + ((cols - 132) / 9) ** 2 < 1
        stack[index] = 12.0 + body * (115.0 + 6.0 * index) + bone * 260.0 + patella * 90.0
    stack[:, 2, 3] = 3000.0
    return stack


def _save(fig: plt.Figure, path: Path) -> None:
    """Lay out, write, and close one Matplotlib figure at the requested PNG path.

    Closing the figure releases its plotting resources so repeated example generation
    does not keep figures open in memory.
    """
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def orientation_example(out_dir: Path) -> Path:
    """Visualize canonical sagittal orientation as transpose-and-flip operations.

    It constructs a deliberately asymmetric one-slice image, applies the real
    ``canonical_inplane_transform`` for a sagittal DICOM orientation, and asserts
    that only pixel positions change. The returned path identifies the saved
    before-and-after PNG in ``out_dir``.
    """
    image = np.zeros((1, 48, 72), dtype=np.float32)
    image[0, 5:18, 7:14] = 1.0
    image[0, 34:42, 54:67] = 0.55
    image[0, 21:26, 29:34] = 0.25

    # Sagittal input whose row and column directions require a transpose and flips.
    iop = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    transform = canonical_inplane_transform(iop, "sagittal")
    oriented = transform.apply(image)
    assert sorted(oriented.ravel()) == sorted(image.ravel())
    assert oriented.shape == (1, 72, 48)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(image[0], cmap="magma", vmin=0, vmax=1)
    axes[0].set_title("Decoded array orientation")
    axes[1].imshow(oriented[0], cmap="magma", vmin=0, vmax=1)
    axes[1].set_title(f"Canonical sagittal: {transform.describe()}")
    for axis in axes:
        axis.set_xlabel("columns")
        axis.set_ylabel("rows")
    path = out_dir / "01_canonical_orientation.png"
    _save(fig, path)
    return path


def crop_centre_example(stack: np.ndarray, out_dir: Path) -> Path:
    """Compare geometric, foreground-centroid, and foreground-extent crop centres.

    The function calls ``estimate_center`` with all supported policies on ``stack``
    and overlays their row/column positions on its middle slice. It verifies that the
    synthetic posterior-heavy foreground pulls the centroid behind the extent centre,
    then returns the generated comparison PNG path.
    """
    centers = {mode: estimate_center(stack, mode) for mode in ("geometric", "foreground", "foreground_extent")}
    assert centers["foreground"][1] < centers["foreground_extent"][1]

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.imshow(stack[len(stack) // 2], cmap="gray")
    colors = {"geometric": "cyan", "foreground": "orange", "foreground_extent": "lime"}
    for mode, (row, col) in centers.items():
        axis.plot(col, row, marker="+", ms=14, mew=2.2, color=colors[mode], label=mode)
    axis.set_title("Crop-centre policies on asymmetric foreground")
    axis.legend(loc="lower left")
    path = out_dir / "02_crop_centres.png"
    _save(fig, path)
    return path


def real_dicom_crop_centre_example(
    series_dir: str | Path,
    out_dir: Path,
    volume_key: str | None = None,
    monochrome1_policy: str = "invert",
    out_name: str = "06_real_dicom_crop_centres.png",
) -> Path:
    """Visualize the real pipeline crop-centre decisions on one decoded DICOM series.

    ``series_dir`` must contain one DICOM series. The function creates its actual
    geometry-aware volume candidates, selects the requested ``volume_key`` or the
    longest usable candidate, then decodes it in geometric slice order and canonical
    in-plane orientation. It overlays the geometric, foreground-centroid, and
    foreground-extent centres plus the 150 mm foreground-centred crop boundary on the
    middle slice. ``monochrome1_policy`` is passed to the production decoder. The
    ``out_name`` selects the PNG filename under ``out_dir``. The returned path
    identifies the saved PNG; no image data are written besides that PNG.
    """
    candidates, _, flags = build_volume_candidates(Path(series_dir))
    if not candidates:
        raise ValueError(f"No geometrically usable DICOM candidates in {series_dir}: {', '.join(flags) or 'unknown error'}")

    if volume_key is not None:
        selected = next((candidate for candidate in candidates if candidate.volume_key == volume_key), None)
        if selected is None:
            available = ", ".join(candidate.volume_key for candidate in candidates)
            raise ValueError(f"Volume key {volume_key!r} not found; available keys: {available}")
    else:
        selected = max(candidates, key=lambda candidate: (candidate.usable(min_slices=4), candidate.n_slices))

    stack = decode_volume(selected, monochrome1_policy=monochrome1_policy)
    row_spacing, col_spacing = selected.inplane.apply_spacing(selected.row_spacing, selected.col_spacing)
    centers = {mode: estimate_center(stack, mode) for mode in ("geometric", "foreground", "foreground_extent")}
    _, crop_info = physical_crop(
        stack,
        row_spacing=row_spacing,
        col_spacing=col_spacing,
        fov_mm=150.0,
        center_rc=centers["foreground"],
    )

    fig, axis = plt.subplots(figsize=(8, 7))
    middle = stack[len(stack) // 2]
    finite = np.nan_to_num(middle, nan=0.0)
    low, high = np.percentile(finite, (1.0, 99.0))
    axis.imshow(finite, cmap="gray", vmin=low, vmax=high if high > low else None)
    axis.add_patch(
        Rectangle(
            (crop_info["crop_col0"], crop_info["crop_row0"]),
            crop_info["crop_cols_px"],
            crop_info["crop_rows_px"],
            fill=False,
            edgecolor="yellow",
            linewidth=1.5,
            label="foreground crop, 150 mm",
        )
    )
    colors = {"geometric": "cyan", "foreground": "orange", "foreground_extent": "lime"}
    for mode, (row, col) in centers.items():
        axis.plot(col, row, marker="+", ms=14, mew=2.2, color=colors[mode], label=mode)
    axis.set_title(
        f"Real DICOM: {selected.plane.plane}, {selected.n_slices} slices, {selected.inplane.describe()}\n"
        f"{row_spacing:.2f} x {col_spacing:.2f} mm; candidate {selected.volume_key}"
    )
    axis.set_xlabel("canonical columns")
    axis.set_ylabel("canonical rows")
    axis.legend(loc="lower left", fontsize=8)
    path = out_dir / out_name
    _save(fig, path)
    return path


def real_dicom_three_plane_examples(out_dir: Path) -> list[Path]:
    """Generate fixed real-DICOM crop-centre examples for all three anatomical planes.

    The embedded local paths identify one sagittal, coronal, and axial acquisition
    from the same study. Every series has a 180-by-180 mm in-plane field of view, so
    the fixed 150 mm crop retains about ``(150 / 180) ** 2 = 69.4%`` of the source
    area. Each candidate is explicitly constrained to acquisition group ``g0`` and
    checked for its expected plane before a separately named PNG is returned.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    series_examples = {
        "sagittal": Path(
            "F:/Kaggle/data/train_series/1.2.826.0.1.3680043.8.498.10009278692606631573540062909909132231/"
            "1.2.826.0.1.3680043.8.498.33836648034724869721686434744559693623"
        ),
        "coronal": Path(
            "F:/Kaggle/data/train_series/1.2.826.0.1.3680043.8.498.10009278692606631573540062909909132231/"
            "1.2.826.0.1.3680043.8.498.12689544351656485342798482846432921455"
        ),
        "axial": Path(
            "F:/Kaggle/data/train_series/1.2.826.0.1.3680043.8.498.10009278692606631573540062909909132231/"
            "1.2.826.0.1.3680043.8.498.26515132645256697953312631089055309042"
        ),
    }
    artifacts = []
    for plane, series_dir in series_examples.items():
        candidates, _, flags = build_volume_candidates(series_dir)
        selected = next((candidate for candidate in candidates if candidate.volume_key == "g0"), None)
        if selected is None:
            raise ValueError(f"Missing g0 candidate for {plane} example at {series_dir}: {', '.join(flags)}")
        if selected.plane.plane != plane:
            raise AssertionError(f"Expected {plane} example, got {selected.plane.plane} at {series_dir}")
        row_spacing, col_spacing = selected.inplane.apply_spacing(selected.row_spacing, selected.col_spacing)
        source_rows, source_cols = selected.slices[0].rows, selected.slices[0].cols
        if selected.inplane.transpose:
            source_rows, source_cols = source_cols, source_rows
        crop_area_fraction = min(1.0, 150.0 / (source_rows * row_spacing)) * min(
            1.0, 150.0 / (source_cols * col_spacing)
        )
        assert 0.68 <= crop_area_fraction <= 0.71, (plane, crop_area_fraction)
        artifacts.append(
            real_dicom_crop_centre_example(
                series_dir,
                out_dir,
                volume_key="g0",
                out_name=f"06_real_dicom_{plane}_crop_centres.png",
            )
        )
    return artifacts


def physical_crop_example(stack: np.ndarray, out_dir: Path) -> Path:
    """Show an anisotropic fixed-mm crop and its not-acquired NaN padding.

    The chosen 0.5 mm row and 1.0 mm column spacings turn a 100 mm field of view into
    a 200-by-100-pixel crop. An off-centre crop deliberately extends past the source
    matrix; the saved PNG marks the source boundary and displays padded NaNs in red.
    The function returns that PNG path.
    """
    center = (20.0, 30.0)
    cropped, info = physical_crop(stack, row_spacing=0.5, col_spacing=1.0, fov_mm=100.0, center_rc=center)
    assert cropped.shape == (stack.shape[0], 200, 100)
    assert info["crop_pad_fraction"] > 0.0

    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("crimson")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    middle = stack[len(stack) // 2]
    axes[0].imshow(middle, cmap="gray")
    axes[0].add_patch(
        Rectangle(
            (info["crop_col0"], info["crop_row0"]),
            info["crop_cols_px"],
            info["crop_rows_px"],
            fill=False,
            edgecolor="cyan",
            linewidth=2,
        )
    )
    axes[0].set_title("Source and 100 mm crop boundary")
    axes[1].imshow(np.ma.masked_invalid(cropped[len(cropped) // 2]), cmap=cmap)
    axes[1].set_title(f"Physical crop: NaN padding = {info['crop_pad_fraction']:.1%}")
    for axis in axes:
        axis.set_xlabel("columns")
        axis.set_ylabel("rows")
    path = out_dir / "03_physical_crop_and_padding.png"
    _save(fig, path)
    return path


def scaling_and_resampling_example(stack: np.ndarray, out_dir: Path) -> tuple[Path, Path]:
    """Illustrate robust intensity normalization followed by antialiased resampling.

    A 100-by-100 crop is scaled with the real foreground p1/p99 percentile rule,
    which clips the synthetic outlier and maps finite values to ``[0, 1]``. It is then
    resized with ``resample_square`` to 96-by-96 pixels. The two returned paths are,
    respectively, the scaling comparison and the final resampling comparison.
    """
    cropped, _ = physical_crop(stack, row_spacing=1.0, col_spacing=1.0, fov_mm=100.0, center_rc=(60.0, 95.0))
    scaled, scale_info = robust_scale(cropped, (1.0, 99.0))
    final = resample_square(scaled, 96)
    assert np.isfinite(scaled).all() and 0.0 <= float(scaled.min()) <= float(scaled.max()) <= 1.0
    assert final.shape == (stack.shape[0], 96, 96)

    index = len(stack) // 2
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(cropped[index], cmap="gray")
    axes[0].set_title("Before scaling: outlier compresses contrast")
    axes[1].imshow(scaled[index], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Robust scale: p1={scale_info['p_low']:.1f}, p99={scale_info['p_high']:.1f}")
    for axis in axes:
        axis.axis("off")
    scale_path = out_dir / "04_robust_intensity_scaling.png"
    _save(fig, scale_path)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(scaled[index], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Scaled crop: {scaled.shape[1]} x {scaled.shape[2]}")
    axes[1].imshow(final[index], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Antialiased square output: 96 x 96")
    for axis in axes:
        axis.axis("off")
    resample_path = out_dir / "05_square_resampling.png"
    _save(fig, resample_path)
    return scale_path, resample_path


def generate_examples(out_dir: str | Path) -> list[Path]:
    """Generate and validate the complete five-image preprocessing example gallery.

    ``out_dir`` is created when absent. The routine builds one shared synthetic stack,
    invokes every individual visualization, verifies each artifact exists and is
    non-empty, then returns the five resulting PNG paths in pipeline order.
    """
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stack = _synthetic_stack()
    artifacts = [orientation_example(destination), crop_centre_example(stack, destination), physical_crop_example(stack, destination)]
    artifacts.extend(scaling_and_resampling_example(stack, destination))
    assert all(path.exists() and path.stat().st_size > 0 for path in artifacts)
    return artifacts


def test_visual_preprocessing_examples(tmp_path: Path) -> None:
    """Exercise gallery generation in pytest's temporary directory.

    This confirms all assertions in the visual examples hold and all five expected
    artifacts can be written without polluting the repository working tree.
    """
    assert len(generate_examples(tmp_path)) == 5


def main() -> None:
    """Parse the optional output directory and print every generated artifact path.

    This is the direct command-line entry point used to create a browsable local
    gallery outside pytest; it relies on ``generate_examples`` for all validation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).with_name("artifacts"))
    parser.add_argument("--dicom-series-dir", type=Path, help="Optional real DICOM series directory for crop-centre visualization.")
    parser.add_argument("--volume-key", help="Optional acquisition-group key within --dicom-series-dir.")
    parser.add_argument("--monochrome1-policy", choices=("invert", "keep"), default="invert")
    args = parser.parse_args()
    for artifact in generate_examples(args.out_dir):
        print(artifact)
    for artifact in real_dicom_three_plane_examples(args.out_dir):
        print(artifact)
    if args.dicom_series_dir is not None:
        print(
            real_dicom_crop_centre_example(
                args.dicom_series_dir,
                args.out_dir,
                volume_key=args.volume_key,
                monochrome1_policy=args.monochrome1_policy,
            )
        )


if __name__ == "__main__":
    main()