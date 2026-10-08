"""Run the production 150 mm crop on the six wide-field-of-view series flagged for a second leg.

A scan of all training series for images containing both knees found none, but six
wide-field series showed a strip of the other leg at the image edge (two or more large
foreground blobs on a middle slice). This script checks what the fixed physical crop keeps
of them, with the config's crop-centre policy and, for comparison, the geometric centre
(``--crop-centers`` selects the extra policies). For every series and policy it saves the
cropped volume as a 3D NRRD file and a PNG screenshot next to it.

Run from ``src`` (pynrrd is needed, e.g. the fov_review env)::

    python tests2\\flagged_wide_fov_crops.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nrrd
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knee_mri.config import load_config
from knee_mri.dicom_io import build_volume_candidates, decode_volume
from knee_mri.preprocess import estimate_center, physical_crop

# (StudyInstanceUID, SeriesInstanceUID, SeriesDescription) of the six flagged series.
FLAGGED_SERIES = [
    ("1.2.826.0.1.3680043.8.498.10306159113324811538703788080836752052",
     "1.2.826.0.1.3680043.8.498.28095032587022735220409990288365943063", "Cor T2 FSE FS_sm_AIRS"),
    ("1.2.826.0.1.3680043.8.498.12553455350653672209409408595948190982",
     "1.2.826.0.1.3680043.8.498.43047475598279996138731702872576045365", "pd_tse_fs_cor_test_d"),
    ("1.2.826.0.1.3680043.8.498.27324284859290040148888098857880927353",
     "1.2.826.0.1.3680043.8.498.11283201007845719296091222026605563223", "DummySeriesDesc!"),
    ("1.2.826.0.1.3680043.8.498.60064080350033586611530748685370040101",
     "1.2.826.0.1.3680043.8.498.12428803357983578145196137818246710592", "DummySeriesDesc!"),
    ("1.2.826.0.1.3680043.8.498.60064080350033586611530748685370040101",
     "1.2.826.0.1.3680043.8.498.63000861685769444367937811934795700703", "DummySeriesDesc!"),
    ("1.2.826.0.1.3680043.8.498.95351925765761617117057799846763403963",
     "1.2.826.0.1.3680043.8.498.68547278990366168272004289944361631728", "DummySeriesDesc!"),
]


def _display_range(image: np.ndarray) -> tuple[float, float]:
    """Return a robust (p1, p99) grey-level window of the finite pixels of ``image``."""
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, (1.0, 99.0))
    return float(low), float(high if high > low else low + 1.0)


def load_series(series_dir: Path, cfg, volume_key: str = "g0") -> dict:
    """Decode one series in the production geometric order and canonical orientation.

    The candidate builder runs with the manifest tolerances of ``cfg``; ``volume_key``
    selects the acquisition group. The returned dict holds the volume candidate, the
    decoded stack and the spacings in mm.
    """
    candidates, _, flags = build_volume_candidates(
        series_dir,
        obliquity_tolerance_deg=float(cfg.manifest.obliquity_tolerance_deg),
        max_gap_ratio=float(cfg.manifest.max_gap_ratio),
        min_slices=int(cfg.manifest.min_slices),
    )
    volume = next((candidate for candidate in candidates if candidate.volume_key == volume_key), None)
    if volume is None:
        raise ValueError(f"No {volume_key} candidate in {series_dir}: {', '.join(flags) or 'unknown error'}")
    stack = decode_volume(volume, monochrome1_policy=cfg.data.monochrome1_policy)
    row_spacing, col_spacing = volume.inplane.apply_spacing(volume.row_spacing, volume.col_spacing)
    return {
        "volume": volume,
        "stack": stack,
        "row_spacing": float(row_spacing),
        "col_spacing": float(col_spacing),
        "slice_spacing": float(volume.order.median_spacing),
    }


def crop_series(series: dict, fov_mm: float, crop_center: str) -> dict:
    """Apply a crop-centre policy and the fixed physical crop to a loaded series.

    ``crop_center`` is any ``estimate_center`` mode (``foreground``, ``foreground_extent``,
    ``geometric``). Returns ``series`` extended with the cropped stack (NaN where the crop
    extends past the acquired matrix), the policy and the ``physical_crop`` info.
    """
    center = estimate_center(series["stack"], crop_center)
    cropped, info = physical_crop(series["stack"], series["row_spacing"], series["col_spacing"], fov_mm, center)
    return {**series, "cropped": cropped, "crop_center": crop_center, "info": info}


def save_nrrd(result: dict, path: Path, study: str, series: str) -> Path:
    """Write the cropped volume as a 3D float32 NRRD with its voxel spacing.

    The array is stored in the pipeline's canonical in-plane orientation, file axes
    (column, row, slice), so the spacings are (col, row, slice) mm. Padding outside the
    acquired matrix (NaN in the pipeline) is written as 0; its fraction is in the header.
    """
    cropped = np.nan_to_num(result["cropped"], nan=0.0).astype(np.float32)
    data = np.transpose(cropped, (2, 1, 0))  # [Z, rows, cols] -> (col, row, slice)
    info = result["info"]
    header = {
        "spacings": [result["col_spacing"], result["row_spacing"], result["slice_spacing"]],
        "kinds": ["domain", "domain", "domain"],
        "labels": ["column", "row", "slice"],
        # Non-standard fields are written as NRRD key/value pairs ("key:=value").
        "StudyInstanceUID": study,
        "SeriesInstanceUID": series,
        "plane": result["volume"].plane.plane,
        "inplane_ops": result["volume"].inplane.describe(),
        "fov_mm": str(info["fov_mm"]),
        "crop_center_policy": result["crop_center"],
        "crop_center_rc": f"{info['crop_center_row']} {info['crop_center_col']}",
        "crop_pad_fraction": str(info["crop_pad_fraction"]),
    }
    nrrd.write(str(path), data, header)
    return path


def save_screenshot(result: dict, path: Path, title: str) -> Path:
    """Save the source middle slice with the crop box next to the cropped middle slice."""
    stack, cropped, info = result["stack"], result["cropped"], result["info"]
    index = len(stack) // 2
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("crimson")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    low, high = _display_range(stack[index])
    axes[0].imshow(stack[index], cmap="gray", vmin=low, vmax=high)
    axes[0].add_patch(
        Rectangle((info["crop_col0"], info["crop_row0"]), info["crop_cols_px"], info["crop_rows_px"],
                  fill=False, edgecolor="yellow", linewidth=1.5)
    )
    axes[0].plot(info["crop_center_col"], info["crop_center_row"], marker="+", ms=14, mew=2.2, color="orange")
    axes[0].set_title(
        f"Source, slice {index + 1}/{len(stack)}: {stack.shape[2] * result['col_spacing']:.0f} x "
        f"{stack.shape[1] * result['row_spacing']:.0f} mm"
    )
    low, high = _display_range(cropped[index])
    axes[1].imshow(np.ma.masked_invalid(cropped[index]), cmap=cmap, vmin=low, vmax=high)
    axes[1].set_title(f"{info['fov_mm']:.0f} mm crop: padding (red) = {info['crop_pad_fraction']:.1%}")
    for axis in axes:
        axis.set_xlabel("canonical columns")
        axis.set_ylabel("canonical rows")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    """Crop the six flagged series with each crop-centre policy and print the written paths.

    The config's policy writes ``<stem>.nrrd/.png``; every other policy in
    ``--crop-centers`` adds a ``_<policy>`` suffix, so the outputs sit side by side.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Config YAML (default: src/config.yaml).")
    parser.add_argument("--dicom-root", type=Path, default=None, help="Default: the config's DICOM root.")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).with_name("artifacts") / "flagged_wide_fov_crops")
    parser.add_argument("--crop-centers", nargs="+", default=["geometric"],
                        choices=("foreground", "foreground_extent", "geometric"),
                        help="Policies to run besides the config's data.crop_center (default: geometric).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dicom_root = args.dicom_root or Path(cfg.paths.dicom_root)
    policies = list(dict.fromkeys([cfg.data.crop_center, *args.crop_centers]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for number, (study, series, description) in enumerate(FLAGGED_SERIES, 1):
        loaded = load_series(dicom_root / study / series, cfg)
        for policy in policies:
            result = crop_series(loaded, float(cfg.data.fov_mm), policy)
            assert result["cropped"].shape[0] == len(result["stack"])
            suffix = "" if policy == cfg.data.crop_center else f"_{policy}"
            stem = f"{number:02d}_{study[-6:]}_{series[-6:]}{suffix}"
            title = (
                f"{study}\n{series}\n{result['volume'].plane.plane}, {description}, {len(result['stack'])} slices, "
                f"crop centre: {policy}"
            )
            print(save_nrrd(result, args.out_dir / f"{stem}.nrrd", study, series))
            print(save_screenshot(result, args.out_dir / f"{stem}.png", title))


if __name__ == "__main__":
    main()
