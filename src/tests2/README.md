# Visual preprocessing examples

This directory contains data-independent visual examples for the deterministic image
preprocessing path:

1. canonical in-plane orientation;
2. foreground crop-centre policies;
3. fixed-physical crop and out-of-acquisition padding;
4. robust per-series intensity scaling; and
5. square resampling.

Run the examples from `src`:

```powershell
python tests2\visual_preprocessing_examples.py
```

The command writes PNG files to `tests2/artifacts/` by default. To select another
location, pass `--out-dir <path>`. The script also contains numeric assertions, so it
fails when an illustrated preprocessing contract no longer holds.

To add `06_real_dicom_crop_centres.png`, provide a directory containing one real
DICOM series. The example uses the production candidate builder and decoder, then
overlays all crop-centre policies and the foreground-centred 150 mm crop on the
canonical middle slice:

```powershell
python tests2\visual_preprocessing_examples.py --dicom-series-dir <study\series-directory>
```

When a DICOM series holds multiple acquisition groups, use `--volume-key g0` (or the
needed group key) to choose one explicitly. Without it, the longest usable candidate
is used.

```powershell
python -m pytest tests2 -q
```

The first five images use synthetic MRI-like stacks only and neither read DICOM files
nor depend on the local cache. The sixth image is optional and reads only the DICOM
series passed through `--dicom-series-dir`.

## 150 mm crop of the flagged wide-FOV series

`flagged_wide_fov_crops.py` runs the production crop-centre policy and 150 mm physical
crop (settings from `src/config.yaml`) on the six wide-field series in which a strip of
the other leg is visible at the image edge, and for comparison the `geometric` centre
(`--crop-centers` selects other extra policies). For each series and policy it writes the cropped volume
as a 3D NRRD (canonical in-plane orientation, voxel spacing set, padding written as 0)
and a PNG with the source middle slice, the crop box and the cropped middle slice. It
reads the DICOM files under the config's DICOM root and needs `pynrrd`:

```powershell
python tests2\flagged_wide_fov_crops.py
```

Output: `tests2/artifacts/flagged_wide_fov_crops/`; the config's policy writes
`<nn>_<study>_<series>.nrrd/.png`, the others add a `_<policy>` suffix.

## Crop-centre policy comparison

`crop_center_policy_comparison.py` runs every selected series of a seeded random study
sample (by default the same 500 studies as `cli qc-edges`) through the production path
with each crop-centre policy (`foreground`, `foreground_extent`, `geometric`) and scores
the result with the `qc-edges` crop-edge metric (edge band tissue fraction > 0.3 = cut)
and the crop padding fraction. It decodes DICOM, so it takes a while:

```powershell
python tests2\crop_center_policy_comparison.py --workers 12
```

Output: `tests2/artifacts/crop_center_comparison/` (`per_series.csv`, `summary.csv`,
`paired.csv`, `divergent_<slot>.png`).