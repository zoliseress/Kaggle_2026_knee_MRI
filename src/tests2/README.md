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