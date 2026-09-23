"""Generate the Kaggle inference notebook and a flat, locally runnable copy.

Usage:  python notebooks/build_04_notebook.py <out.ipynb> <out_flat.py>

The notebook cells live here as strings so that the .py used for local verification
and the .ipynb shipped to Kaggle are generated from exactly the same source. Edit the
notebook *here*: a change made directly in the .ipynb makes this file stale.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# ======================================================================================
md(r"""
# RSNA Knee Abnormality Detection — teszt inferencia

EfficientNet-B0 2.5D MIL baseline (`knee_mri` csomag) futtatása a verseny **teszt**
adathalmazán, `submission.csv` előállításával.

A notebook **nem** duplikálja a pipeline-t: a repóban levő, tanításkor is használt
függvényeket hívja (`build_manifest` → `select_series` → `build_cache` → modell),
így a teszt képek pontosan ugyanazon a determinisztikus úton készülnek, mint a tanító
képek. Ezt a notebook ellenőrzi is: a checkpointba mentett preprocessing-hash-nek
egyeznie kell az itt számolttal.

{{INTRO_PATHS}}

## Hogyan fut

A rejtett teszt mérete előre nem ismert, ezért a feldolgozás **chunkokban** megy:
`CHUNK_STUDIES` study-ra épül manifest → series-kiválasztás → preprocessing cache →
predikció, majd a cache törlődik. Így a `/kaggle/working` mérete a teszt méretétől
függetlenül korlátos, és egy hibás study nem viszi el az egész beadást.

Internet nem kell: a súlyok a checkpointból jönnek (`model.weights="none"`), semmit
nem töltünk le.

> Ellenőrizve: ugyanez az út a fold-0 validációs study-kra lefuttatva a tanításkor
> exportált `validation_predictions.csv` értékeit adja vissza (max. eltérés 1.0e-3,
> átlag 2.9e-5 — ez a bf16 autocast numerikus zaja).
""")

# ======================================================================================
md(r"""
## 1. Beállítások

Ez az egyetlen cella, amit rendes esetben módosítani kell.
""")

code(r"""
# --------------------------------------------------------------------------------------
# Settings. Everything you are likely to change lives here.
# --------------------------------------------------------------------------------------

# Paths. Nothing is searched for: every one of these must exist as written.
{{PATHS}}

SMOKE_TEST_STUDIES = None   # None = all test studies; e.g. 8 = first 8 only (debugging)
CHUNK_STUDIES      = 32     # studies per manifest -> cache -> predict -> cleanup cycle
CACHE_WORKERS      = None   # None = os.cpu_count() capped at 8; DICOM decoding is CPU bound
MP_START_METHOD    = "spawn"  # "fork" starts faster but can deadlock after CUDA init
DATALOADER_WORKERS = 2      # cache readers feeding the GPU (0 on Windows)
EVAL_BATCH_STUDIES = 2      # studies per forward pass; lower it if the GPU runs out
AMP                = "auto" # auto|bf16|fp16|fp32

REQUIRE_PREP_MATCH = True   # abort if a checkpoint was trained on other pixels
TIME_BUDGET_HOURS  = 8.0    # stop early and still write a submission before the limit
MAX_CHUNK_FAILURES = 5      # abort after this many failed chunks
KEEP_CACHE         = False  # True = keep the preprocessed npz files (small tests only)
""")

# ======================================================================================
md(r"""
## 2. Környezet: útvonalak ellenőrzése

Semmit nem keresünk: a fenti útvonalaknak úgy kell létezniük, ahogy le vannak írva.

{{LAYOUT}}

A checkpointok mellé a saját futásuk `config.yaml`-ja is kell: a képalkotó paraméterek
onnan jönnek. Egy modellel futtatáshoz szűkítsd a `CHECKPOINT_GLOB`-ot egyetlen foldra.
""")

code(r"""
import gc
import glob
import json
import logging
import multiprocessing
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

COMP_DIR = Path(COMPETITION_DIR)
CODE_DIR = Path(CODE_DIR)
# Absolute: knee_mri resolves a relative config path against its own repo root, which
# would put the cache somewhere else than the rest of the working directory.
WORK_DIR = Path(WORK_DIR).resolve()

for label, path in (("COMPETITION_DIR", COMP_DIR / "test.csv"),
                    ("COMPETITION_DIR", COMP_DIR / "test_series"),
                    ("CODE_DIR", CODE_DIR / "knee_mri" / "__init__.py")):
    if not path.exists():
        raise FileNotFoundError("{} is wrong: {} does not exist".format(label, path))

CHECKPOINTS = sorted(Path(p) for p in glob.glob(CHECKPOINT_GLOB))
if not CHECKPOINTS:
    raise FileNotFoundError("CHECKPOINT_GLOB matches nothing: " + CHECKPOINT_GLOB)

sys.path.insert(0, str(CODE_DIR))
WORK_DIR.mkdir(parents=True, exist_ok=True)
# Capped: os.cpu_count() can report the host's cores, and more decoders than real cores
# only thrash the read-only input mount.
CACHE_WORKERS = int(CACHE_WORKERS or min(os.cpu_count() or 1, 8))

# The cache workers call torch (preprocess.resample_square uses F.interpolate). Forking
# them from a process that already holds CUDA models and torch's thread pools inherits
# those threads' locks without the threads, which deadlocks on the child's first torch
# call - the manifest workers never touch torch, which is why only the cache stage hangs.
# spawn costs a few seconds of import per pool and avoids that class of hang entirely.
if MP_START_METHOD:
    multiprocessing.set_start_method(MP_START_METHOD, force=True)

# One math thread per worker process: N workers x N intra-op threads on N cores only
# contend. Set before torch is imported so the setting also reaches spawned children.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

print("mp start method:", multiprocessing.get_start_method())

print("competition    :", COMP_DIR)
print("code (sys.path):", CODE_DIR)
print("work dir       :", WORK_DIR)
print("cache workers  :", CACHE_WORKERS)
print("checkpoints    :")
for path in CHECKPOINTS:
    print("  - {}  ({:.0f} MB)".format(path, path.stat().st_size / 1e6))
""")

# ======================================================================================
code(r"""
import torch

from knee_mri.config import load_config, resolve_paths, validate_config
from knee_mri.constants import SERIES_ID, STUDY_ID, TARGETS
from knee_mri.dataset import StudyBagDataset, collate_studies
from knee_mri.evaluate import load_checkpoint_for_inference
from knee_mri.manifest import build_manifest, select_series
from knee_mri.preprocess import build_cache, cache_root, preprocess_hash, preprocess_signature
from knee_mri.utils import LOG, autocast_ctx, environment_report, seed_everything, select_device

print(json.dumps(environment_report(), indent=1, default=str))
print("\ntargets:", TARGETS)
""")

# ======================================================================================
md(r"""
## 3. Inferencia-konfiguráció

A konfiguráció a tanító futás `config.yaml`-jából jön (a checkpoint mellől), hogy a
képalkotás minden paramétere egyezzen. Csak az útvonalak és a futásidejű beállítások
íródnak felül.
""")

code(r"""
CONFIG_PATH = CHECKPOINTS[0].parent / "config.yaml"
if not CONFIG_PATH.is_file():
    raise FileNotFoundError(
        "No config.yaml next to {}. Upload each run directory with its own "
        "config.yaml: the image parameters must come from the training run.".format(
            CHECKPOINTS[0])
    )
print("config:", CONFIG_PATH)

cfg = load_config(CONFIG_PATH, resolve=False)

# Paths: read the test export, write everything else under the working directory.
cfg.set_dotted("paths.data_root", str(COMP_DIR))
cfg.set_dotted("paths.dicom_root", str(COMP_DIR / "test_series"))
cfg.set_dotted("paths.train_csv", str(COMP_DIR / "test.csv"))            # not read here
cfg.set_dotted("paths.train_series_csv", str(COMP_DIR / "test_series.csv"))
cfg.set_dotted("paths.work_dir", str(WORK_DIR))
cfg.set_dotted("paths.cache_dir", str(WORK_DIR / "cache"))
cfg.set_dotted("paths.output_dir", str(WORK_DIR / "runs"))
for key in (
    "reference_csv",
    "labels_details_csv",
    "labels_statuses_csv",
    "labels_predictions_csv",
    "labels_predictions_exclude_borderline_csv",
):
    cfg.set_dotted("paths." + key, None)

# Runtime: no label files, no augmentation, no pretrained download.
cfg.set_dotted("model.weights", "none")
cfg.set_dotted("augment.enabled", False)
cfg.set_dotted("manifest.workers", CACHE_WORKERS)
cfg.set_dotted("train.num_workers", int(DATALOADER_WORKERS))
cfg.set_dotted("train.prefetch_factor", None)
cfg.set_dotted("train.eval_batch_studies", int(EVAL_BATCH_STUDIES))
cfg.set_dotted("train.amp", AMP)

resolve_paths(cfg)
validate_config(cfg)
seed_everything(int(cfg.seed))

PREP_HASH = preprocess_hash(cfg)
print(json.dumps(preprocess_signature(cfg), indent=1))
print("\npreprocessing hash:", PREP_HASH)
print("image size {}, slots {}, centers/series {}".format(
    cfg.data.image_size, list(cfg.data.series_slots), cfg.data.centers_per_series))
""")

# ======================================================================================
md(r"""
## 4. Checkpointok betöltése

Minden checkpoint ellenőrzése: formátumverzió, célváltozó-sorrend, és a tanításkori
preprocessing-hash. Eltérő hash azt jelentené, hogy a modell más képeket látott, mint
amit itt előállítunk — ez alapértelmezetten hiba, nem figyelmeztetés.

> A `model.weights=none: the encoder starts from RANDOM initialisation` figyelmeztetés
> itt **várt**: az architektúra üresen épül fel (nincs letöltés), a súlyok a rá
> következő `load_state_dict` hívásból jönnek.
""")

code(r"""
device_spec = select_device(cfg.train.amp)
print(device_spec.describe())

models = []
ckpt_rows = []
for path in CHECKPOINTS:
    model, payload = load_checkpoint_for_inference(cfg, path)
    stored_prep = str(payload.get("versions", {}).get("prep_hash", ""))
    if stored_prep and stored_prep != PREP_HASH:
        message = (
            "{}: trained with preprocessing hash {}, but this notebook produces {}. "
            "The model would see different pixels than in training."
        ).format(path, stored_prep, PREP_HASH)
        if REQUIRE_PREP_MATCH:
            raise RuntimeError(message + " Set REQUIRE_PREP_MATCH=False to override.")
        LOG.warning(message)
    model = model.to(device_spec.device).eval()
    models.append(model)
    ckpt_rows.append(
        {
            "checkpoint": str(path),
            "epoch": payload.get("epoch"),
            "best_epoch": payload.get("best_epoch"),
            "best_score": payload.get("best_score"),
            "prep_hash": stored_prep,
            "versions": payload.get("versions", {}),
        }
    )
    del payload

print("\n{} checkpoint(s) loaded on {}; probabilities are averaged over them.".format(
    len(models), device_spec.device))
print(pd.DataFrame(ckpt_rows)[["checkpoint", "best_epoch", "best_score", "prep_hash"]]
      .to_string(index=False))
""")

# ======================================================================================
md(r"""
## 5. Teszt adatok és a beadás vázlata

A beadás sor- és oszlopsorrendje a `sample_submission.csv`-ből jön, az oszlopok név
szerint töltődnek — így egy esetleges oszlopsorrend-változás sem cseréli fel a
célváltozókat.
""")

code(r"""
test_csv = pd.read_csv(COMP_DIR / "test.csv", dtype={STUDY_ID: "string"})
series_meta = pd.read_csv(
    COMP_DIR / "test_series.csv", dtype={STUDY_ID: "string", SERIES_ID: "string"}
)
sample_path = COMP_DIR / "sample_submission.csv"
sample_sub = (
    pd.read_csv(sample_path, dtype={STUDY_ID: "string"}) if sample_path.is_file() else None
)

if sample_sub is not None:
    missing_cols = sorted(set(TARGETS) - set(sample_sub.columns))
    extra_cols = sorted(set(sample_sub.columns) - set(TARGETS) - {STUDY_ID})
    if missing_cols or extra_cols:
        raise ValueError(
            "sample_submission columns do not match the project targets. "
            "missing={} unexpected={}".format(missing_cols, extra_cols)
        )
    SUB_COLUMNS = list(sample_sub.columns)
    study_ids = sample_sub[STUDY_ID].astype(str).tolist()
else:
    LOG.warning("No sample_submission.csv; falling back to test.csv order.")
    SUB_COLUMNS = [STUDY_ID] + list(TARGETS)
    study_ids = test_csv[STUDY_ID].astype(str).tolist()

if len(set(study_ids)) != len(study_ids):
    raise ValueError("Duplicate StudyInstanceUID in the submission index.")

dicom_root = Path(cfg.paths.dicom_root)
on_disk = {p.name for p in dicom_root.iterdir() if p.is_dir()}
absent = [s for s in study_ids if s not in on_disk]
if absent:
    LOG.warning(
        "%d study/studies have no directory under %s; they will get the fallback "
        "prediction (first: %s)", len(absent), dicom_root, absent[:3]
    )

if SMOKE_TEST_STUDIES:
    study_ids = study_ids[: int(SMOKE_TEST_STUDIES)]
    LOG.warning("SMOKE_TEST_STUDIES=%s: only %d studies will be predicted.",
                SMOKE_TEST_STUDIES, len(study_ids))

n_series = int(series_meta[series_meta[STUDY_ID].isin(study_ids)].shape[0])
print("studies to predict :", len(study_ids))
print("series listed      : {} ({:.1f} per study)".format(
    n_series, n_series / max(1, len(study_ids))))
print("submission columns :", SUB_COLUMNS)
""")

# ======================================================================================
md(r"""
## 6. Az inferencia lépései

`predict_studies_ensemble` a `knee_mri.evaluate.predict_studies` mintáját követi
(`model.eval()`, `torch.inference_mode()`, augmentáció és TTA nélkül), annyi
eltéréssel, hogy egy cache-olvasásból mind a három modellt kiszolgálja.
""")

code(r"""
@torch.inference_mode()
def predict_studies_ensemble(cfg, models, study_ids, device_spec):
    '''Deterministic inference over one chunk; mean of the per-checkpoint sigmoids.'''
    from torch.utils.data import DataLoader

    dataset = StudyBagDataset(cfg, list(study_ids), label_table=None, train=False)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(cfg.train.eval_batch_studies)),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        collate_fn=collate_studies,
        pin_memory=device_spec.device.type == "cuda",
    )
    ids: list[str] = []
    scores: list[np.ndarray] = []
    slots_present: list[int] = []
    for batch in loader:
        images = batch["images"].to(device_spec.device, non_blocking=True)
        slice_valid = batch["slice_valid_mask"].to(device_spec.device, non_blocking=True)
        present = batch["series_present_mask"].to(device_spec.device, non_blocking=True)
        total = None
        for model in models:
            with autocast_ctx(device_spec):
                logits = model(images, slice_valid, present)
            probs = torch.sigmoid(logits.float())
            total = probs if total is None else total + probs
        scores.append((total / len(models)).cpu().numpy())
        ids.extend(batch["study_ids"])
        slots_present.extend(int(m["n_present_slots"]) for m in batch["meta"])
    stacked = (
        np.concatenate(scores, axis=0)
        if scores
        else np.zeros((0, len(TARGETS)), dtype=np.float32)
    )
    return ids, stacked, slots_present


def stage(prefix, name, detail, seconds):
    '''One line per finished stage: most of a chunk's time is spent inside these.'''
    print("{}   {:<9} {:<44} {:6.1f} s".format(prefix, name, detail, seconds), flush=True)


def process_chunk(cfg, models, chunk_ids, series_meta, device_spec, chunk_dir, prefix=""):
    '''Manifest -> selection -> cache -> prediction for one chunk of studies.'''
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_slots = len(cfg.data.series_slots)

    started = time.time()
    manifest = build_manifest(cfg, study_ids=chunk_ids, out_path=chunk_dir / "manifest.csv")
    usable = int(manifest["usable"].fillna(False).astype(bool).sum())
    stage(prefix, "manifest",
          "{} volume candidates, {} usable".format(len(manifest), usable),
          time.time() - started)

    started = time.time()
    selection = select_series(
        cfg, manifest, series_meta=series_meta, out_path=chunk_dir / "series_selection.csv"
    )
    selected = selection["selected"].fillna(False).astype(bool)
    per_study = selection[selected].groupby(STUDY_ID).size() if int(selected.sum()) else []
    stage(prefix, "selection",
          "{} series, {}/{} studies with all {} slots".format(
              int(selected.sum()), int(sum(1 for n in per_study if n == n_slots)),
              len(chunk_ids), n_slots),
          time.time() - started)

    started = time.time()
    if int(selected.sum()):
        report = build_cache(cfg, selection, study_ids=chunk_ids, workers=CACHE_WORKERS)
        counts = report["status"].value_counts().to_dict()
    else:
        counts = {}
        LOG.warning("No usable series in this chunk; predicting from empty inputs.")
    stage(prefix, "cache",
          ", ".join("{} {}".format(v, k) for k, v in counts.items()) or "nothing to cache",
          time.time() - started)

    started = time.time()
    ids, scores, slots_present = predict_studies_ensemble(cfg, models, chunk_ids, device_spec)
    stage(prefix, "predict",
          "{} {} x {} model(s)".format(
              len(ids), "study" if len(ids) == 1 else "studies", len(models)),
          time.time() - started)
    coverage = (
        selection[selected].groupby(STUDY_ID)["slot"].apply(list).to_dict()
        if int(selected.sum())
        else {}
    )
    return ids, scores, slots_present, coverage


def drop_cache(cfg, chunk_ids):
    '''Free the working directory again: the cached pixels are not needed twice.'''
    root = cache_root(cfg)
    for study in chunk_ids:
        shutil.rmtree(root / str(study), ignore_errors=True)
""")

# ======================================================================================
md(r"""
## 7. Futtatás

Minden chunk négy szakaszból áll (manifest → kiválasztás → cache → predikció), és
mindegyik a végén kiír egy sort a saját idejével és darabszámaival — így futás közben
látszik, melyik szakaszban jár és mi lassú. A chunk záró sora adja az összesítést:
eddig kész study-k, studies/min, eltelt idő és ETA.

Chunkonként menti a részeredményt is (`predictions_partial.csv`).
Egy chunk hibája nem állítja le a futást: az érintett study-k a tartalék predikciót kapják,
a hiba pedig a `run_meta.json`-be kerül. Az időkeret elérésekor a maradék study-k szintén
tartalék predikciót kapnak, hogy a beadás mindenképp teljes legyen.
""")

code(r"""
CHUNKS = [
    study_ids[i : i + int(CHUNK_STUDIES)] for i in range(0, len(study_ids), int(CHUNK_STUDIES))
]
budget_s = float(TIME_BUDGET_HOURS) * 3600.0
started = time.time()

pred_ids: list[str] = []
pred_scores: list[np.ndarray] = []
coverage_rows: list[dict] = []
failures: list[dict] = []
stopped_early = None

# The stage lines below carry the same information as the knee_mri INFO log, more
# compactly. Warnings and errors still come through.
LOG.setLevel(logging.WARNING)

print("{} chunk(s) x {} studies, budget {:.1f} h\n".format(
    len(CHUNKS), CHUNK_STUDIES, TIME_BUDGET_HOURS))

for index, chunk_ids in enumerate(CHUNKS, 1):
    elapsed = time.time() - started
    done = len(pred_ids)
    if done:
        projected = elapsed / done * len(chunk_ids)
        if elapsed + 1.5 * projected > budget_s:
            stopped_early = (
                "time budget: {:.2f} h used of {:.1f} h, stopping before chunk {}/{}"
            ).format(elapsed / 3600, TIME_BUDGET_HOURS, index, len(CHUNKS))
            LOG.warning(stopped_early)
            break

    chunk_started = time.time()
    prefix = "[{:>4}/{}]".format(index, len(CHUNKS))
    first = study_ids.index(chunk_ids[0]) + 1
    print("{} studies {}-{} of {}".format(
        prefix, first, first + len(chunk_ids) - 1, len(study_ids)), flush=True)
    try:
        ids, scores, slots_present, coverage = process_chunk(
            cfg, models, chunk_ids, series_meta, device_spec,
            WORK_DIR / "chunks" / "{:05d}".format(index), prefix,
        )
        pred_ids.extend(ids)
        pred_scores.append(scores)
        for study, n_slots in zip(ids, slots_present):
            coverage_rows.append(
                {
                    STUDY_ID: study,
                    "n_present_slots": n_slots,
                    "slots": ";".join(coverage.get(study, [])),
                }
            )
    except Exception as exc:  # one bad chunk must not cost the whole submission
        failures.append(
            {"chunk": index, "studies": list(chunk_ids),
             "error": "{}: {}".format(type(exc).__name__, exc)}
        )
        LOG.error("Chunk %d failed: %s", index, exc)
        traceback.print_exc()
        if len(failures) > int(MAX_CHUNK_FAILURES):
            stopped_early = "{} chunks failed; aborting the loop".format(len(failures))
            LOG.error(stopped_early)
            break
    finally:
        if not KEEP_CACHE:
            drop_cache(cfg, chunk_ids)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - started
    done = len(pred_ids)
    rate = done / max(elapsed, 1e-6)
    remaining = (len(study_ids) - done) / max(rate, 1e-9)
    print(
        "{}-> {}/{} done in {:.1f} s | {:.1f} studies/min | elapsed {:.1f} min | "
        "ETA {:.1f} min\n".format(
            " " * len(prefix), done, len(study_ids), time.time() - chunk_started,
            rate * 60, elapsed / 60, remaining / 60),
        flush=True,
    )

    if pred_scores:
        partial = pd.DataFrame(np.concatenate(pred_scores, axis=0), columns=list(TARGETS))
        partial.insert(0, STUDY_ID, pred_ids)
        partial.to_csv(WORK_DIR / "predictions_partial.csv", index=False)

print("\npredicted {} / {} studies in {:.1f} min".format(
    len(pred_ids), len(study_ids), (time.time() - started) / 60))
""")

# ======================================================================================
md(r"""
## 8. A beadás összeállítása és ellenőrzése

A `submission.csv` csak akkor íródik ki, ha minden ellenőrzés átment: a sorok pontosan
a `sample_submission.csv` study-jai, ugyanabban a sorrendben, az oszlopok név szerint
kitöltve, minden érték véges és `[0, 1]`-ben van.
""")

code(r"""
predictions = pd.DataFrame(
    np.concatenate(pred_scores, axis=0) if pred_scores else np.zeros((0, len(TARGETS))),
    columns=list(TARGETS),
)
predictions.insert(0, STUDY_ID, pd.Series(pred_ids, dtype="string"))
if predictions[STUDY_ID].duplicated().any():
    raise ValueError("A study was predicted twice; the chunking is wrong.")

index_df = pd.DataFrame({STUDY_ID: pd.Series(study_ids, dtype="string")})
merged = index_df.merge(predictions, on=STUDY_ID, how="left")

incomplete = merged[list(TARGETS)].isna().any(axis=1)
n_missing = int(incomplete.sum())
if n_missing:
    # Never leave an empty cell: fall back to the per-target mean of what we did predict,
    # or to 0.5 when nothing was predicted at all. This is a stated fallback, not a score.
    fallback = (
        predictions[list(TARGETS)].mean(axis=0)
        if len(predictions)
        else pd.Series(0.5, index=list(TARGETS))
    )
    LOG.warning(
        "%d study/studies without a prediction; filling with %s",
        n_missing,
        "the per-target mean" if len(predictions) else "0.5",
    )
    for target in TARGETS:
        merged.loc[incomplete, target] = float(fallback[target])

values = merged[list(TARGETS)].to_numpy(dtype=np.float64)
if not np.isfinite(values).all():
    raise ValueError("Non-finite value in the submission.")
if values.min() < 0.0 or values.max() > 1.0:
    raise ValueError("Submission values outside [0, 1]: [{}, {}]".format(
        values.min(), values.max()))
if sample_sub is not None and not SMOKE_TEST_STUDIES and len(merged) != len(sample_sub):
    raise ValueError("Row count {} != sample_submission {}".format(len(merged), len(sample_sub)))

submission = merged.reindex(columns=SUB_COLUMNS)
out_path = Path(SUBMISSION_PATH)
submission.to_csv(out_path, index=False, float_format="%.6f")

coverage_df = pd.DataFrame(coverage_rows)
if len(coverage_df):
    coverage_df.to_csv(WORK_DIR / "test_coverage.csv", index=False)

run_meta = {
    "n_studies_indexed": len(study_ids),
    "n_studies_predicted": int(len(predictions)),
    "n_studies_filled": n_missing,
    "elapsed_minutes": round((time.time() - started) / 60.0, 2),
    "stopped_early": stopped_early,
    "chunk_failures": failures,
    "checkpoints": ckpt_rows,
    "preprocess_hash": PREP_HASH,
    "device": device_spec.describe(),
    "smoke_test_studies": SMOKE_TEST_STUDIES,
    "submission": str(out_path),
    "slot_coverage": (
        coverage_df["n_present_slots"].value_counts().sort_index().to_dict()
        if len(coverage_df)
        else {}
    ),
}
(WORK_DIR / "run_meta.json").write_text(
    json.dumps(run_meta, indent=2, default=str), encoding="utf-8")

print(json.dumps({k: v for k, v in run_meta.items() if k != "checkpoints"},
                 indent=1, default=str))
print("\nsubmission written:", out_path)
print(submission.head().to_string(index=False))
print("\nper-target mean probability:")
print(submission[list(TARGETS)].mean().round(4).to_string())
""")

# ======================================================================================
md(r"""
## Megjegyzések

**Mennyi ideig fut?** A szűk keresztmetszet a DICOM dekódolás és a preprocessing, nem a
GPU. A futás közben kiírt `studies/min` az első chunk után már reális — ebből látszik, hogy
belefér-e az időkeretbe. Ha nem: `CHUNK_STUDIES` növelése nem segít, viszont a
`data.centers_per_series` csökkentése (pl. 24 → 16) közel arányosan gyorsít a GPU oldalon,
egyetlen foldra szűkített `CHECKPOINT_GLOB` pedig harmadolja a forward időt.
Mindkettő ront a pontosságon.

**`submission.csv` mindenképpen készül.** Hibás chunk vagy időtúllépés esetén az érintett
study-k a már kiszámolt predikciók célváltozónkénti átlagát kapják; a `run_meta.json`
megmondja, hány sor készült így.

**Ha a preprocessing-hash nem egyezik** (`REQUIRE_PREP_MATCH`): a `CHECKPOINT_GLOB` első
találata melletti `config.yaml` nem ahhoz a futáshoz tartozik. Minden run könyvtárat a
`best.pt` mellett a saját `config.yaml`-jával tölts fel.

**Ha sok study 0 vagy 1 síkot kap** (`test_coverage.csv`, `slot_coverage`): a teszt
sorozatok másként vannak felcímkézve, mint a tanítóké. A `series_selection.csv` chunkonként
megmarad a `WORK_DIR/chunks/` alatt, a `reason` oszlopban benne van, miért nem lett
kiválasztva egy sorozat.

**Tömörített DICOM.** Ez az export Explicit VR Little Endian, plain pydicom dekódolja.
Ha a rejtett teszt tömörített, a manifest hangosan elhasal — ilyenkor a `pylibjpeg`
(vagy `gdcm`) wheel-eket dataset-ként kell csatolni, mert internet nincs.

**GPU memória.** Alapértelmezetten `EVAL_BATCH_STUDIES=2`, ami studynként
3 × 24 = 72 tripletet jelent, 224 × 224-en. Kevesebb memóriához vedd 1-re.
""")


# ======================================================================================
# The two deployments. Everything else in this file is shared between them.
# ======================================================================================

REPO = Path(__file__).resolve().parents[1]

VARIANTS = {
    "local": {
        "notebook": REPO / "notebooks" / "04_kaggle_test_local.ipynb",
        "package": "knee_mri",
        "paths": '''COMPETITION_DIR    = "f:/Kaggle/data"
CODE_DIR           = "h:/Work/ai_development_sandbox/kaggle_2026/src"   # contains knee_mri/
CHECKPOINT_GLOB    = "h:/Work/ai_development_sandbox/kaggle_2026/work/runs/cv3_20260920_*/best.pt"
WORK_DIR           = "h:/Work/ai_development_sandbox/kaggle_2026/work/knee_infer"
SUBMISSION_PATH    = "h:/Work/ai_development_sandbox/kaggle_2026/work/submission.csv"''',
        "intro_paths": '''## Ez a **lokális** változat

A repo `src/` mappájából importál és az `F:/Kaggle/data` exportból dolgozik — a publikus
teszt 3 study-ján próbálható. A Kaggle-re felmenő párja a `04_kaggle_test_online.ipynb`;
a kettő csak az 1. cella útvonalaiban és a csomagnévben tér el.''',
        "layout": '''```
f:/Kaggle/data/                       test.csv, test_series/
<repo>/src/                           knee_mri/*.py
<repo>/work/runs/cv3_*_fold[012]/     best.pt, config.yaml
```''',
    },
    "online": {
        "notebook": REPO / "notebooks" / "04_kaggle_test_online.ipynb",
        "package": "srcknee2",
        "paths": '''COMPETITION_DIR    = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
CODE_DIR           = "/kaggle/input/datasets/zoltanseress"   # contains srcknee2/
CHECKPOINT_GLOB    = "/kaggle/input/datasets/zoltanseress/checkpointsfold*best-py/best.pt"
WORK_DIR           = "/kaggle/working/knee_infer"
SUBMISSION_PATH    = "/kaggle/working/submission.csv"''',
        "intro_paths": '''## Ez a **Kaggle** változat

Csatolt datasetekből dolgozik; a lokális párja a `04_kaggle_test_local.ipynb`, a kettő
csak az 1. cella útvonalaiban és a csomagnévben tér el. Ha átnevezed a datasetjeidet,
az 1. cellában írd át az útvonalakat.''',
        "layout": '''```
/kaggle/input/competitions/rsna-knee-abnormality-detection/   test.csv, test_series/
/kaggle/input/datasets/zoltanseress/                          srcknee2/*.py
/kaggle/input/datasets/zoltanseress/checkpointsfold0best-py/  best.pt, config.yaml
/kaggle/input/datasets/zoltanseress/checkpointsfold1best-py/  best.pt, config.yaml
/kaggle/input/datasets/zoltanseress/checkpointsfold2best-py/  best.pt, config.yaml
```''',
    },
}


def render(source: str, variant: dict) -> str:
    '''Fill the per-deployment placeholders and rename the package.'''
    text = (
        source.replace("{{PATHS}}", variant["paths"])
        .replace("{{INTRO_PATHS}}", variant["intro_paths"])
        .replace("{{LAYOUT}}", variant["layout"])
    )
    return text.replace("knee_mri", variant["package"])


def build(variant: dict) -> Path:
    notebook = {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": render(source, variant).splitlines(keepends=True),
                **({"execution_count": None, "outputs": []} if kind == "code" else {}),
            }
            for kind, source in CELLS
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    nb_path = Path(variant["notebook"])
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    nb_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote {} ({} cells, package {})".format(nb_path, len(CELLS), variant["package"]))
    return nb_path



# Injected on top of the settings cell in the flat dry-run copy only.
DRY_RUN_OVERRIDES = """
# --- injected for the local dry run only -------------------------------
import os as _os
COMPETITION_DIR = _os.environ.get('KNEE_COMPETITION_DIR', COMPETITION_DIR)
CODE_DIR = _os.environ.get('KNEE_CODE_DIR', CODE_DIR)
CHECKPOINT_GLOB = _os.environ.get('KNEE_CHECKPOINT_GLOB', CHECKPOINT_GLOB)
WORK_DIR = _os.environ.get('KNEE_WORK_DIR', 'work_infer')
SUBMISSION_PATH = _os.environ.get('KNEE_SUBMISSION', 'submission.csv')
SMOKE_TEST_STUDIES = int(_os.environ.get('KNEE_SMOKE', '3'))
CHUNK_STUDIES = int(_os.environ.get('KNEE_CHUNK', '2'))
CACHE_WORKERS = int(_os.environ.get('KNEE_CACHE_WORKERS', '1'))
DATALOADER_WORKERS = 0
TIME_BUDGET_HOURS = 1.0
# ----------------------------------------------------------------------
"""


def build_flat(variant: dict, flat: Path) -> None:
    """Flat copy for dry runs: the same code cells, with overridable paths on top."""
    blocks: list[str] = []
    for kind, source in CELLS:
        if kind != "code":
            continue
        blocks.append(render(source, variant))
        if "SMOKE_TEST_STUDIES = None" in source:
            blocks.append(DRY_RUN_OVERRIDES)
    flat.parent.mkdir(parents=True, exist_ok=True)
    separator = "\n\n# " + "=" * 70 + "\n\n"
    header = (
        '"""Generated from build_04_notebook.py - do not edit."""\n\n'
        "# A notebook's __main__ has no __file__, so multiprocessing spawn does not\n"
        "# re-import it in the workers. This script must behave the same way, or every\n"
        "# spawned worker would re-run the whole pipeline.\n"
        "import sys as _sys\n"
        "if hasattr(_sys.modules['__main__'], '__file__'):\n"
        "    del _sys.modules['__main__'].__file__\n\n"
    )
    flat.write_text(header + separator.join(blocks) + "\n", encoding="utf-8")
    print("wrote {}".format(flat))


if __name__ == "__main__":
    for variant in VARIANTS.values():
        build(variant)
    if len(sys.argv) > 1:                  # optional flat copy of the local variant
        build_flat(VARIANTS["local"], Path(sys.argv[1]))
