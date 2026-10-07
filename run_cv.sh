#!/usr/bin/env bash
# ======================================================================
#  3-fold keresztvalidacio: a foldokat (split.fold = 0,1,2) egymas utan
#  futtatja, majd osszefesuli a per-fold predikciokat OOF-fa.
#  (A run_cv.bat Linux-os megfeleloje.)
#
#  Hasznalat:
#     ./run_cv.sh
#     ./run_cv.sh --set data.image_size=288 --set train.max_epochs=10
#     ./run_cv.sh V2soft_img320 --set data.image_size=320
#     ./run_cv.sh --gpu 1 V2soft_img320 --set data.image_size=320
#  Minden ide irt extra argumentum MINDHAROM foldra ervenyes.
#  Ha az elso (nem --gpu) argumentum nem "-"-vel kezdodik, az a run-group
#  elotagja:
#     V2soft_img320_cv3_<idobelyeg>_fold<N>, ..._oof
#  A --gpu N (vagy --gpu=N) barhol allhat; CUDA_VISIBLE_DEVICES=N-t allit,
#  igy a python oldalon a kivalasztott GPU lesz a cuda:0.
#
#  Folytatas: --resume <run-group> (vagy --resume=<run-group>, barhol allhat)
#     ./run_cv.sh --gpu 3 --resume V2S_E3_img320_train_v3_cv3_20261001_111938 --set ...
#  Egy megszakadt run-groupot visz tovabb ugyanazzal a nevvel. Foldonkent:
#     run_summary.json van  -> kesz, kihagyja
#     csak last.pt van      -> onnan folytatja (train.resume, epochhatarrol)
#     egyik sincs           -> elolrol inditja
#  majd ujra osszefesuli az OOF-ot. Run-group elotag mellette nem adhato meg.
#  Az extra argumentumok ugyanazok legyenek, mint az eredeti inditasnal: a
#  folytatott foldnal ezt a resume-ellenorzes ki is kenyszeriti, egy elolrol
#  indulo foldnal nem.
#
#  Kornyezeti valtozokkal felulirhato:
#     CONFIG     (alap: src/config.linux.yaml)
#     CONDA_ENV  (alap: kaggle_2026; ures ertek = nincs aktivalas)
#     GPU        (alap: ures = nem nyul a CUDA_VISIBLE_DEVICES-hez;
#                 a --gpu kapcsolo felulirja)
#     NFOLDS     (alap: 3)
#     TRAIN_CSV  (alap: a config paths.train_csv-je)
#     SPLITS     (alap: work/splits/cv<NFOLDS>_<train_csv stem>/splits.csv)
#
#  Kimenet: work/runs/<train_csv stem>/ (pl. work/runs/train_v3/).
#
#  Ha a SPLITS meg nincs meg, a script legyartja (make-splits a TRAIN_CSV
#  studyjaibol, NFOLDS folddal), majd minden inditasnal ellenorzi
#  (check-splits): a fold-szam, a TRAIN_CSV es a kizarasi lista
#  (paths.exclude_from_training_csv, alap: a ref208) egyezzen.
#  Teacher CV (3 fold = alap, ref208 nelkul, a train_v8.csv-bol):
#     ./run_cv.sh --gpu 0 B0_teacher --set data.image_size=320
#  Regi, 4407 studys work/splits/splits.csv-vel indult run-group folytatasa
#  (a ref208 abban tanitott, ezert a kizarast is ki kell kapcsolni):
#     SPLITS=work/splits/splits.csv ./run_cv.sh --resume ... --set paths.exclude_from_training_csv=
# ======================================================================

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NFOLDS="${NFOLDS:-3}"
CONDA_ENV="${CONDA_ENV-kaggle_2026}"
CONFIG="${CONFIG:-$ROOT/src/config.linux.yaml}"

GPU="${GPU-}"
RESUME=""

# --gpu N / --gpu=N es --resume G / --resume=G kiszedese (barhol allhat),
# a tobbi argumentum marad
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            if [[ $# -lt 2 || -z "$2" ]]; then
                echo "[HIBA] A --resume utan meg kell adni a run-group nevet."
                exit 1
            fi
            RESUME="$2"
            shift 2
            ;;
        --resume=*)
            RESUME="${1#--resume=}"
            shift
            ;;
        --gpu)
            if [[ $# -lt 2 ]]; then
                echo "[HIBA] A --gpu utan meg kell adni a GPU id-t."
                exit 1
            fi
            GPU="$2"
            shift 2
            ;;
        --gpu=*)
            GPU="${1#--gpu=}"
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

PREFIX=""
if [[ $# -gt 0 && "$1" != -* ]]; then
    if [[ -n "$RESUME" ]]; then
        echo "[HIBA] --resume mellett nem adhato run-group elotag ($1); a nev a --resume-bol jon."
        exit 1
    fi
    PREFIX="${1}_"
    shift
fi
EXTRA=("$@")
# A TRAIN_CSV a tobbi argumentum ele kerul, igy egy kezzel irt --set paths.train_csv felulirja.
if [[ -n "${TRAIN_CSV:-}" ]]; then
    EXTRA=(--set "paths.train_csv=$TRAIN_CSV" "${EXTRA[@]+"${EXTRA[@]}"}")
fi

if [[ -n "$GPU" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# --- conda kornyezet aktivalasa -------------------------------------
if [[ -n "$CONDA_ENV" && "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        if ! conda activate "$CONDA_ENV"; then
            echo "[HIBA] Nem sikerult aktivalni a conda kornyezetet: $CONDA_ENV"
            exit 1
        fi
    else
        echo "[FIGYELEM] Nincs conda a PATH-on."
        echo "           Az eppen aktiv python-t hasznalom: $(command -v python)"
    fi
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "[HIBA] Nincs meg a config: $CONFIG"
    exit 1
fi

# A runok a tanito CSV szerinti almappaba kerulnek: work/runs/<train_csv stem>/
if ! OUT="$(python -m knee_mri.cli output-dir --config "$CONFIG" "${EXTRA[@]+"${EXTRA[@]}"}")"; then
    echo "$OUT"
    echo "[HIBA] Nem sikerult meghatarozni a kimeneti mappat (output-dir)."
    exit 1
fi
RUNS_DIR="$(printf '%s\n' "$OUT" | tail -n 1)"
if ! OUT="$(python -m knee_mri.cli output-dir --train-csv --config "$CONFIG" "${EXTRA[@]+"${EXTRA[@]}"}")"; then
    echo "$OUT"
    echo "[HIBA] Nem sikerult meghatarozni a tanito CSV-t (output-dir --train-csv)."
    exit 1
fi
TRAIN_CSV_RESOLVED="$(printf '%s\n' "$OUT" | tail -n 1)"
TRAIN_STEM="$(basename "$TRAIN_CSV_RESOLVED")"
TRAIN_STEM="${TRAIN_STEM%.*}"

# --- CV split: legyartas (ha nincs) es ellenorzes --------------------
SPLITS="${SPLITS:-$ROOT/work/splits/cv${NFOLDS}_${TRAIN_STEM}/splits.csv}"
EXTRA=(--set "paths.splits_csv=$SPLITS" "${EXTRA[@]+"${EXTRA[@]}"}")
if [[ ! -f "$SPLITS" ]]; then
    if [[ -n "$RESUME" ]]; then
        echo "[HIBA] --resume, de nincs meg a split: $SPLITS"
        echo "       A folytatashoz ugyanaz a split kell, mint az eredeti inditasnal (SPLITS)."
        exit 1
    fi
    echo "[INFO] CV split keszitese: $SPLITS  ($NFOLDS fold, $TRAIN_CSV_RESOLVED)"
    if ! python -m knee_mri.cli make-splits --config "$CONFIG" "${EXTRA[@]}" --set "split.n_folds=$NFOLDS"; then
        echo "[HIBA] A make-splits nem sikerult."
        exit 1
    fi
fi
if ! python -m knee_mri.cli check-splits --config "$CONFIG" "${EXTRA[@]}" --n-folds "$NFOLDS"; then
    echo "[HIBA] A split nem illik a futashoz: $SPLITS"
    echo "       Ellenorizd az NFOLDS, TRAIN_CSV, SPLITS kornyezeti valtozokat."
    exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -n "$RESUME" ]]; then
    GROUP="$RESUME"
    if [[ ! -d "$RUNS_DIR/${GROUP}_fold0" ]]; then
        echo "[HIBA] --resume: nincs ilyen run-group: $RUNS_DIR/${GROUP}_fold0"
        echo "       Ellenorizd a nevet, es hogy a CONFIG / train_csv ugyanaz-e, mint az eredeti inditasnal."
        exit 1
    fi
else
    GROUP="${PREFIX}cv${NFOLDS}_${STAMP}"
fi
LAST=$((NFOLDS - 1))

fail() {
    echo
    echo "======================================================================"
    echo " MEGSZAKADT: $GROUP"
    echo "======================================================================"
    exit 1
}

echo "======================================================================"
echo " Run group  : $GROUP"
if [[ -n "$RESUME" ]]; then
    echo " Mod        : folytatas (kesz fold kihagyva, last.pt-bol folytatva)"
fi
echo " Foldok     : 0 .. $LAST"
echo " Config     : $CONFIG"
echo " Train CSV  : $TRAIN_CSV_RESOLVED"
echo " Split      : $SPLITS"
echo " GPU        : ${CUDA_VISIBLE_DEVICES:-(alapertelmezett)}"
echo " Extra args : ${EXTRA[*]:-}"
echo " Kimenet    : $RUNS_DIR/${GROUP}_fold<N>"
echo "======================================================================"

# --- foldok egymas utan ---------------------------------------------
for ((F = 0; F <= LAST; F++)); do
    NAME="${GROUP}_fold${F}"
    RUN="$RUNS_DIR/$NAME"
    START=()
    echo
    echo "----------------------------------------------------------------"
    echo " FOLD $F / $LAST   ($NAME)   start: $(date '+%F %T')"
    echo "----------------------------------------------------------------"
    if [[ -n "$RESUME" ]]; then
        if [[ -f "$RUN/run_summary.json" ]]; then
            echo " FOLD $F mar kesz (run_summary.json), kihagyom."
            continue
        fi
        if [[ -f "$RUN/last.pt" ]]; then
            echo " Folytatas: $RUN/last.pt"
            START=(--set "train.resume=$RUN/last.pt")
        fi
    fi
    if ! python "$ROOT/src/train.py" --mode fold --config "$CONFIG" \
            --set "split.fold=$F" --name "$NAME" "${EXTRA[@]}" "${START[@]+"${START[@]}"}"; then
        echo
        echo "[HIBA] A $F. fold hibaval leallt. A script nem folytatja."
        echo "       Log: $RUN/run.log"
        echo "       Folytatas: ./run_cv.sh --resume $GROUP <ugyanazok az extra argumentumok>"
        fail
    fi
    echo " FOLD $F kesz.  vege: $(date '+%F %T')"
done

# --- OOF merge -------------------------------------------------------
echo
echo "----------------------------------------------------------------"
echo " OOF merge"
echo "----------------------------------------------------------------"
# A merge-oof a --out melle FIX neven irja az oof_metrics_per_class.csv-t
# es az oof_summary.json-t, ezert minden run-group sajat alkonyvtarba kerul.
OOFDIR="$RUNS_DIR/${GROUP}_oof"
mkdir -p "$OOFDIR"
PREDS=()
for ((F = 0; F <= LAST; F++)); do
    PREDS+=("$RUNS_DIR/${GROUP}_fold${F}/validation_predictions.csv")
done
if ! python -m knee_mri.cli merge-oof --config "$CONFIG" "${EXTRA[@]}" "${PREDS[@]}" \
        --out "$OOFDIR/oof_predictions.csv"; then
    echo "[FIGYELEM] Az OOF merge nem sikerult, de mindharom fold lefutott."
    echo "           A per-fold eredmenyek megvannak a run mappakban."
fi

echo
echo "======================================================================"
echo " KESZ. Mind a $NFOLDS fold lefutott: $GROUP"
echo " OOF        : $OOFDIR"
echo "======================================================================"
