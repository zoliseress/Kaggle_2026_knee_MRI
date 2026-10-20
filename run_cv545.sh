#!/usr/bin/env bash
# ======================================================================
#  3-fold CV a 545-os radiologus-referencian (rsna_reference_545_3fold_soft.zip):
#     foldonkent egy fix holdout run (mint a run_holdout.sh), N = 0, 1, 2:
#       tanitas  : train_v12_ref545_<LABEL_MODE>_fold<N>.csv
#                  = train_v11 az OSSZES 545 referencia-study nelkul (3862 LLM soft sor)
#                  + a masik ket fold radiologus-cimkei (363/364 sor; soft: 0/.15/.85/1)
#       validacio: ref545_fold<N>_val.csv (a fold 182/181 studyja, BINARIS radiologus-cimke)
#     a vegen merge-oof --fold-assignments: pooled OOF az 545-on, foldonkenti
#     makro AUC (fold_summary.csv) es full_545 / original_208 / new_242 / new_95
#     bontas (oof_subset_summary.csv).
#  A bemeneteket a notebooks/build_ref545_cv.py kesziti (egyszer):
#     python -I notebooks/build_ref545_cv.py
#  Az early stopping / best.pt a fold sajat validaciojan valaszt: fejlesztesi CV,
#  az OOF ertek enyhen optimista. A Kaggle-tesztre a 3 foldmodell atlagolhato.
#
#  Hasznalat:
#     ./run_cv545.sh --gpu 2,3 E3_img320 --set data.image_size=320
#     LABEL_MODE=binary ./run_cv545.sh --gpu 2,3 E3_img320 --set data.image_size=320   # kontroll
#  Probafuttatas (--n-studies N): a tanitas az elso N studyn, a validacio
#  foldonkent az elso N/2 studyn fut (N >= 364 eseten a teljes fold, ekkor az
#  OOF merge is lefut). max_epochs > warmup_epochs (2) kell:
#     RUNS_DIR=work/runs/smoke_v12 ./run_cv545.sh --gpu 1 --n-studies 600 smoke --set train.max_epochs=3
#  Parhuzamos foldok (egy fold / GPU; 3 GPU-n DDP nem megy, mert az
#  accumulation_steps=8 nem oszthato 3-mal): --folds N valasztja a foldot,
#  a GROUP kornyezeti valtozo adja a kozos run-group nevet. Az OOF merge
#  akkor fut, amikor mindharom fold kesz (az utoljara vegzo futas csinalja):
#     G=V2S_cv545soft_$(date +%Y%m%d_%H%M%S)
#     GROUP=$G ./run_cv545.sh --gpu 1 --folds 0 ...   (es --gpu 2 --folds 1, --gpu 3 --folds 2)
#  Folytatas (--resume <run-group>, a --gpu utan elso argumentumkent): a kesz
#  foldok kimaradnak, a felbehagyott fold a last.pt-bol folytatodik:
#     ./run_cv545.sh --gpu 2,3 --resume E3_img320_cv545soft_20261020_101500 --set data.image_size=320
#
#  Kornyezeti valtozok:
#     CONFIG     (alap: src/config.linux.yaml)
#     CONDA_ENV  (alap: kaggle_2026; ures ertek = nincs aktivalas)
#     GPU        (alap: ures; a --gpu kapcsolo felulirja)
#     DATA_ROOT  (alap: /ai-storage/Temp/SZ/kaggle_knee/data)
#     LABEL_MODE (alap: soft) soft | binary: a radiologus-tanitocimkek fajtaja
#     MANIFEST   (alap: notebooks/reference_545_manifest.csv) subset + fold
#     GROUP      (alap: ures = <prefix>_cv545<LABEL_MODE>_<idobelyeg>) fix run-group nev
#     RUNS_DIR   (alap: work/runs/train_v12)
# ======================================================================

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV-kaggle_2026}"
CONFIG="${CONFIG:-$ROOT/src/config.linux.yaml}"
GPU="${GPU-}"
DATA_ROOT="${DATA_ROOT:-/ai-storage/Temp/SZ/kaggle_knee/data}"
LABEL_MODE="${LABEL_MODE:-soft}"
MANIFEST="${MANIFEST:-$ROOT/notebooks/reference_545_manifest.csv}"
NFOLDS=3
LAST=$((NFOLDS - 1))
RUNS_DIR="${RUNS_DIR:-$ROOT/work/runs/train_v12}"
GROUP="${GROUP-}"

if [[ "$LABEL_MODE" != "soft" && "$LABEL_MODE" != "binary" ]]; then
    echo "[HIBA] LABEL_MODE soft vagy binary legyen, kapott: $LABEL_MODE"
    exit 1
fi

# --gpu N / --gpu=N es --n-studies N / --n-studies=N kiszedese (barhol
# allhat), a tobbi argumentum marad
N_STUDIES=""
FOLDS="0,1,2"
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-studies)
            if [[ $# -lt 2 ]]; then
                echo "[HIBA] A --n-studies utan meg kell adni a studyk szamat."
                exit 1
            fi
            N_STUDIES="$2"
            shift 2
            ;;
        --n-studies=*)
            N_STUDIES="${1#--n-studies=}"
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
        --folds)
            if [[ $# -lt 2 ]]; then
                echo "[HIBA] A --folds utan meg kell adni a foldokat, pl. 0 vagy 1,2"
                exit 1
            fi
            FOLDS="$2"
            shift 2
            ;;
        --folds=*)
            FOLDS="${1#--folds=}"
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${ARGS[@]+"${ARGS[@]}"}"
if [[ ! "$FOLDS" =~ ^[0-2](,[0-2])*$ ]]; then
    echo "[HIBA] A --folds 0..2 kozotti foldok vesszovel, kapott: $FOLDS"
    exit 1
fi
IFS=',' read -ra FOLD_LIST <<< "$FOLDS"
if [[ -n "$N_STUDIES" && ! "$N_STUDIES" =~ ^[1-9][0-9]*$ ]]; then
    echo "[HIBA] A --n-studies erteke pozitiv egesz szam legyen, kapott: $N_STUDIES"
    exit 1
fi

PREFIX=""
RESUME=""
if [[ $# -gt 0 && "$1" == "--resume" ]]; then
    if [[ $# -lt 2 ]]; then
        echo "[HIBA] A --resume utan meg kell adni a run-group nevet, pl. E3_img320_cv545soft_20261020_101500"
        exit 1
    fi
    RESUME="$2"
    shift 2
elif [[ $# -gt 0 && "$1" != -* ]]; then
    PREFIX="${1}_"
    shift
fi
EXTRA=("$@")

if [[ -n "$GPU" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

NPROC=1
if [[ -n "$GPU" ]]; then
    IFS=',' read -ra GPU_IDS <<< "$GPU"
    NPROC=${#GPU_IDS[@]}
fi
if (( NPROC > 1 )); then
    TRAIN_LAUNCH=(python -m torch.distributed.run --standalone --nproc_per_node="$NPROC")
else
    TRAIN_LAUNCH=(python)
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

COMMON=(
    --set "paths.reference_subsets_csv=$MANIFEST"
    --set "paths.output_dir=$RUNS_DIR"
)

# A fold N futasahoz tartozo utvonalak (FOLD_PATHS, TRAIN_CSV, VAL_CSV)
fold_paths() {
    local f="$1"
    TRAIN_CSV="$DATA_ROOT/train_v12_ref545_${LABEL_MODE}_fold${f}.csv"
    VAL_CSV="$DATA_ROOT/ref545_fold${f}_val.csv"
    FOLD_PATHS=(
        "${COMMON[@]}"
        --set "paths.train_csv=$TRAIN_CSV"
        --set "paths.splits_csv=$ROOT/work/splits/ref545_${LABEL_MODE}_fold${f}/splits.csv"
        --set "paths.frozen_reference_csv=$ROOT/work/labels/frozen_reference_ref545_fold${f}.csv"
        --set "paths.reference_csv=$VAL_CSV"
        --set "paths.exclude_from_training_csv=$VAL_CSV"
    )
}

# --- egyszeri elokeszites + ellenorzes minden foldra (a tanitas elott) ---
for f in "$CONFIG" "$MANIFEST"; do
    if [[ ! -f "$f" ]]; then
        echo "[HIBA] Nincs meg: $f"
        exit 1
    fi
done
for ((F = 0; F <= LAST; F++)); do
    fold_paths "$F"
    for f in "$TRAIN_CSV" "$VAL_CSV"; do
        if [[ ! -f "$f" ]]; then
            echo "[HIBA] Nincs meg: $f"
            echo "       Elobb: python -I notebooks/build_ref545_cv.py"
            exit 1
        fi
    done
    SPLITS="$ROOT/work/splits/ref545_${LABEL_MODE}_fold${F}/splits.csv"
    FROZEN_REF="$ROOT/work/labels/frozen_reference_ref545_fold${F}.csv"
    if [[ ! -f "$SPLITS" ]]; then
        echo "[INFO] Fix split keszitese (fold $F): $SPLITS"
        if ! python -m knee_mri.cli make-fixed-split --config "$CONFIG" "${FOLD_PATHS[@]}" \
                --validation-csv "$VAL_CSV"; then
            echo "[HIBA] A make-fixed-split nem sikerult (fold $F)."
            exit 1
        fi
    fi
    if [[ ! -f "$FROZEN_REF" ]]; then
        echo "[INFO] Validacios referencia fagyasztasa (fold $F): $FROZEN_REF"
        if ! python -m knee_mri.cli freeze-reference --config "$CONFIG" "${FOLD_PATHS[@]}" \
                --from-csv "$VAL_CSV" --out "$FROZEN_REF"; then
            echo "[HIBA] A freeze-reference nem sikerult (fold $F)."
            exit 1
        fi
    fi
    if ! python -m knee_mri.cli check-fixed-split --config "$CONFIG" "${FOLD_PATHS[@]}" \
            --validation-csv "$VAL_CSV"; then
        echo "[HIBA] A fold $F split nem illik a TRAIN_CSV / VAL_CSV parhoz."
        exit 1
    fi
done

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -n "$RESUME" ]]; then
    GROUP="$RESUME"
    if ! compgen -G "$RUNS_DIR/${GROUP}_fold[0-2]" >/dev/null; then
        echo "[HIBA] --resume: nincs ilyen run-group: $RUNS_DIR/${GROUP}_fold<N>"
        echo "       Ellenorizd a nevet, a RUNS_DIR-t es a LABEL_MODE-ot (most: $LABEL_MODE)."
        exit 1
    fi
elif [[ -z "$GROUP" ]]; then
    GROUP="${PREFIX}cv545${LABEL_MODE}_${STAMP}"
fi

fail() {
    echo
    echo "======================================================================"
    echo " MEGSZAKADT: $GROUP"
    echo "======================================================================"
    exit 1
}

echo "======================================================================"
echo " Run group  : $GROUP"
[[ -n "$RESUME" ]] && echo " Mod        : folytatas (kesz fold kihagyva, last.pt-bol folytatva)"
echo " Cimkek     : radiologus $LABEL_MODE tanitas, binaris validacio"
echo " Tanitas    : $DATA_ROOT/train_v12_ref545_${LABEL_MODE}_fold<N>.csv"
echo " Validacio  : $DATA_ROOT/ref545_fold<N>_val.csv"
echo " Config     : $CONFIG"
echo " GPU        : ${CUDA_VISIBLE_DEVICES:-(alapertelmezett)}$( (( NPROC > 1 )) && echo "  ($NPROC processz, DDP)")"
echo " Extra args : ${EXTRA[*]:-}"
[[ -n "$N_STUDIES" ]] && echo " Reszhalmaz : --n-studies $N_STUDIES (proba, nem teljes run)"
echo " Foldok     : $FOLDS"
echo " Kimenet    : $RUNS_DIR/${GROUP}_fold<N>"
echo "======================================================================"

# --- foldok egymas utan ---------------------------------------------
for F in "${FOLD_LIST[@]}"; do
    fold_paths "$F"
    NAME="${GROUP}_fold${F}"
    RUN="$RUNS_DIR/$NAME"
    START=()
    if [[ -n "$N_STUDIES" ]]; then
        START+=(--n-studies "$N_STUDIES")
    fi
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
            START+=(--set "train.resume=$RUN/last.pt")
        fi
    fi
    if ! "${TRAIN_LAUNCH[@]}" "$ROOT/src/train.py" --mode fold --config "$CONFIG" \
            --set "split.fold=0" "${FOLD_PATHS[@]}" --name "$NAME" "${EXTRA[@]}" "${START[@]+"${START[@]}"}"; then
        echo
        echo "[HIBA] A $F. fold hibaval leallt. A script nem folytatja."
        echo "       Log: $RUN/run.log"
        echo "       Folytatas: ${LABEL_MODE:+LABEL_MODE=$LABEL_MODE }./run_cv545.sh ${GPU:+--gpu $GPU }--folds $FOLDS ${N_STUDIES:+--n-studies $N_STUDIES }--resume $GROUP <ugyanazok az extra argumentumok>"
        fail
    fi
    echo " FOLD $F kesz.  vege: $(date '+%F %T')"
done

# --- OOF merge (foldok a manifest fold oszlopabol) -------------------
echo
echo "----------------------------------------------------------------"
echo " OOF merge"
echo "----------------------------------------------------------------"
OOFDIR="$RUNS_DIR/${GROUP}_oof"
PREDS=()
for ((F = 0; F <= LAST; F++)); do
    if [[ ! -f "$RUNS_DIR/${GROUP}_fold${F}/run_summary.json" ]]; then
        echo " A fold $F meg nincs kesz ($RUNS_DIR/${GROUP}_fold${F}); az OOF merge-et"
        echo " az utoljara vegzo fold futtatja, vagy kezzel: --resume $GROUP"
        exit 0
    fi
    PREDS+=("$RUNS_DIR/${GROUP}_fold${F}/validation_predictions.csv")
done
mkdir -p "$OOFDIR"
if ! python -m knee_mri.cli merge-oof --config "$CONFIG" "${COMMON[@]}" "${EXTRA[@]}" "${PREDS[@]}" \
        --fold-assignments "$MANIFEST" --out "$OOFDIR/oof_predictions.csv"; then
    echo "[FIGYELEM] Az OOF merge nem sikerult, de mindharom fold lefutott."
    echo "           A per-fold eredmenyek megvannak a run mappakban."
fi

echo
echo "======================================================================"
echo " KESZ. Mind a $NFOLDS fold lefutott: $GROUP"
echo " OOF        : $OOFDIR  (oof_summary.json, fold_summary.csv, oof_subset_summary.csv)"
echo "======================================================================"
