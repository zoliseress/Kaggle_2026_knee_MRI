#!/usr/bin/env bash
# ======================================================================
#  Fix train/val felosztas, egyetlen futas (nincs fold-ciklus, nincs OOF):
#     tanitas  : train_v11.csv           (3957 study = train_v8 - a 242 uj referencia)
#     validacio: train_labeled_450_reference.csv  (450 radiologus-cimkezett study
#                = eredeti 208 + 242 uj; rsna_reference_450_package.zip)
#  Az early stopping es a best.pt is a 450-en mert macro_soft_auc alapjan
#  valaszt, ezert az ott mert ertek enyhen optimista.
#  A run vegi riport (report.html, subset_summary.csv, subset_metrics_per_class.csv)
#  kulon is kiadja a full_450, original_208 es new_242 eredmenyt
#  (paths.reference_subsets_csv = notebooks/reference_450_manifest.csv).
#  A 3-fold CV (run_cv.sh) ettol fuggetlen, valtozatlanul hasznalhato.
#
#  Hasznalat:
#     ./run_holdout.sh
#     ./run_holdout.sh E3_img320 --set data.image_size=320
#     ./run_holdout.sh --gpu 1 E3_img320 --set data.image_size=320
#     ./run_holdout.sh --gpu 0,1 E3_img320 --set data.image_size=320
#  Tobb GPU-id (--gpu 0,1): egyetlen tanitas mindket GPU-n (DDP, torchrun),
#  ugyanazzal a kiserlettel (8 study / optimizer lepes, ugyanannyi lepes /
#  epoch). GPU-nkent train.num_workers worker indul. Egy igy inditott run
#  csak ugyanennyi GPU-val folytathato (--resume).
#
#  Probafuttatas reszhalmazon (--n-studies N, barhol allhat): a tanitas a
#  split elso N studyjan fut, a validacio az elso N/2 referencia-studyn
#  (N >= 900 eseten mind a 450). A szamok igy nem vethetok ossze a teljes
#  runokkal; folytatasnal ugyanazt az N-t kell megadni.
#     ./run_holdout.sh --gpu 0,1 --n-studies 600 DDP_smoke --set train.max_epochs=4
#  Ha az elso (nem --gpu) argumentum nem "-"-vel kezdodik, az a run nevenek
#  elotagja: E3_img320_holdout450_<idobelyeg>
#
#  Folytatas (--resume <run nev>, a --gpu utan elso argumentumkent): egy
#  megszakadt run a last.pt-bol folytatodik ugyanabban a mappaban
#  (train.resume, epochhatarrol; a felbehagyott epoch elveszik):
#     ./run_holdout.sh --gpu 1 --resume E3_img320_holdout450_20261015_143722 --set data.image_size=320
#  Az extra argumentumok es a kornyezeti valtozok ugyanazok legyenek, mint az
#  eredeti inditasnal; az elteres a resume-ellenorzesen elakad.
#
#  Az elso futas elokesziti (ha meg nincsenek meg):
#     $SPLITS     - make-fixed-split (train_v11 = fold 1, 450 = fold 0)
#     $FROZEN_REF - freeze-reference --from-csv a 450-es CSV-bol
#
#  Kornyezeti valtozokkal felulirhato:
#     CONFIG     (alap: src/config.linux.yaml)
#     CONDA_ENV  (alap: kaggle_2026; ures ertek = nincs aktivalas)
#     GPU        (alap: ures; a --gpu kapcsolo felulirja)
#     DATA_ROOT  (alap: /ai-storage/Temp/SZ/kaggle_knee/data - ide kell masolni
#                 a train_v11.csv-t es a train_labeled_450_reference.csv-t)
#     TRAIN_CSV, VAL_CSV, SPLITS, FROZEN_REF (lasd lent)
#     HOLDOUT    (alap: 450 = eredeti 208 + 242 uj) a referenciahalmaz; VAL_CSV,
#                SPLITS, FROZEN_REF es a run neve (holdout<HOLDOUT>) ebbol kepzodik.
#                Regi 208-as holdout (train_v8-hoz, a korabbi runokkal osszevetheto):
#                HOLDOUT=208 TRAIN_CSV=$DATA_ROOT/train_v8.csv ./run_holdout.sh ...
#                Regi 158-as holdout (train_v4/v6-hoz):
#                HOLDOUT=158 TRAIN_CSV=$DATA_ROOT/train_v6.csv ./run_holdout.sh ...
#                A TRAIN_CSV nem tartalmazhatja a holdout studykat.
# ======================================================================

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV-kaggle_2026}"
CONFIG="${CONFIG:-$ROOT/src/config.linux.yaml}"
GPU="${GPU-}"
DATA_ROOT="${DATA_ROOT:-/ai-storage/Temp/SZ/kaggle_knee/data}"
TRAIN_CSV="${TRAIN_CSV:-$DATA_ROOT/train_v11.csv}"
HOLDOUT="${HOLDOUT:-450}"
VAL_CSV="${VAL_CSV:-$DATA_ROOT/train_labeled_${HOLDOUT}_reference.csv}"
SPLITS="${SPLITS:-$ROOT/work/splits/holdout${HOLDOUT}/splits.csv}"
FROZEN_REF="${FROZEN_REF:-$ROOT/work/labels/frozen_reference_ref${HOLDOUT}.csv}"

# --gpu N / --gpu=N es --n-studies N / --n-studies=N kiszedese (barhol
# allhat), a tobbi argumentum marad
N_STUDIES=""
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
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${ARGS[@]+"${ARGS[@]}"}"
if [[ -n "$N_STUDIES" && ! "$N_STUDIES" =~ ^[1-9][0-9]*$ ]]; then
    echo "[HIBA] A --n-studies erteke pozitiv egesz szam legyen, kapott: $N_STUDIES"
    exit 1
fi

PREFIX=""
RESUME=""
if [[ $# -gt 0 && "$1" == "--resume" ]]; then
    if [[ $# -lt 2 ]]; then
        echo "[HIBA] A --resume utan meg kell adni a run nevet, pl. B0_E3_img320_holdout450_20261015_143722"
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

# Tobb GPU (--gpu 0,1): egy tanitas DDP-vel, GPU-nkent egy processz
# (torchrun = python -m torch.distributed.run, a conda kornyezet python-javal).
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

for f in "$CONFIG" "$TRAIN_CSV" "$VAL_CSV"; do
    if [[ ! -f "$f" ]]; then
        echo "[HIBA] Nincs meg: $f"
        exit 1
    fi
done

PATHS=(
    --set "paths.train_csv=$TRAIN_CSV"
    --set "paths.splits_csv=$SPLITS"
    --set "paths.frozen_reference_csv=$FROZEN_REF"
    --set "paths.reference_csv=$VAL_CSV"
    --set "paths.exclude_from_training_csv=$VAL_CSV"
)

# --- egyszeri elokeszites --------------------------------------------
if [[ ! -f "$SPLITS" ]]; then
    echo "[INFO] Fix split keszitese: $SPLITS"
    if ! python -m knee_mri.cli make-fixed-split --config "$CONFIG" "${PATHS[@]}" \
            --validation-csv "$VAL_CSV"; then
        echo "[HIBA] A make-fixed-split nem sikerult."
        exit 1
    fi
fi
if [[ ! -f "$FROZEN_REF" ]]; then
    echo "[INFO] Validacios referencia fagyasztasa: $FROZEN_REF"
    if ! python -m knee_mri.cli freeze-reference --config "$CONFIG" "${PATHS[@]}" \
            --from-csv "$VAL_CSV" --out "$FROZEN_REF"; then
        echo "[HIBA] A freeze-reference nem sikerult."
        exit 1
    fi
fi

# --- a split illeszkedik-e a TRAIN_CSV-hez es a VAL_CSV-hez ---------
# (pl. egy korabbi shellbol ottmaradt TRAIN_CSV: holdout studyk a tanito CSV-ben)
if ! python -m knee_mri.cli check-fixed-split --config "$CONFIG" "${PATHS[@]}" \
        --validation-csv "$VAL_CSV"; then
    echo "[HIBA] A split nem illik a TRAIN_CSV / VAL_CSV parhoz. Ellenorizd a"
    echo "       TRAIN_CSV, HOLDOUT, VAL_CSV, SPLITS kornyezeti valtozokat."
    exit 1
fi

# A run a tanito CSV szerinti almappaba kerul: work/runs/<train_csv stem>/
if ! OUT="$(python -m knee_mri.cli output-dir --config "$CONFIG" "${PATHS[@]}" "${EXTRA[@]}")"; then
    echo "$OUT"
    echo "[HIBA] Nem sikerult meghatarozni a kimeneti mappat (output-dir)."
    exit 1
fi
RUNS_DIR="$(printf '%s\n' "$OUT" | tail -n 1)"

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -n "$RESUME" ]]; then
    NAME="$RESUME"
else
    NAME="${PREFIX}holdout${HOLDOUT}_${STAMP}"
fi
RUN="$RUNS_DIR/$NAME"

START=()
if [[ -n "$N_STUDIES" ]]; then
    START+=(--n-studies "$N_STUDIES")
fi
if [[ -n "$RESUME" ]]; then
    if [[ -f "$RUN/run_summary.json" ]]; then
        echo "[INFO] A run mar kesz (run_summary.json), nincs mit folytatni: $RUN"
        exit 0
    fi
    if [[ ! -f "$RUN/last.pt" ]]; then
        echo "[HIBA] --resume: nincs last.pt: $RUN/last.pt"
        echo "       Ellenorizd a nevet, es hogy a TRAIN_CSV ugyanaz-e, mint az eredeti inditasnal."
        exit 1
    fi
    START+=(--set "train.resume=$RUN/last.pt")
fi

echo "======================================================================"
echo " Run        : $NAME"
[[ -n "$RESUME" ]] && echo " Mod        : folytatas a last.pt-bol (epochhatarrol)"
echo " Tanitas    : $TRAIN_CSV"
echo " Validacio  : $VAL_CSV"
echo " Config     : $CONFIG"
echo " GPU        : ${CUDA_VISIBLE_DEVICES:-(alapertelmezett)}$( (( NPROC > 1 )) && echo "  ($NPROC processz, DDP)")"
echo " Extra args : ${EXTRA[*]:-}"
[[ -n "$N_STUDIES" ]] && echo " Reszhalmaz : --n-studies $N_STUDIES (proba, nem teljes run)"
echo " Kimenet    : $RUN"
echo "======================================================================"

if ! "${TRAIN_LAUNCH[@]}" "$ROOT/src/train.py" --mode fold --config "$CONFIG" \
        --set "split.fold=0" "${PATHS[@]}" --name "$NAME" "${EXTRA[@]}" "${START[@]+"${START[@]}"}"; then
    echo
    echo "[HIBA] A tanitas hibaval leallt. Log: $RUN/run.log"
    echo "       Folytatas: ./run_holdout.sh ${GPU:+--gpu $GPU }${N_STUDIES:+--n-studies $N_STUDIES }--resume $NAME <ugyanazok az extra argumentumok>"
    exit 1
fi

echo
echo "======================================================================"
echo " KESZ: $NAME   vege: $(date '+%F %T')"
echo "======================================================================"
