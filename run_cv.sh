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
#  Minden ide irt extra argumentum MINDHAROM foldra ervenyes.
#  Ha az elso argumentum nem "-"-vel kezdodik, az a run-group elotagja:
#     V2soft_img320_cv3_<idobelyeg>_fold<N>, ..._oof
#
#  Kornyezeti valtozokkal felulirhato:
#     CONFIG     (alap: src/config.linux.yaml)
#     CONDA_ENV  (alap: kaggle_2026; ures ertek = nincs aktivalas)
#
#  Elofeltetel: a work/splits/splits.csv mar letezik es ugyanazzal az
#  n_folds ertekkel keszult (make-splits), mint amennyit itt futtatunk.
# ======================================================================

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NFOLDS=3
CONDA_ENV="${CONDA_ENV-kaggle_2026}"
CONFIG="${CONFIG:-$ROOT/src/config.linux.yaml}"

PREFIX=""
if [[ $# -gt 0 && "$1" != -* ]]; then
    PREFIX="${1}_"
    shift
fi
EXTRA=("$@")

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

STAMP="$(date +%Y%m%d_%H%M%S)"
GROUP="${PREFIX}cv${NFOLDS}_${STAMP}"
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
echo " Foldok     : 0 .. $LAST"
echo " Config     : $CONFIG"
echo " Extra args : ${EXTRA[*]:-}"
echo " Kimenet    : $ROOT/work/runs/${GROUP}_fold<N>"
echo "======================================================================"

# --- foldok egymas utan ---------------------------------------------
for ((F = 0; F <= LAST; F++)); do
    NAME="${GROUP}_fold${F}"
    echo
    echo "----------------------------------------------------------------"
    echo " FOLD $F / $LAST   ($NAME)   start: $(date '+%F %T')"
    echo "----------------------------------------------------------------"
    if ! python "$ROOT/src/train.py" --mode fold --config "$CONFIG" \
            --set "split.fold=$F" --name "$NAME" "${EXTRA[@]}"; then
        echo
        echo "[HIBA] A $F. fold hibaval leallt. A script nem folytatja."
        echo "       Log: $ROOT/work/runs/$NAME/run.log"
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
OOFDIR="$ROOT/work/runs/${GROUP}_oof"
mkdir -p "$OOFDIR"
PREDS=()
for ((F = 0; F <= LAST; F++)); do
    PREDS+=("$ROOT/work/runs/${GROUP}_fold${F}/validation_predictions.csv")
done
if ! python -m knee_mri.cli merge-oof --config "$CONFIG" "${PREDS[@]}" \
        --out "$OOFDIR/oof_predictions.csv"; then
    echo "[FIGYELEM] Az OOF merge nem sikerult, de mindharom fold lefutott."
    echo "           A per-fold eredmenyek megvannak a run mappakban."
fi

echo
echo "======================================================================"
echo " KESZ. Mind a $NFOLDS fold lefutott: $GROUP"
echo " OOF        : $OOFDIR"
echo "======================================================================"
