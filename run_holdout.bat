@echo off
setlocal enabledelayedexpansion
REM ======================================================================
REM  Fix train/val felosztas, egyetlen futas (nincs fold-ciklus, nincs OOF):
REM     tanitas  : train_v4.csv            (4249 study)
REM     validacio: train_labeled_158_reference.csv  (158 radiologus-cimkezett study)
REM  Az early stopping es a best.pt is a 158-on mert macro_soft_auc alapjan
REM  valaszt, ezert az ott mert ertek enyhen optimista.
REM  A 3-fold CV (run_cv.bat) ettol fuggetlen, valtozatlanul hasznalhato.
REM
REM  Hasznalat:
REM     run_holdout.bat
REM     run_holdout.bat E3_img320 --set data.image_size=320
REM  Ha az elso argumentum nem "-"-vel kezdodik, az a run nevenek elotagja:
REM     E3_img320_holdout158_<idobelyeg>
REM
REM  Az elso futas elokesziti (ha meg nincsenek meg):
REM     SPLITS     - make-fixed-split (train_v4 = fold 1, 158 = fold 0)
REM     FROZEN_REF - freeze-reference --from-csv a 158-as CSV-bol
REM  DATA_ROOT, TRAIN_CSV, VAL_CSV, SPLITS, FROZEN_REF kornyezeti
REM  valtozokkal felulirhato.
REM ======================================================================

set "ROOT=%~dp0"
set "CONDA_ENV=kaggle_2026"
set "CONDA_ROOT=%LOCALAPPDATA%\miniconda3"
if not defined DATA_ROOT set "DATA_ROOT=F:\Kaggle\data"
if not defined TRAIN_CSV set "TRAIN_CSV=%DATA_ROOT%\train_v5.csv"
if not defined VAL_CSV set "VAL_CSV=%DATA_ROOT%\train_labeled_158_reference.csv"
if not defined SPLITS set "SPLITS=%ROOT%work\splits\holdout158\splits.csv"
if not defined FROZEN_REF set "FROZEN_REF=%ROOT%work\labels\frozen_reference_ref158.csv"

set "EXTRA=%*"
set "PREFIX="
set "FIRST=%~1"
if defined FIRST if not "!FIRST:~0,1!"=="-" (
    set "PREFIX=!FIRST!_"
    set "EXTRA=!EXTRA:*%1=!"
)
cd /d "%ROOT%"
set "PYTHONPATH=%ROOT%src;%PYTHONPATH%"

REM --- conda kornyezet aktivalasa -------------------------------------
if exist "%CONDA_ROOT%\Scripts\activate.bat" (
    call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ENV%"
    if errorlevel 1 (
        echo [HIBA] Nem sikerult aktivalni a conda kornyezetet: %CONDA_ENV%
        goto :fail
    )
) else (
    echo [FIGYELEM] Nincs meg: %CONDA_ROOT%\Scripts\activate.bat
    echo            Az eppen aktiv python-t hasznalom.
)

if not exist "%TRAIN_CSV%" (
    echo [HIBA] Nincs meg: %TRAIN_CSV%
    goto :fail
)
if not exist "%VAL_CSV%" (
    echo [HIBA] Nincs meg: %VAL_CSV%
    goto :fail
)

set PATHS=--set "paths.train_csv=%TRAIN_CSV%" --set "paths.splits_csv=%SPLITS%" --set "paths.frozen_reference_csv=%FROZEN_REF%" --set "paths.reference_csv=%VAL_CSV%"

REM --- egyszeri elokeszites --------------------------------------------
if not exist "%SPLITS%" (
    echo [INFO] Fix split keszitese: %SPLITS%
    python -m knee_mri.cli make-fixed-split %PATHS% --validation-csv "%VAL_CSV%"
    if errorlevel 1 (
        echo [HIBA] A make-fixed-split nem sikerult.
        goto :fail
    )
)
if not exist "%FROZEN_REF%" (
    echo [INFO] Validacios referencia fagyasztasa: %FROZEN_REF%
    python -m knee_mri.cli freeze-reference %PATHS% --from-csv "%VAL_CSV%" --out "%FROZEN_REF%"
    if errorlevel 1 (
        echo [HIBA] A freeze-reference nem sikerult.
        goto :fail
    )
)

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%i"
set "NAME=%PREFIX%holdout158_%STAMP%"

echo ======================================================================
echo  Run        : %NAME%
echo  Tanitas    : %TRAIN_CSV%
echo  Validacio  : %VAL_CSV%
echo  Extra args : %EXTRA%
echo  Kimenet    : %ROOT%work\runs\%NAME%
echo ======================================================================

python "%ROOT%src\train.py" --mode fold --set split.fold=0 %PATHS% --name "%NAME%" %EXTRA%
if errorlevel 1 (
    echo.
    echo [HIBA] A tanitas hibaval leallt. Log: %ROOT%work\runs\%NAME%\run.log
    goto :fail
)

echo.
echo ======================================================================
echo  KESZ: %NAME%   vege: !DATE! !TIME!
echo ======================================================================
call :maybe_pause
endlocal
exit /b 0

:fail
echo.
echo ======================================================================
echo  MEGSZAKADT: %NAME%
echo ======================================================================
call :maybe_pause
endlocal
exit /b 1

:maybe_pause
REM csak dupla kattintasnal varjon billentyure; "set NOPAUSE=1" kikapcsolja
if "%NOPAUSE%"=="1" exit /b 0
echo %cmdcmdline% | find /i "%~nx0" >nul && pause
exit /b 0
