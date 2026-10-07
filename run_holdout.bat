@echo off
setlocal enabledelayedexpansion
REM ======================================================================
REM  Fix train/val felosztas, egyetlen futas (nincs fold-ciklus, nincs OOF):
REM     tanitas  : train_v8.csv            (4199 study)
REM     validacio: train_labeled_208_reference.csv  (208 radiologus-cimkezett study)
REM  Az early stopping es a best.pt is a 208-on mert macro_soft_auc alapjan
REM  valaszt, ezert az ott mert ertek enyhen optimista.
REM  A 3-fold CV (run_cv.bat) ettol fuggetlen, valtozatlanul hasznalhato.
REM
REM  Hasznalat:
REM     run_holdout.bat
REM     run_holdout.bat E3_img320 --set data.image_size=320
REM  Ha az elso argumentum nem "-"-vel kezdodik, az a run nevenek elotagja:
REM     E3_img320_holdout208_<idobelyeg>
REM
REM  Folytatas (--resume <run nev>, csak elso argumentumkent): egy megszakadt
REM  run a last.pt-bol folytatodik ugyanabban a mappaban (train.resume,
REM  epochhatarrol; a felbehagyott epoch elveszik):
REM     run_holdout.bat --resume E3_img320_holdout208_20261005_143722 --set data.image_size=320
REM  Az extra argumentumok es a kornyezeti valtozok ugyanazok legyenek, mint az
REM  eredeti inditasnal; az elteres a resume-ellenorzesen elakad.
REM
REM  Az elso futas elokesziti (ha meg nincsenek meg):
REM     SPLITS     - make-fixed-split (train_v8 = fold 1, 208 = fold 0)
REM     FROZEN_REF - freeze-reference --from-csv a 208-as CSV-bol
REM  DATA_ROOT, TRAIN_CSV, VAL_CSV, SPLITS, FROZEN_REF kornyezeti
REM  valtozokkal felulirhato.
REM  HOLDOUT (alap: 208 = 158 + Andrew 50) valasztja a referenciahalmazt;
REM  VAL_CSV, SPLITS, FROZEN_REF es a run neve (holdout<HOLDOUT>) ebbol kepzodik.
REM  Regi 158-as holdout (train_v4/v6-hoz):
REM     set HOLDOUT=158
REM     set TRAIN_CSV=F:\Kaggle\data\train_v6.csv
REM  A TRAIN_CSV nem tartalmazhatja a holdout studykat.
REM ======================================================================

set "ROOT=%~dp0"
set "CONDA_ENV=kaggle_2026"
set "CONDA_ROOT=%LOCALAPPDATA%\miniconda3"
if not defined DATA_ROOT set "DATA_ROOT=F:\Kaggle\data"
if not defined TRAIN_CSV set "TRAIN_CSV=%DATA_ROOT%\train_v8.csv"
if not defined HOLDOUT set "HOLDOUT=208"
if not defined VAL_CSV set "VAL_CSV=%DATA_ROOT%\train_labeled_%HOLDOUT%_reference.csv"
if not defined SPLITS set "SPLITS=%ROOT%work\splits\holdout%HOLDOUT%\splits.csv"
if not defined FROZEN_REF set "FROZEN_REF=%ROOT%work\labels\frozen_reference_ref%HOLDOUT%.csv"

set "EXTRA=%*"
set "PREFIX="
set "RESUME="
set "FIRST=%~1"
if /i "%~1"=="--resume" goto :parse_resume
if defined FIRST if not "!FIRST:~0,1!"=="-" (
    set "PREFIX=!FIRST!_"
    set "EXTRA=!EXTRA:*%1=!"
)
goto :parsed

:parse_resume
if "%~2"=="" (
    echo [HIBA] A --resume utan meg kell adni a run nevet, pl. B0_E3_img320_holdout208_20261005_143722
    exit /b 1
)
set "RESUME=%~2"
set "EXTRA=!EXTRA:*%~2=!"

:parsed
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

set PATHS=--set "paths.train_csv=%TRAIN_CSV%" --set "paths.splits_csv=%SPLITS%" --set "paths.frozen_reference_csv=%FROZEN_REF%" --set "paths.reference_csv=%VAL_CSV%" --set "paths.exclude_from_training_csv=%VAL_CSV%"

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

REM --- a split illeszkedik-e a TRAIN_CSV-hez es a VAL_CSV-hez ---------
REM (pl. egy korabbi terminalbol ottmaradt TRAIN_CSV: holdout studyk a tanito CSV-ben)
python -m knee_mri.cli check-fixed-split %PATHS% --validation-csv "%VAL_CSV%"
if errorlevel 1 (
    echo [HIBA] A split nem illik a TRAIN_CSV / VAL_CSV parhoz. Ellenorizd a
    echo        TRAIN_CSV, HOLDOUT, VAL_CSV, SPLITS kornyezeti valtozokat.
    goto :fail
)

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%i"
if defined RESUME (
    set "NAME=%RESUME%"
) else (
    set "NAME=%PREFIX%holdout%HOLDOUT%_%STAMP%"
)

REM --- kimeneti mappa: work\runs\<train_csv stem>\ ---------------------
REM Az utolso kiirt sor az utvonal (a config figyelmeztetesei is stdout-ra mennek).
set "RUNS_DIR_FILE=%TEMP%\knee_mri_runs_dir_%STAMP%.txt"
python -m knee_mri.cli output-dir %PATHS% %EXTRA% > "%RUNS_DIR_FILE%"
if errorlevel 1 (
    type "%RUNS_DIR_FILE%"
    del "%RUNS_DIR_FILE%" >nul 2>&1
    echo [HIBA] Nem sikerult meghatarozni a kimeneti mappat ^(output-dir^).
    goto :fail
)
for /f "usebackq delims=" %%i in ("%RUNS_DIR_FILE%") do set "RUNS_DIR=%%i"
del "%RUNS_DIR_FILE%" >nul 2>&1

set "RUN=%RUNS_DIR%\%NAME%"
set "START="
if defined RESUME (
    if exist "%RUN%\run_summary.json" (
        echo [INFO] A run mar kesz ^(run_summary.json^), nincs mit folytatni: %RUN%
        call :maybe_pause
        endlocal
        exit /b 0
    )
    if not exist "%RUN%\last.pt" (
        echo [HIBA] --resume: nincs last.pt: %RUN%\last.pt
        echo        Ellenorizd a nevet, es hogy a TRAIN_CSV ugyanaz-e, mint az eredeti inditasnal.
        goto :fail
    )
    set START=--set "train.resume=%RUN%\last.pt"
)

echo ======================================================================
echo  Run        : %NAME%
if defined RESUME echo  Mod        : folytatas a last.pt-bol ^(epochhatarrol^)
echo  Tanitas    : %TRAIN_CSV%
echo  Validacio  : %VAL_CSV%
echo  Extra args : %EXTRA%
echo  Kimenet    : %RUN%
echo ======================================================================

python "%ROOT%src\train.py" --mode fold --set split.fold=0 %PATHS% --name "%NAME%" %EXTRA% %START%
if errorlevel 1 (
    echo.
    echo [HIBA] A tanitas hibaval leallt. Log: %RUN%\run.log
    echo        Folytatas: run_holdout.bat --resume %NAME% ^<ugyanazok az extra argumentumok^>
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
