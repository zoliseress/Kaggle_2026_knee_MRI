@echo off
setlocal enabledelayedexpansion
REM ======================================================================
REM  3-fold keresztvalidacio: a foldokat (split.fold = 0,1,2) egymas utan
REM  futtatja, majd osszefesuli a per-fold predikciokat OOF-fa.
REM
REM  Hasznalat:
REM     run_cv.bat
REM     run_cv.bat --set data.image_size=288 --set train.max_epochs=10
REM     run_cv.bat V2soft_img320 --set data.image_size=320
REM     run_cv.bat --resume V2soft_img320_cv3_20260924_232150 --set data.image_size=320
REM  Minden ide irt extra argumentum MINDHAROM foldra ervenyes.
REM  Ha az elso argumentum nem "-"-vel kezdodik, az a run-group elotagja:
REM     V2soft_img320_cv3_<idobelyeg>_fold<N>, ..._oof
REM
REM  Folytatas (--resume <run-group>, csak elso argumentumkent): egy megszakadt
REM  run-groupot visz tovabb ugyanazzal a nevvel. Foldonkent:
REM     run_summary.json van  -> kesz, kihagyja
REM     csak last.pt van      -> onnan folytatja (train.resume, epochhatarrol)
REM     egyik sincs           -> elolrol inditja
REM  majd ujra osszefesuli az OOF-ot. Az extra argumentumok (a --config is)
REM  ugyanazok legyenek, mint az eredeti inditasnal: a folytatott foldnal ezt
REM  a resume-ellenorzes ki is kenyszeriti, egy elolrol indulo foldnal nem.
REM
REM  Kornyezeti valtozokkal felulirhato:
REM     NFOLDS     (alap: 3)
REM     TRAIN_CSV  (alap: a config paths.train_csv-je)
REM     SPLITS     (alap: work\splits\cv<NFOLDS>_<train_csv stem>\splits.csv)
REM  Ha a SPLITS meg nincs meg, a script legyartja (make-splits a TRAIN_CSV
REM  studyjaibol, NFOLDS folddal), majd minden inditasnal ellenorzi
REM  (check-splits): a fold-szam, a TRAIN_CSV es a kizarasi lista
REM  (paths.exclude_from_training_csv, alap: a ref208) egyezzen.
REM  Teacher CV (3 fold = alap, ref208 nelkul, a train_v8.csv-bol):
REM     run_cv.bat B0_teacher --set data.image_size=320
REM  Regi, 4407 studys work\splits\splits.csv-vel indult run-group folytatasa
REM  (a ref208 abban tanitott, ezert a kizarast is ki kell kapcsolni):
REM     set SPLITS=%~dp0work\splits\splits.csv
REM     run_cv.bat --resume ... --set paths.exclude_from_training_csv=
REM ======================================================================

set "ROOT=%~dp0"
if not defined NFOLDS set "NFOLDS=3"
set "CONDA_ENV=kaggle_2026"
set "CONDA_ROOT=%LOCALAPPDATA%\miniconda3"

set "EXTRA=%*"
set "PREFIX="
set "RESUME="
set "FIRST=%~1"
REM A %1, %2 ... a "="-nel is darabol, ezert az EXTRA a %*-bol, levagassal keszul.
if /i "%~1"=="--resume" goto :parse_resume
if defined FIRST if not "!FIRST:~0,1!"=="-" (
    set "PREFIX=!FIRST!_"
    set "EXTRA=!EXTRA:*%1=!"
)
goto :parsed

:parse_resume
if "%~2"=="" (
    echo [HIBA] A --resume utan meg kell adni a run-group nevet, pl. R50_E3_cv3_20261001_110141
    exit /b 1
)
set "RESUME=%~2"
set "EXTRA=!EXTRA:*%~2=!"

:parsed
REM A TRAIN_CSV a tobbi argumentum ele kerul, igy egy kezzel irt --set paths.train_csv felulirja.
if defined TRAIN_CSV set EXTRA=--set "paths.train_csv=%TRAIN_CSV%" !EXTRA!
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

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%i"
if defined RESUME (
    set "GROUP=%RESUME%"
) else (
    set "GROUP=%PREFIX%cv%NFOLDS%_%STAMP%"
)
set /a LAST=%NFOLDS%-1

REM --- kimeneti mappa: work\runs\<train_csv stem>\ ---------------------
REM Az utolso kiirt sor az utvonal (a config figyelmeztetesei is stdout-ra mennek).
set "RUNS_DIR_FILE=%TEMP%\knee_mri_runs_dir_%STAMP%.txt"
python -m knee_mri.cli output-dir %EXTRA% > "%RUNS_DIR_FILE%"
if errorlevel 1 (
    type "%RUNS_DIR_FILE%"
    del "%RUNS_DIR_FILE%" >nul 2>&1
    echo [HIBA] Nem sikerult meghatarozni a kimeneti mappat ^(output-dir^).
    goto :fail
)
for /f "usebackq delims=" %%i in ("%RUNS_DIR_FILE%") do set "RUNS_DIR=%%i"
python -m knee_mri.cli output-dir --train-csv %EXTRA% > "%RUNS_DIR_FILE%"
if errorlevel 1 (
    type "%RUNS_DIR_FILE%"
    del "%RUNS_DIR_FILE%" >nul 2>&1
    echo [HIBA] Nem sikerult meghatarozni a tanito CSV-t ^(output-dir --train-csv^).
    goto :fail
)
for /f "usebackq delims=" %%i in ("%RUNS_DIR_FILE%") do set "TRAIN_CSV_RESOLVED=%%i"
del "%RUNS_DIR_FILE%" >nul 2>&1
for %%i in ("%TRAIN_CSV_RESOLVED%") do set "TRAIN_STEM=%%~ni"

REM --- CV split: legyartas (ha nincs) es ellenorzes --------------------
if not defined SPLITS set "SPLITS=%ROOT%work\splits\cv%NFOLDS%_%TRAIN_STEM%\splits.csv"
set EXTRA=--set "paths.splits_csv=%SPLITS%" !EXTRA!
if not exist "%SPLITS%" (
    if defined RESUME (
        echo [HIBA] --resume, de nincs meg a split: %SPLITS%
        echo        A folytatashoz ugyanaz a split kell, mint az eredeti inditasnal ^(SPLITS^).
        goto :fail
    )
    echo [INFO] CV split keszitese: %SPLITS%  ^(%NFOLDS% fold, %TRAIN_CSV_RESOLVED%^)
    python -m knee_mri.cli make-splits %EXTRA% --set split.n_folds=%NFOLDS%
    if errorlevel 1 (
        echo [HIBA] A make-splits nem sikerult.
        goto :fail
    )
)
python -m knee_mri.cli check-splits %EXTRA% --n-folds %NFOLDS%
if errorlevel 1 (
    echo [HIBA] A split nem illik a futashoz: %SPLITS%
    echo        Ellenorizd az NFOLDS, TRAIN_CSV, SPLITS kornyezeti valtozokat.
    goto :fail
)

if defined RESUME if not exist "%RUNS_DIR%\%GROUP%_fold0" (
    echo [HIBA] --resume: nincs ilyen run-group: %RUNS_DIR%\%GROUP%_fold0
    echo        Ellenorizd a nevet, es hogy a --config / train_csv ugyanaz-e, mint az eredeti inditasnal.
    goto :fail
)

echo ======================================================================
echo  Run group  : %GROUP%
if defined RESUME echo  Mod        : folytatas ^(kesz fold kihagyva, last.pt-bol folytatva^)
echo  Foldok     : 0 .. %LAST%
echo  Train CSV  : %TRAIN_CSV_RESOLVED%
echo  Split      : %SPLITS%
echo  Extra args : %EXTRA%
echo  Kimenet    : %RUNS_DIR%\%GROUP%_fold^<N^>
echo ======================================================================

REM --- foldok egymas utan ---------------------------------------------
for /l %%F in (0,1,%LAST%) do (
    set "NAME=%GROUP%_fold%%F"
    set "RUN=%RUNS_DIR%\%GROUP%_fold%%F"
    set "SKIP="
    set "START="
    if defined RESUME (
        if exist "!RUN!\run_summary.json" set "SKIP=1"
        if not defined SKIP if exist "!RUN!\last.pt" set "START=--set "train.resume=!RUN!\last.pt""
    )
    echo.
    echo ----------------------------------------------------------------
    echo  FOLD %%F / %LAST%   ^(!NAME!^)   start: !DATE! !TIME!
    echo ----------------------------------------------------------------
    if defined SKIP (
        echo  FOLD %%F mar kesz ^(run_summary.json^), kihagyom.
    ) else (
        if defined START echo  Folytatas: !RUN!\last.pt
        python "%ROOT%src\train.py" --mode fold --set split.fold=%%F --name "!NAME!" %EXTRA% !START!
        if errorlevel 1 (
            echo.
            echo [HIBA] A %%F. fold hibaval leallt. A script nem folytatja.
            echo        Log: %RUNS_DIR%\!NAME!\run.log
            echo        Folytatas: run_cv.bat --resume %GROUP% ^<ugyanazok az extra argumentumok^>
            goto :fail
        )
        echo  FOLD %%F kesz.  vege: !DATE! !TIME!
    )
)

REM --- OOF merge -------------------------------------------------------
echo.
echo ----------------------------------------------------------------
echo  OOF merge
echo ----------------------------------------------------------------
REM A merge-oof a --out melle FIX neven irja az oof_metrics_per_class.csv-t
REM es az oof_summary.json-t, ezert minden run-group sajat alkonyvtarba kerul.
set "OOFDIR=%RUNS_DIR%\%GROUP%_oof"
if not exist "%OOFDIR%" mkdir "%OOFDIR%"
set PREDS=
for /l %%F in (0,1,%LAST%) do set PREDS=!PREDS! "%RUNS_DIR%\%GROUP%_fold%%F\validation_predictions.csv"
python -m knee_mri.cli merge-oof %PREDS% --out "%OOFDIR%\oof_predictions.csv" %EXTRA%
if errorlevel 1 (
    echo [FIGYELEM] Az OOF merge nem sikerult, de mindharom fold lefutott.
    echo            A per-fold eredmenyek megvannak a run mappakban.
)

echo.
echo ======================================================================
echo  KESZ. Mind a %NFOLDS% fold lefutott: %GROUP%
echo  OOF        : %RUNS_DIR%\%GROUP%_oof
echo ======================================================================
call :maybe_pause
endlocal
exit /b 0

:fail
echo.
echo ======================================================================
echo  MEGSZAKADT: %GROUP%
echo ======================================================================
call :maybe_pause
endlocal
exit /b 1

:maybe_pause
REM csak dupla kattintasnal varjon billentyure; "set NOPAUSE=1" kikapcsolja
if "%NOPAUSE%"=="1" exit /b 0
echo %cmdcmdline% | find /i "%~nx0" >nul && pause
exit /b 0
