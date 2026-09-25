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
REM  Minden ide irt extra argumentum MINDHAROM foldra ervenyes.
REM  Ha az elso argumentum nem "-"-vel kezdodik, az a run-group elotagja:
REM     V2soft_img320_cv3_<idobelyeg>_fold<N>, ..._oof
REM
REM  Elofeltetel: a work\splits\splits.csv mar letezik es ugyanazzal az
REM  n_folds ertekkel keszult (make-splits), mint amennyit itt futtatunk.
REM ======================================================================

set "ROOT=%~dp0"
set "NFOLDS=3"
set "CONDA_ENV=kaggle_2026"
set "CONDA_ROOT=%LOCALAPPDATA%\miniconda3"

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

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%i"
set "GROUP=%PREFIX%cv%NFOLDS%_%STAMP%"
set /a LAST=%NFOLDS%-1

echo ======================================================================
echo  Run group  : %GROUP%
echo  Foldok     : 0 .. %LAST%
echo  Extra args : %EXTRA%
echo  Kimenet    : %ROOT%work\runs\%GROUP%_fold^<N^>
echo ======================================================================

REM --- foldok egymas utan ---------------------------------------------
for /l %%F in (0,1,%LAST%) do (
    set "NAME=%GROUP%_fold%%F"
    echo.
    echo ----------------------------------------------------------------
    echo  FOLD %%F / %LAST%   ^(!NAME!^)   start: !DATE! !TIME!
    echo ----------------------------------------------------------------
    python "%ROOT%src\train.py" --mode fold --set split.fold=%%F --name "!NAME!" %EXTRA%
    if errorlevel 1 (
        echo.
        echo [HIBA] A %%F. fold hibaval leallt. A script nem folytatja.
        echo        Log: %ROOT%work\runs\!NAME!\run.log
        goto :fail
    )
    echo  FOLD %%F kesz.  vege: !DATE! !TIME!
)

REM --- OOF merge -------------------------------------------------------
echo.
echo ----------------------------------------------------------------
echo  OOF merge
echo ----------------------------------------------------------------
REM A merge-oof a --out melle FIX neven irja az oof_metrics_per_class.csv-t
REM es az oof_summary.json-t, ezert minden run-group sajat alkonyvtarba kerul.
set "OOFDIR=%ROOT%work\runs\%GROUP%_oof"
if not exist "%OOFDIR%" mkdir "%OOFDIR%"
set PREDS=
for /l %%F in (0,1,%LAST%) do set PREDS=!PREDS! "%ROOT%work\runs\%GROUP%_fold%%F\validation_predictions.csv"
python -m knee_mri.cli merge-oof %PREDS% --out "%OOFDIR%\oof_predictions.csv"
if errorlevel 1 (
    echo [FIGYELEM] Az OOF merge nem sikerult, de mindharom fold lefutott.
    echo            A per-fold eredmenyek megvannak a run mappakban.
)

echo.
echo ======================================================================
echo  KESZ. Mind a %NFOLDS% fold lefutott: %GROUP%
echo  OOF        : %ROOT%work\runs\%GROUP%_oof
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
