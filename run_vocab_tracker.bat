@echo off
setlocal

set PY=%~dp0.venv\Scripts\python.exe
set SCRIPT=%~dp0book_vocab_tracker.py
set VOCAB=%~dp0vocab_b1_goethe_full.tsv

if "%~1"=="" (
    echo Usage: run_vocab_tracker.bat input.epub [output.epub]
    echo.
    echo Options (optional^):
    echo   --vocab   path to vocab TSV  (default: vocab_b1_goethe_full.tsv^)
    echo   --interval N                 (default: 100^)
    echo   --max N                      (default: 9^)
    echo.
    echo Example:
    echo   run_vocab_tracker.bat "Berlin fuer die Hosentasche.epub"
    echo   run_vocab_tracker.bat input.epub output.epub --interval 150 --max 4
    exit /b 1
)

set INPUT=%~1

if "%~2"=="" (
    set OUTPUT=%~dpn1_annotated.epub
) else (
    set OUTPUT=%~2
    shift
)

set EXTRA_ARGS=
:parse
shift
if "%~1"=="" goto run
set EXTRA_ARGS=%EXTRA_ARGS% %1
goto parse

:run
echo.
echo Input:    %INPUT%
echo Output:   %OUTPUT%
echo Vocab:    %VOCAB%
echo.

set PYTHONIOENCODING=utf-8
"%PY%" "%SCRIPT%" "%INPUT%" "%OUTPUT%" --vocab "%VOCAB%" %EXTRA_ARGS%

if %errorlevel%==0 (
    echo.
    echo Done: %OUTPUT%
) else (
    echo.
    echo Error during processing.
)

endlocal
