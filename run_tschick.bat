@echo off
setlocal
set PYTHONIOENCODING=utf-8
set PY=%~dp0.venv\Scripts\python.exe
"%PY%" "%~dp0book_vocab_tracker.py" "%~dp0Tschick.epub" "%~dp0Tschick_annotated.epub" --vocab "%~dp0vocab_b1_goethe_full.tsv" --interval 100 --max 9
endlocal
