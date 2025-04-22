@echo off
echo Creating executable for EPUB Bilingual Translator...

REM Check if virtual environment exists
if not exist .venv (
    echo Creating virtual environment...
    uv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate

REM Install dependencies
echo Installing dependencies...
uv pip install -e .

REM Build executable with PyInstaller
echo Building executable with PyInstaller...
pyinstaller --onefile --icon=icon.ico --name=epub-bilingual-translator book_nllb.py

echo.
echo Build complete! The executable is located in the 'dist' folder.
echo You can now test it by running: dist\epub-bilingual-translator.exe --help
echo.
echo To publish this executable to GitHub Releases, follow the instructions in RELEASE_INSTRUCTIONS.md
echo.

pause
