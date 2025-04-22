#!/bin/bash

echo "Creating executable for EPUB Bilingual Translator..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install -e .

# Build executable with PyInstaller
echo "Building executable with PyInstaller..."
pyinstaller --onefile --icon=icon.ico --name=epub-bilingual-translator book_nllb.py

echo ""
echo "Build complete! The executable is located in the 'dist' folder."
echo "You can now test it by running: ./dist/epub-bilingual-translator --help"
echo ""
echo "To publish this executable to GitHub Releases, follow the instructions in RELEASE_INSTRUCTIONS.md"
echo ""

# Make the executable file executable
chmod +x dist/epub-bilingual-translator

read -p "Press Enter to continue..."
