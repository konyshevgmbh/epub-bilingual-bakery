# Creating and Publishing Releases for EPUB Bilingual Penetration

This document provides instructions for creating an executable artifact for `book_nllb.py` using PyInstaller and publishing it to GitHub Releases.

## Prerequisites

- Python 3.8 or higher
- Git repository cloned locally
- PyInstaller (included in project dependencies)
- GitHub account with write access to the repository

## Creating the Executable

### 1. Set up the Environment

```bash
# Create and activate a virtual environment using uv
uv venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

### 2. Create the Executable with PyInstaller

PyInstaller will package the application into a standalone executable with the specified icon.

```bash
# Run PyInstaller with the icon
pyinstaller --onefile --icon=icon.ico --name=epub-bilingual-translator book_nllb.py
```

This command:
- `--onefile`: Creates a single executable file
- `--icon=icon.ico`: Uses the provided icon for the executable
- `--name=epub-bilingual-translator`: Names the output executable
- `book_nllb.py`: The main script to package

The executable will be created in the `dist` directory.

### 3. Test the Executable

Before publishing, test that the executable works correctly:

```bash
# On Windows
dist\epub-bilingual-translator.exe --help

# On macOS/Linux
./dist/epub-bilingual-translator --help
```

## Publishing to GitHub Releases

### 1. Create a New Release on GitHub

1. Go to your repository on GitHub
2. Click on "Releases" in the right sidebar
3. Click "Create a new release" or "Draft a new release"

### 2. Fill in Release Information

1. **Tag version**: Create a new tag (e.g., `v1.0.0`)
2. **Release title**: Provide a descriptive title (e.g., "EPUB Bilingual Translator v1.0.0")
3. **Description**: Add release notes describing the features, improvements, and bug fixes

### 3. Upload the Executable

1. Drag and drop the executable from the `dist` directory to the "Attach binaries" section
   - For Windows: `dist\epub-bilingual-translator.exe`
   - For macOS/Linux: `dist/epub-bilingual-translator`
2. Alternatively, click "Upload assets" and select the executable file

### 4. Publish the Release

1. If you want to mark this as a pre-release or beta version, check the appropriate box
2. Click "Publish release" to make it available to users

## Automating Releases with GitHub Actions (Optional)

For automated builds and releases, you can set up a GitHub Actions workflow:

1. Create a `.github/workflows/release.yml` file with the following content:

```yaml
name: Build and Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [windows-latest, ubuntu-latest, macos-latest]
        
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        python -m pip install uv
        uv pip install -e .
        
    - name: Build with PyInstaller
      run: |
        pyinstaller --onefile --icon=icon.ico --name=epub-bilingual-translator-${{ matrix.os }} book_nllb.py
        
    - name: Upload artifact
      uses: actions/upload-artifact@v3
      with:
        name: epub-bilingual-translator-${{ matrix.os }}
        path: dist/*
        
  release:
    needs: build
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Download all artifacts
      uses: actions/download-artifact@v3
      
    - name: Create Release
      uses: softprops/action-gh-release@v1
      with:
        files: |
          epub-bilingual-translator-windows-latest/*
          epub-bilingual-translator-ubuntu-latest/*
          epub-bilingual-translator-macos-latest/*
        draft: false
        prerelease: false
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

2. Push this file to your repository
3. Create and push a tag (e.g., `git tag v1.0.0 && git push origin v1.0.0`) to trigger the workflow

## Notes

- The executable size may be large due to the inclusion of ML models and dependencies
- Consider adding a README section about downloading and using the executable
- For frequent releases, automating with GitHub Actions is recommended
