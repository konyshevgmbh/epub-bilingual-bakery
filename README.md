# 🇩🇪→🇷🇺 German to Bilingual German-Russian EPUB Translator

This tool is built for language learners and enthusiasts who want to convert German-language EPUB files into bilingual German-Russian format. Each sentence is followed by a partial translation, with key words and phrases explained in Russian. This method naturally enhances comprehension and expands vocabulary through context.

**Default behavior:** **German in → German-Russian EPUB out**  
Other language pairs can be configured via parameters.

---

## 📘 `book_nllb.py` — Main Translation Script


This script converts German EPUBs into bilingual German-Russian EPUBs using:

- **Meta’s NLLB** for machine translation  
- **KeyBERT** for keyword extraction  
- Inline translation to support contextual learning

> 💡 Tip: Use **VS Code** with a **devcontainer** for a consistent development environment.  
> Alternatively, use the traditional workflow:

```bash
uv venv
.venv\Scripts\activate  # On Windows
uv pip install .
python book_nllb.py input.epub output.epub
```

![Sample output of book_nllb.py](sample1.png)

---

## 🗃️ `convert_sqlite.py` — Vocabulary Support Script

This script loads German nouns from a JSON file into an SQLite database.  
It's useful in combination with `book_nllb.py` for word lookups and vocabulary review.

Reference file:  
[german_nouns_output.json](https://github.com/Hanttone/der-die-das-game/blob/master/data/german_nouns_output.json)

---

## 🤖 `book_gemma.py` — Experimental Translation Script

An alternative translation method using **Gemma 3** via **Ollama**.

- Outputs bilingual German-Russian translation in JSON format
- Generates word frequency statistics
- Allows filtering and selective translation
- Designed for experimentation; not optimized for production use

> ⚠️ Demo code only — no setup or containerization included.

---

## 📦 Executable Release

A standalone executable version is available for Windows, macOS, and Linux.  
No need to install Python or any dependencies.

### Download

Grab the latest executable from the [Releases](https://github.com/konyshevgmbh/epub-bilingual-penetration/releases) page.

### Usage

```bash
# Windows
epub-bilingual-translator.exe input.epub output.epub

# macOS/Linux
./epub-bilingual-translator input.epub output.epub
```

Example with language override:

```bash
epub-bilingual-translator.exe --tgt-lang eng_Latn in.epub out.epub
```

![Sample output of book_nllb.py](sample2.png)

---

## 🌐 Available Languages

See all supported languages for each model family here:  
https://dl-translate.readthedocs.io/en/latest/available_languages/

---

### 🔧 Build the Executable Yourself

If you'd rather build it manually:

1. On Windows: run `build_executable.bat`  
2. On macOS/Linux: run `./build_executable.sh`  
   (Make it executable first: `chmod +x build_executable.sh`)

For release instructions, see [RELEASE_INSTRUCTIONS.md](RELEASE_INSTRUCTIONS.md).

---

Pull requests are welcome. Suggestions appreciated.  
Yes, there might be a bug or two — feel free to point them out.

