# German to bilingual German-Russian EPUB Translator

This tool helps transform foreign-language texts into bilingual format, where each sentence is followed by a partial translation with key words and phrases explained. This approach aids in intuitive understanding and vocabulary expansion through context.

All scripts are created just for fun. By default, it converts German EPUB text into a bilingual German-Russian format.

---

## 📘 `book_nllb.py`

Translates German EPUB text into a bilingual German-Russian format. The solution uses **NLLB** for translation and **KeyBERT** for keyword extraction.

- Supports other language pairs via `src_lang` and `tgt_lang` parameters
- Output contains inline translations of key vocabulary for contextual learning

![Sample output of book_nllb.py](sample.png)

> 💡 Use **VS Code** and a **devcontainer** to run the script.

---

## 🗃️ `convert_sqlite.py`

Reads German nouns from a JSON file and stores them in an SQLite database.  
Useful for vocabulary lookup in conjunction with `book_nllb.py`.

Source:  
[german_nouns_output.json](https://github.com/Hanttone/der-die-das-game/blob/master/data/german_nouns_output.json)

---

## 🤖 `book_gemma.py`

Experimental translation script using **Gemma 3** via **Ollama**.

- Translates German EPUB to bilingual German-Russian JSON format
- Generates structured output with word frequencies
- Supports filtering and selective translation
- Not optimized for performance — meant for testing

> ⚠️ No setup or devcontainer for this script — demo code only.
