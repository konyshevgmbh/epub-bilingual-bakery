# German to bilingual German-Russian EPUB Translator

![epub-bilingual-penetration banner](banner.png)

This tool transforms foreign-language texts into a bilingual battleground, where every sentence is followed by a half-baked translation and a desperate attempt to explain the keywords. It’s like learning through osmosis, panic, and a healthy dose of linguistic trauma.

All scripts are made just for fun — the kind of fun where German grammar haunts your dreams. By default, it converts German EPUB files into a German-Russian format, because pain loves company.

---

## 📘 `book_nllb.py`

Translates German EPUBs into bilingual German-Russian format using **NLLB** for translation and **KeyBERT** for keyword extraction, because what’s better than AI-powered confusion?

- Supports other language pairs via `src_lang` and `tgt_lang` if you feel like suffering in new and exciting ways
- Outputs bilingual text with inline keyword translations, for that immersive “am I doing this right?” experience

![Sample output of book_nllb.py](sample.png)

> 💡 Run it in **VS Code** using a **devcontainer** if you enjoy setting up environments more than actual learning.

---

## 🗃️ `convert_sqlite.py`

Reads German nouns from a JSON file and crams them into an SQLite database.  
Because nothing screams “language learning” like SQL queries and existential dread.

Source:  
[german_nouns_output.json](https://github.com/Hanttone/der-die-das-game/blob/master/data/german_nouns_output.json)

Use it with `book_nllb.py` if you want to know which article goes with which noun — or at least pretend you do.

---

## 🤖 `book_gemma.py`

Experimental translation script using **Gemma 3** via **Ollama**.  
Because one working script wasn't chaotic enough.

- Translates German EPUBs into bilingual German-Russian JSON format
- Outputs structured data with word frequencies, in case you're writing your thesis on why "doch" appears 900 times
- Supports filtering and selective translation, like a Tinder for words
- Absolutely not optimized — this is demo code, and you *will* feel it

> ⚠️ No setup, no devcontainer, no pity. Just raw, experimental code. Use at your own risk — or better yet, don’t.
