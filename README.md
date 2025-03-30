# German to Russian EPUB Translator
All scripts are written just for fun. Convert German EPUB text to Russian,German bilingual text.

## book_gemma.py
Translates German EPUB text into a bilingual German-Russian format using Gemma 3 with assistance from Ollama. Generates natural, structured JSON translations, stores translations and word frequencies, and supports customizable word filtering. Designed for testing purposes—performance is slow.

## book_nllb.py
Translates German EPUB text into a bilingual German-Russian format. The solution utilizes NLLB and KeyBERT. You can use it to translate between any languages by specifying the src_lang and tgt_lang parameters.

## convert_sqlite.py
Reads German nouns from JSON, stores them in an SQLite database. For use with book_nllb.py.

## german_nouns_output.json
https://github.com/Hanttone/der-die-das-game/blob/master/data/german_nouns_output.json
