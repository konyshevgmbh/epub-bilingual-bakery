# German to Russian EPUB Translator
All scripts are written just for fun. Convert German EPUB text to Russian,German bilingual text.

## book_gemma.py
Translates German EPUB text to Russian using SQLite and ollama. Prompts for natural, structured JSON translations. Stores translations and word frequencies. Customizable word filtering. It is slow. Just for test purposes.

## book_nllb.py
Translates German EPUB text to Russian using SQLite, NLLB, and KeyBERT. Manages word frequency and elementary word filtering.

## convert_sqlite.py
Reads German nouns from JSON, stores them in an SQLite database. For use with book_nllb.py.

## german_nouns_output.json
https://github.com/Hanttone/der-die-das-game/blob/master/data/german_nouns_output.json
