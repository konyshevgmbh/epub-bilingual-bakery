import sqlite3
import json
from bs4 import BeautifulSoup, NavigableString
from ebooklib import epub
import ebooklib
import torch
from collections import Counter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from keybert import KeyBERT

# Initialize SQLite database
translations_database_connection = sqlite3.connect('translations.sqlite')
cursor = translations_database_connection.cursor()
device = "cuda" if torch.cuda.is_available() else "cpu"

cursor.execute('''
    CREATE TABLE IF NOT EXISTS translations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        german_text TEXT UNIQUE,
        json_translation TEXT
    )
''')


# Add a table for tracking word frequency
cursor.execute('''
    CREATE TABLE IF NOT EXISTS word_frequency (
        word TEXT PRIMARY KEY,
        count INTEGER DEFAULT 1
    )
''')

cursor.execute(
    "CREATE INDEX IF NOT EXISTS idx_translations ON translations(german_text)")
cursor.execute(
    "CREATE INDEX IF NOT EXISTS idx_word_frequency ON word_frequency(word)")
translations_database_connection.commit()

# List of elementary words to exclude
ELEMENTARY_WORDS = [
    "der", "die", "das", "den", "dem", "ein", "eine", "einer", "eines", "einem", "einen",
    "an", "in", "zu", "auf", "mit", "von", "bei", "nach", "aus", "für", "um",
    "durch", "über", "unter", "vor", "hinter", "neben", "zwischen", "und", "oder",
    "aber", "denn", "weil", "wenn", "als", "dass", "ob", "ist", "sind", "war",
    "waren", "sein", "haben", "hatte", "hatten", "werden", "wurde", "wurden",
    "nicht", "kein", "keine", "keinem", "keinen", "keiner", "keines", "nur", "auch",
    "schon", "noch", "wieder", "immer", "dann", "darum", "deshalb", "trotzdem"
]

# Loading models
print("Loading NLLB and KeyBERT models...")
tokenizer = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-1.3B", src_lang="deu_Latn", tgt_lang="rus_Cyrl")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-1.3B").to(device)
kw_model = KeyBERT()
print("Models loaded")

def clean_text(text):
    text = text.replace('\u00a0', ' ')
    text = text.replace('\u2013', '-')
    return text

def translate_text(german_text):
    """Translate text using NLLB"""
    cleaned_german_text = clean_text(german_text)
    inputs = tokenizer(cleaned_german_text, return_tensors="pt").to(device)
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("rus_Cyrl"),
        max_length=500
    )
    translated_text = tokenizer.batch_decode(
        translated_tokens, skip_special_tokens=True)[0]
    return translated_text


wordlist_connection = sqlite3.connect('wordlist.sqlite')
wordlist_cursor = wordlist_connection.cursor()


def get_db_word(keyword):
    """Replace the keyword with a word from the database if it exists"""
    wordlist_cursor.execute(
        "SELECT word FROM words WHERE key = ?", (keyword,))
    result = wordlist_cursor.fetchone()

    if result:
        return result[0]
    return keyword


def extract_keywords(text, top_n=10):
    """Extract keywords from the text"""
    if len(text.split()) < 5:
        return []   # Skip short texts

    keywords = kw_model.extract_keywords(text, top_n=top_n)

    updated_keywords = []
    for keyword, score in keywords:
        if len( keyword) < 3:   
            continue
        if keyword.lower() in ELEMENTARY_WORDS:
            continue
        if keyword.isdigit():   
            continue
        updated_keyword = get_db_word(keyword)
        updated_keywords.append((updated_keyword, score))

    return updated_keywords


def translate_keyword(word):
    """Translate an individual word or phrase"""
    inputs = tokenizer(word, return_tensors="pt").to(device)
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("rus_Cyrl"),
        max_length=50
    )
    translated_word = tokenizer.batch_decode(
        translated_tokens, skip_special_tokens=True)[0]
    return translated_word


def create_json_translation(german_text, russian_text, keywords):
    """Create JSON structure with translation"""
    # Split text into sentences (simple splitting by periods)
    german_sentences = german_text.replace('. ', '.|').split('|')
    russian_sentences = russian_text.replace('. ', '.|').split('|')

    # Align the number of sentences
    min_len = min(len(german_sentences), len(russian_sentences))
    german_sentences = german_sentences[:min_len]
    russian_sentences = russian_sentences[:min_len]

    translations = []

    # If the text is short, process it as a single sentence
    if len(german_text.split()) <= 20:
        word_list = []
        for keyword, score in keywords:
            if score > 0.2:  # Use only words with sufficient weight
                translated_keyword = translate_keyword(keyword)
                word_list.append({keyword: translated_keyword})

        entry = {
            "ge": german_text,
            "ru": russian_text,
            "w": word_list
        }
        translations.append(entry)
    else:
        # Split long text into parts
        for i in range(min_len):
            if not german_sentences[i].strip() or not russian_sentences[i].strip():
                continue

            # Extract keywords for each sentence
            sentence_keywords = extract_keywords(german_sentences[i], top_n=5)

            word_list = []
            for keyword, score in sentence_keywords:
                if score > 0.2 and keyword.lower() not in ELEMENTARY_WORDS:
                    translated_keyword = translate_keyword(keyword)
                    word_list.append({keyword: translated_keyword})

            entry = {
                "ge": german_sentences[i],
                "ru": russian_sentences[i],
                "w": word_list
            }
            translations.append(entry)

    return translations


def filter_words(word_objects):
    filtered_words = []

    for word_obj in word_objects:
        if isinstance(word_obj, str):
            continue
        for german_word, translation in word_obj.items():
            # Check if the word is elementary
            if any(elem_word == german_word.lower().strip() for elem_word in ELEMENTARY_WORDS):
                continue

            # Skip words with more than 3 parts
            if len(german_word.split()) > 3:
                continue

            # Check word frequency in the database
            cursor.execute(
                "SELECT count FROM word_frequency WHERE word = ?", (german_word,))
            result = cursor.fetchone()

            if result:
                word_count = result[0]
                if word_count < 3:  # Add the word only if it has been encountered less than 3 times
                    filtered_words.append(word_obj)
                    # Increment the word counter
                    cursor.execute(
                        "UPDATE word_frequency SET count = count + 1 WHERE word = ?", (german_word,))
                    translations_database_connection.commit()
            else:
                # If the word is encountered for the first time, add it to the database and the filtered list
                filtered_words.append(word_obj)
                cursor.execute(
                    "INSERT INTO word_frequency (word, count) VALUES (?, 1)", (german_word,))
                translations_database_connection.commit()

    return filtered_words

def make_string_translation(json_translation):
    if not json_translation:
        return None

    soup = BeautifulSoup("", "html.parser")

    for entry in json_translation:
        ge = entry.get("ge", "").strip()
        ru = entry.get("ru", "").strip()
        w = entry.get("w", [])

        if not ge or not ru:
            continue
        
        if len( ge ) < 3:   
            continue

        # Skip numbers
        if ge.isdigit():
            soup.append(" "+ge+" ")
            continue

        filtered_w = filter_words(w)

        if ge == ru or not ru:
            soup.append(ge)
            continue

        bold_ge = soup.new_tag("b")
        bold_ge.string = ge

        italic_ru = soup.new_tag("i")
        italic_ru.string = ru

        translation_part = soup.new_tag("span")
        translation_part.append(bold_ge)
        translation_part.append(f" ( ")

        translation_part.append(italic_ru)

        # Format the word translations
        if filtered_w:
            word_translations = []
            for word_obj in filtered_w:
                for german, russian in word_obj.items():
                    word_translations.append(f"{german} - {russian}")
            translation_part.append(f"; {'; '.join(word_translations)} )")
        else:
            translation_part.append(" )")

        soup.append(translation_part)

    return soup


def initialize_word_frequency():
    """Initialize word frequency counter from existing translations in the database"""
    # First, clear the word frequency table
    cursor.execute("DELETE FROM word_frequency")
    translations_database_connection.commit()


def get_json_translation(german_text):
    """Get JSON translation from the database or create a new one"""
    сleaned_german_text = clean_text(german_text)
    if сleaned_german_text.isdigit() or len(сleaned_german_text) < 3:  # Skip numbers
        return [{"ge": сleaned_german_text, "ru": сleaned_german_text, "w": []}]


    cursor.execute(
        "SELECT json_translation FROM translations WHERE german_text = ?", (сleaned_german_text,))
    result = cursor.fetchone()
    if result:
        return json.loads(result[0])

    try:
        # Translate the text
        russian_text = translate_text(сleaned_german_text)

        # Extract keywords
        keywords = extract_keywords(сleaned_german_text, top_n=10)

        # Create JSON structure
        translated_data = create_json_translation(
            сleaned_german_text, russian_text, keywords)

        # Save to database
        cursor.execute("INSERT INTO translations (german_text, json_translation) VALUES (?, ?)",
                       (сleaned_german_text, json.dumps(translated_data)))
        translations_database_connection.commit()

        return translated_data

    except Exception as e:
        print(f"Translation error: {e}")
        return []


# Initialize word frequency counter when the script starts
print("Initializing word frequency counter from existing translations...")
initialize_word_frequency()


def process_epub(input_file, output_file, translate_limit=0):
    print(f"Processing file {input_file}...")
    book = epub.read_epub(input_file)
    translated_count = 0

    for item in book.items:
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            print(f"Processing document...")
            soup = BeautifulSoup(item.get_content(), 'html.parser')

            # Modify this list to include other tags if needed
            for tag in soup.find_all(['p']):
                # Get all text nodes within the tag while preserving nested elements
                text_nodes = [node for node in tag.descendants if isinstance(
                    node, NavigableString)]

                for node in text_nodes:
                    if translate_limit > 0 and translated_count >= translate_limit:
                        break  # Stop if the translation limit is reached

                    text = node.strip()
                    if text:
                        print(
                            f"Translation ({translated_count + 1}): {text[:50]}...")
                        translated_count += 1

                        json_translation = get_json_translation(text)
                        translated_text = make_string_translation(
                            json_translation)

                        if translated_text:
                            # Replace only the text node, keeping the structure intact
                            node.replace_with(translated_text)

            item.set_content(str(soup).encode('utf-8'))

    # Save the translated EPUB
    epub.write_epub(output_file, book)
    print(f"EPUB successfully translated and saved to {output_file}")
    print(f"Total text blocks translated: {translated_count}")


# Run EPUB translation
if __name__ == "__main__":
    input_file = 'input.epub'
    output_file = 'output.epub'
    translate_limit = 0  # 0 means translate all text blocks

    process_epub(input_file, output_file, translate_limit)

    translations_database_connection.close()

    wordlist_connection.close()
