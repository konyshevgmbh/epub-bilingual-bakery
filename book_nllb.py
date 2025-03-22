import sqlite3
import json
from bs4 import BeautifulSoup, NavigableString
from ebooklib import epub
import ebooklib
import torch
from collections import Counter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from keybert import KeyBERT
import nltk
from nltk.tokenize import sent_tokenize
import re


def segment_text(text, min_words=5, max_words=15):
    """
    Break text into meaningful segments of 5-15 words.
    Prioritizes splitting at sentence boundaries, then at commas and other punctuation.
    """
    # First, split into sentences
    sentences = sent_tokenize(text)
    segments = []
    
    for sentence in sentences:
        # If sentence is already in the desired word range, keep it as is
        word_count = len(sentence.split())
        if min_words <= word_count <= max_words:
            segments.append(sentence)
            continue
            
        # If sentence is too short, we'll handle it separately
        if word_count < min_words:
            segments.append(sentence)
            continue
            
        # If sentence is too long, split at commas, semicolons, etc.
        clause_separators = re.split(r'([,;:])', sentence)
        current_segment = ""
        current_word_count = 0
        
        for i in range(len(clause_separators)):
            part = clause_separators[i]
            part_word_count = len(part.split())
            
            # If this is a punctuation separator (single item)
            if part_word_count == 0 and part in [',', ';', ':']:
                current_segment += part
                continue
                
            # Check if adding this part would exceed max_words
            if current_word_count + part_word_count > max_words:
                # If current segment is not empty and has enough words, add it
                if current_segment and len(current_segment.split()) >= min_words:
                    segments.append(current_segment.strip())
                    current_segment = part
                    current_word_count = part_word_count
                else:
                    # If we can't create a segment with min_words, use simpler splitting
                    words = part.split()
                    temp_segment = current_segment
                    for word in words:
                        if len(temp_segment.split()) + 1 > max_words:
                            segments.append(temp_segment.strip())
                            temp_segment = word
                        else:
                            temp_segment += " " + word
                    current_segment = temp_segment
                    current_word_count = len(current_segment.split())
            else:
                # Add part to current segment
                if current_segment:
                    current_segment += part
                else:
                    current_segment = part
                current_word_count += part_word_count
        
        # Add the last segment if it's not empty
        if current_segment:
            segments.append(current_segment.strip())
    
    # Final cleanup of segments
    return [segment for segment in segments if segment and len(segment.split()) >= min_words or len(segment) > 10]

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
        if keyword.isdigit():   
            continue
        updated_keyword = get_db_word(keyword)
        if(updated_keyword == '-'):
            continue
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

    translations = []
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
    
    return translations


def filter_words(word_objects):
    filtered_words = []

    for word_obj in word_objects:
        if isinstance(word_obj, str):
            continue
        for german_word, translation in word_obj.items():

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


def process_epub(input_file, output_file, translate_limit=0):
    print(f"Processing file {input_file}...")
    book = epub.read_epub(input_file)
    translated_count = 0

    for item in book.items:
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            print(f"Processing document...")
            soup = BeautifulSoup(item.get_content(), 'html.parser')

            for tag in soup.find_all(['p']):
                # Extract full text from paragraph, ignoring individual elements
                full_text = tag.get_text()
                
                if not full_text.strip():
                    continue
                
                # Get segments from the full text
                segments = segment_text(full_text)
                
                # Create a new paragraph to replace the old one
                new_p = soup.new_tag('p')
                
                for segment in segments:
                    if translate_limit > 0 and translated_count >= translate_limit:
                        break  # Stop if the translation limit is reached
                    
                    print(f"Translation ({translated_count + 1}): {segment[:50]}...")
                    translated_count += 1
                    
                    json_translation = get_json_translation(segment)
                    translated_segment = make_string_translation(json_translation)
                    
                    if translated_segment:
                        new_p.append(translated_segment)
                        # Add a space between segments for readability
                        new_p.append(" ")
                
                # Replace the original paragraph with the new one
                tag.replace_with(new_p)

            item.set_content(str(soup).encode('utf-8'))

    # Save the translated EPUB
    epub.write_epub(output_file, book)
    print(f"EPUB successfully translated and saved to {output_file}")
    print(f"Total text blocks translated: {translated_count}")


# Initialiаze punkt tokenizer
nltk.download('punkt')
# Initialiаze SQLite database
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

# Loading models
print("Loading NLLB and KeyBERT models...")
tokenizer = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-1.3B", src_lang="deu_Latn", tgt_lang="rus_Cyrl")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-1.3B").to(device)
kw_model = KeyBERT()
print("Models loaded")

wordlist_connection = sqlite3.connect('wordlist.sqlite')
wordlist_cursor = wordlist_connection.cursor()

# Initialize word frequency counter when the script starts
print("Initializing word frequency counter from existing translations...")
initialize_word_frequency()

# Run EPUB translation
if __name__ == "__main__":
    input_file = 'input.epub'
    output_file = 'output.epub'
    translate_limit = 0  # 0 means translate all text blocks

    process_epub(input_file, output_file, translate_limit)

    translations_database_connection.close()

    wordlist_connection.close()
