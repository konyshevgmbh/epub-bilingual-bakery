import sqlite3
import json
import hashlib
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
nltk.download('punkt')

# Initialize SQLite database
translations_database_connection = sqlite3.connect('translations.sqlite')
cursor = translations_database_connection.cursor()
device = "cuda" if torch.cuda.is_available() else "cpu"


def segment_text(text, min_words=5, max_words=15):
    """
    Break text into meaningful segments of 5-15 words.
    Prioritizes splitting at sentence boundaries, then at commas and other punctuation.
    Returns segments with unique tags.
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

# Update the database schema to use a hash key instead of the full text
cursor.execute('''
    CREATE TABLE IF NOT EXISTS translations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text_hash TEXT UNIQUE,
        german_text TEXT,
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
    "CREATE INDEX IF NOT EXISTS idx_translations ON translations(text_hash)")
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

def generate_hash(text):
    """Generate a unique hash for the given text"""
    # Clean the text first to ensure consistent hashing
    cleaned_text = clean_text(text)
    # Create SHA256 hash and return the hexadecimal digest
    return hashlib.sha256(cleaned_text.encode('utf-8')).hexdigest()

def translate_text(german_text, max_repetition_count=3):
    """Translate text using NLLB with repetition handling"""
    cleaned_german_text = clean_text(german_text)
    inputs = tokenizer(cleaned_german_text, return_tensors="pt").to(device)
    
    # Configure generation with parameters to reduce repetition
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("rus_Cyrl"),
        max_length=1500,
        num_beams=5,           # Increase beam search width
        no_repeat_ngram_size=3, # Prevent repeating 3-grams
        repetition_penalty=2.5, # Penalize token repetition
        length_penalty=1.0,     # Avoid generating very short or long outputs
        early_stopping=True     # Stop when all beams reach EOS
    )
    
    translated_text = tokenizer.batch_decode(
        translated_tokens, skip_special_tokens=True)[0]
    
    # Post-processing to remove excessive repetition
    translated_text = fix_repetitions(translated_text, max_repetition_count)
    
    return translated_text

def fix_repetitions(text, max_repetition_count=3):
    """Post-process translated text to fix excessive repetitions"""
    # Split into words
    words = text.split()
    result = []
    
    # Track consecutive occurrences of each word
    repetition_count = {}
    last_word = None
    
    for word in words:
        if word == last_word:
            repetition_count[word] = repetition_count.get(word, 1) + 1
            
            # Skip if we've seen this word too many times consecutively
            if repetition_count[word] > max_repetition_count:
                continue
        else:
            repetition_count = {word: 1}
        
        result.append(word)
        last_word = word
    
    # Check for patterns of repeating phrases (e.g., "hello world hello world hello world")
    final_text = " ".join(result)
    for phrase_length in range(2, 6):  # Check for repeating phrases of lengths 2-5
        for i in range(0, len(result) - phrase_length * 2):
            phrase1 = " ".join(result[i:i+phrase_length])
            phrase2 = " ".join(result[i+phrase_length:i+phrase_length*2])
            
            if phrase1 == phrase2:
                # If we find a repeating phrase pattern, replace it with just one occurrence
                pattern = (phrase1 + " ") * 2 + "+"
                final_text = re.sub(pattern, phrase1 + " ", final_text)
    
    return final_text

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
        if len(keyword) < 3:   
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


def create_tagged_translation_text(segments):
    """Create a single text with tagged segments for translation."""
    tagged_text = ""
    for i, segment in enumerate(segments):
        tag = f"[{i:04d}]"
        tagged_text += f" {tag} {segment} "
    return tagged_text.strip()


def split_translated_text(translated_text):
    """Split the translated text based on tags."""
    pattern = r"\[(\d{4})\](.*?)(?=\[\d{4}\]|$)"
    matches = re.findall(pattern, translated_text, re.DOTALL)
    result = {}
    
    for tag_num, content in matches:
        result[int(tag_num)] = content.strip()
    
    return result


def create_json_translation(german_segments, russian_segments):
    """Create JSON structure with translation using segments."""
    translations = []
    
    for i, german_segment in enumerate(german_segments):
        if i not in russian_segments:
            continue
            
        russian_segment = russian_segments[i]
        
        # Skip empty segments
        if not german_segment.strip() or not russian_segment.strip():
            continue
            
        # Extract keywords for this segment
        keywords = extract_keywords(german_segment, top_n=5)
        
        word_list = []
        for keyword, score in keywords:
            if score > 0.2:  # Use only words with sufficient weight
                translated_keyword = translate_keyword(keyword)
                word_list.append({keyword: translated_keyword})
                
        entry = {
            "ge": german_segment,
            "ru": russian_segment,
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
        
        if len(ge) < 3:   
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
    
    # Rebuild word frequency from existing translations
    cursor.execute("SELECT json_translation FROM translations")
    results = cursor.fetchall()
    
    for result in results:
        try:
            json_data = json.loads(result[0])
            for entry in json_data:
                if 'w' in entry:
                    for word_obj in entry['w']:
                        for german_word, _ in word_obj.items():
                            cursor.execute(
                                "INSERT OR IGNORE INTO word_frequency (word) VALUES (?)", 
                                (german_word,)
                            )
                            cursor.execute(
                                "UPDATE word_frequency SET count = count + 1 WHERE word = ?", 
                                (german_word,)
                            )
        except Exception as e:
            print(f"Error rebuilding word frequency: {e}")
    
    translations_database_connection.commit()


def get_cached_translation(german_text):
    """Check if a translation exists in the database using hash as key"""
    text_hash = generate_hash(german_text)
    print(f"Checking cache for hash {text_hash}.")
  # Ensure a fresh connection
 
    cursor.execute("SELECT json_translation FROM translations WHERE text_hash = ?", (text_hash,))
    result = cursor.fetchone()

    if result:
        return json.loads(result[0])  # Decode JSON properly
    return None



def save_translation_to_db(german_text, json_data):
    cleaned_text = clean_text(german_text)
    text_hash = generate_hash(cleaned_text)

    print(f"Saving translation to database with hash {text_hash}")

    try:
        cursor.execute(
            "INSERT INTO translations (text_hash, german_text, json_translation) VALUES (?, ?, ?) "
            "ON CONFLICT(text_hash) DO UPDATE SET json_translation = excluded.json_translation",
            (text_hash, german_text[:1000], json.dumps(json_data))
        )
        translations_database_connection.commit()  # 🔥 Ensure data is saved
        print("✅ Translation saved successfully!")
    except sqlite3.IntegrityError:
        print("❌ UNIQUE constraint failed! Entry already exists.")

def get_paragraph_translation(paragraph_text):
    """Translate a full paragraph by breaking it into segments, tagging, translating, and reassembling."""
    # Check if we already have this paragraph hash in the database
    cached_translation = get_cached_translation(paragraph_text)
    if cached_translation:
        return cached_translation
        
    # Break paragraph into segments
    segments = segment_text(paragraph_text)
    
    if not segments:
        return []
        
    # If only one segment and it's very short, translate directly
    if len(segments) == 1 and len(segments[0].split()) < 10:
        russian_text = translate_text(segments[0])
        keywords = extract_keywords(segments[0], top_n=5)
        json_data = create_json_translation([segments[0]], {0: russian_text})
        save_translation_to_db(paragraph_text, json_data)
        return json_data
        
    # Create tagged text for translation
    tagged_text = create_tagged_translation_text(segments)
    print(f"Tagged text: {tagged_text[:100]}...")
    # Translate the entire tagged text
    translated_tagged_text = translate_text(tagged_text)
    print(f"Translated text: {translated_tagged_text[:100]}...")
    # Split the translated text back into segments
    translated_segments = split_translated_text(translated_tagged_text)
    
    # Create JSON structure for each segment
    json_data = create_json_translation(segments, translated_segments)
    
    # Save the whole paragraph translation to the database
    save_translation_to_db(paragraph_text, json_data)
    
    return json_data


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

            for tag in soup.find_all(['p']):
                # Skip if we've reached the translation limit
                if translate_limit > 0 and translated_count >= translate_limit:
                    break
                
                # Extract full text from paragraph
                full_text = tag.get_text().strip()
                
                if not full_text:
                    continue
                
                print(f"Translating paragraph ({translated_count + 1}): {full_text[:50]}...")
                translated_count += 1
                
                # Translate the entire paragraph at once
                json_translation = get_paragraph_translation(full_text)
                
                # Create the translated content
                translated_content = make_string_translation(json_translation)
                
                # Replace the original paragraph with the translated version
                if translated_content:
                    new_p = soup.new_tag('p')
                    new_p.append(translated_content)
                    tag.replace_with(new_p)

            item.set_content(str(soup).encode('utf-8'))

    # Save the translated EPUB
    epub.write_epub(output_file, book)
    print(f"EPUB successfully translated and saved to {output_file}")
    print(f"Total paragraphs translated: {translated_count}")

 

# Run EPUB translation
if __name__ == "__main__":
    input_file = 'input.epub'
    output_file = 'output.epub'
    translate_limit = 26  # 0 means translate all text blocks

    process_epub(input_file, output_file, translate_limit)

    translations_database_connection.close()

    wordlist_connection.close()