import sqlite3
import ollama
import json
import time
from bs4 import BeautifulSoup
from ebooklib import epub
import ebooklib
import torch
from collections import Counter

# Initialize SQLite database
conn = sqlite3.connect('translations.sqlite')
cursor = conn.cursor()
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

cursor.execute("CREATE INDEX IF NOT EXISTS idx_translations ON translations(german_text)")

cursor.execute("CREATE INDEX IF NOT EXISTS idx_word_frequency ON word_frequency(word)")
conn.commit()

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

# Main prompt that is passed once
BASE_PROMPT = '''
    Ihre Aufgabe besteht darin, einen hochwertigen Übersetzungstext vom Deutschen ins Russische in einem strukturierten JSON-Format zu erstellen.  

**Anforderungen an die Übersetzung:**  
1. **Natürlichkeit** – Die Übersetzung sollte im Russischen natürlich klingen und den Kontext berücksichtigen.  
2. **Strukturerhaltung** – Lange Sätze dürfen in sinnvolle Teile zerlegt werden, aber die Reihenfolge soll erhalten bleiben.  
3. **Schlüsselvokabular** – Alle Wörter aus dem deutschen Satz müssen in der Wortliste („w“) enthalten sein, mit ihrer Grundform und Bedeutung (ohne Eigennamen).  
4. **Zeichensetzung** – Die Übersetzung sollte die ursprüngliche Interpunktion und Satzstruktur respektieren.  

**Beispiel:**  
**Eingabetext:**  
Die dunklen Wolken am Himmel zogen langsam weiter, während ein kalter Wind durch die Straßen pfiff.

**Ausgabe-JSON:**  
```json
{
  "translations": [
    {
      "ge": "Die dunklen Wolken am Himmel zogen langsam weiter",
      "ru": "Тёмные облака в небе медленно проплывали",
      "w": [
        {"dunkel": "тёмный"},
        {"die Wolke": "облако"},
        {"der Himmel": "небо"},
        {"ziehen": "тянуть, двигаться"},
        {"langsam": "медленно"},
        {"weiter": "дальше"}
      ]
    },
    {
      "ge": ", während ein kalter Wind durch die Straßen pfiff.",
      "ru": ", а холодный ветер свистел в переулках.",
      "w": [
        {"während": "в то время как"},
        {"kalt": "холодный"},
        {"der Wind": "ветер"},
        {"durch": "через, сквозь"},
        {"die Straße": "улица"},
        {"pfeifen": "свистеть"}
      ]
    }
  ]
}
'''

def parse_json_response(response_text):
    try:
        json_obj = json.loads(response_text)
        return json_obj.get("translations", [])
    except json.JSONDecodeError:
        print("Error parsing JSON.")
        return []

def initialize_word_frequency():
    """Initialize word frequency counter from existing translations in the database"""
    # First, clear the word frequency table
    cursor.execute("DELETE FROM word_frequency")
    conn.commit()
    
    # Get all existing translations
    cursor.execute("SELECT json_translation FROM translations")
    existing_translations = cursor.fetchall()
    
    word_counts = Counter()
    
    # Count word frequency in all existing translations
    for row in existing_translations:
        try:
            json_translations = json.loads(row[0])
            for entry in json_translations:
                w_list = entry.get("w", [])
                for word_obj in w_list:
                    if isinstance( word_obj, str):
                        continue
                    for german_word in word_obj.keys():
                        word_counts[german_word] += 1
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error processing existing translation: {e}")
    
    # Save word frequency to the database
    for word, count in word_counts.items():
        cursor.execute("INSERT INTO word_frequency (word, count) VALUES (?, ?)", (word, count))
    
    conn.commit()
    print(f"Word frequency counter initialized: processed {len(word_counts)} unique words")

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
            cursor.execute("SELECT count FROM word_frequency WHERE word = ?", (german_word,))
            result = cursor.fetchone()
            
            if result:
                word_count = result[0]
                if word_count < 3:  # Add the word only if it has been encountered less than 3 times
                    filtered_words.append(word_obj)
                    # Increment the word counter
                    cursor.execute("UPDATE word_frequency SET count = count + 1 WHERE word = ?", (german_word,))
                    conn.commit()
            else:
                # If the word is encountered for the first time, add it to the database and the filtered list
                filtered_words.append(word_obj)
                cursor.execute("INSERT INTO word_frequency (word, count) VALUES (?, 1)", (german_word,))
                conn.commit()
                
    return filtered_words


def make_string_translation(json_translation):
    if not json_translation:
        return None
    
    soup = BeautifulSoup("", "html.parser")

    for entry in json_translation:
        ge = entry.get("ge", "")
        ru = entry.get("ru", "")
        w = entry.get("w", [])
        
        if not ge or not ru:
            continue
        
        # Filter the word list before displaying
        filtered_w = filter_words(w)
        
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
def correct_translation(input_text, translated_json):
    """Ensure the translation is relevant and matches expected structure."""
    if not translated_json:
        return translated_json

    # Split input into words
    input_words = input_text.split()
    if len(input_words) <= 3:
        if len(translated_json) != 1 or translated_json[0].get("ge", "").split() != input_words :
            print(f"Translation mismatch: {input_text} -> {translated_json}")
            return [{"ge": input_text, "ru": input_text, "w": []}]
    return translated_json
def get_json_translation(german_text, retry_limit=6):
    cursor.execute("SELECT json_translation FROM translations WHERE german_text = ?", (german_text,))
    result = cursor.fetchone()
    if result:
        return json.loads(result[0])

    translated_data = []
    
# Parameters for the API request
    params = {
        'temperature': 0.8,   
        'top_p': 0.9,
        'num_ctx': 16384
    }
    for attempt in range(retry_limit):
# Adjust temperature and top_p for each attempt
        params['temperature'] = min(0.8 + attempt * 0.05, 1.0)   
        params['top_p'] = max(0.9 - attempt * 0.1, 0.5)   
        messages = [
            {"role": "system", "content": BASE_PROMPT},
            {"role": "user", "content": german_text}
        ]
        
        try:
            response = ollama.chat(
                model='gemma3:12b',
                messages=messages,
                format='json',
                options={
                    "device": device,
                    "temperature": params['temperature'],
                    "top_p": params['top_p'] ,
                    "num_ctx": params['num_ctx']
                }
            )
            response_text = response['message']['content']
            print(response_text)
            json_translation = parse_json_response(response_text)

            json_translation = correct_translation(german_text, json_translation)
            
            if json_translation:
                translated_data = json_translation
                break
            else:
                print(f"Attempt {attempt + 1} failed. Retrying with params: {params}")

        except Exception as e:
            print(f"API Error: {e}")
    
    if not translated_data:
        print("Failed to translate the text.")
        return []
    
    cursor.execute("INSERT INTO translations (german_text, json_translation) VALUES (?, ?)",
                  (german_text, json.dumps(translated_data)))
    conn.commit()
    return translated_data


# Initialize word frequency counter when the script starts
print("Initializing word frequency counter from existing translations...")
initialize_word_frequency()

# Read and translate EPUB
input_file = 'input.epub'
output_file = 'output.epub'
book = epub.read_epub(input_file )
translated_count = 0
translated_limit = 0
for item in book.items:
    if item.get_type() == ebooklib.ITEM_DOCUMENT:
        soup = BeautifulSoup(item.get_content(), 'html.parser')

        for tag_name in ['p']:
            for tag in soup.find_all(tag_name):
                if translated_limit>0 and translated_count > translated_limit:
                    break
                if tag.string:
                    print(translated_count)
                    translated_count += 1
                    json_translation = get_json_translation(tag.string.strip())
                    translated_soup = make_string_translation(json_translation)

                    if translated_soup:
                        tag.clear()
                        tag.append(translated_soup)

        item.set_content(str(soup).encode('utf-8'))

# Save translated EPUB
epub.write_epub(output_file, book )
print(f"EPUB successfully translated and saved to {output_file}")
conn.close()