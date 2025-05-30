import json
import sqlite3
import os

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

# Reading data from a JSON file
try:
    with open('german_nouns_output.json', 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    print(f"Successfully loaded {len(json_data)} records from german_nouns_output.json")
except FileNotFoundError:
    print("Error: File german_nouns_output.json not found")
    exit(1)
except json.JSONDecodeError:
    print("Error: Invalid JSON format in file german_nouns_output.json")
    exit(1)

# Name of the embedded database
db_filename = 'wordlist.sqlite'

# Deleting the existing database if it exists
if os.path.exists(db_filename):
    os.remove(db_filename)

# Connecting to the embedded SQLite database
conn = sqlite3.connect(db_filename)
cursor = conn.cursor()

# Creating a table with the required structure
cursor.execute('''
    CREATE TABLE words (
        key TEXT,
        word TEXT
    )
''')

# Processing and inserting data
inserted_count = 0
# Add elementary words to database with empty word value
for elementary_word in ELEMENTARY_WORDS:
    cursor.execute("INSERT INTO words (key, word) VALUES (?, ?)", (elementary_word, "-"))
    inserted_count += 1

for item in json_data:
    if "germanNoun" in item and "gender" in item:
        key = item["germanNoun"].lower()
        word = f"{item['gender']} {item['germanNoun']}"
        cursor.execute("INSERT INTO words (key, word) VALUES (?, ?)", (key, word))
        inserted_count += 1

print(f"Inserted {inserted_count} records into the database")
 
# Creating an index to speed up searches
cursor.execute("CREATE INDEX idx_key ON words(key)")
print("Created index on the key column to speed up searches")

# Committing changes and closing the connection
conn.commit()
conn.close()
print(f"Database saved to file: {db_filename}")