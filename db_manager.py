import json
import sqlite3
import logging
from typing import List, Dict, Optional

logger = logging.getLogger("ebook_translator")

class DBManager:
    """Manages database connections and operations."""

    def __init__(self, translation_db_path: str, wordlist_db_path: str):
        """Initialize database connections."""
        self.translation_db_path = translation_db_path
        self.wordlist_db_path = wordlist_db_path
        self.trans_conn = None
        self.trans_cursor = None
        self.wordlist_conn = None
        self.wordlist_cursor = None
        self._setup_databases()

    def _setup_databases(self):
        """Set up database connections and tables."""
        # Translation database
        self.trans_conn = sqlite3.connect(self.translation_db_path)
        self.trans_cursor = self.trans_conn.cursor()
        
        self.trans_cursor.execute('''
            CREATE TABLE IF NOT EXISTS translations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                german_text TEXT UNIQUE,
                json_translation TEXT
            )
        ''')
        
        self.trans_cursor.execute('''
            CREATE TABLE IF NOT EXISTS word_frequency (
                word TEXT PRIMARY KEY,
                count INTEGER DEFAULT 1
            )
        ''')
        
        self.trans_cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_translations ON translations(german_text)")
        self.trans_cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_word_frequency ON word_frequency(word)")
        self.trans_conn.commit()
        
        # Wordlist database
        self.wordlist_conn = sqlite3.connect(self.wordlist_db_path)
        self.wordlist_cursor = self.wordlist_conn.cursor()

    def reset_word_frequency(self):
        """Reset the word frequency counter."""
        self.trans_cursor.execute("DELETE FROM word_frequency")
        self.trans_conn.commit()
        logger.info("Word frequency counter reset")

    def get_db_word(self, keyword: str) -> str:
        """Get a replacement word from the database if it exists."""
        self.wordlist_cursor.execute(
            "SELECT word FROM words WHERE key = ?", (keyword,))
        result = self.wordlist_cursor.fetchone()
        return result[0] if result else keyword

    def get_translation_from_db(self, text: str) -> Optional[List[Dict]]:
        """Get translation from the database if it exists."""
        self.trans_cursor.execute(
            "SELECT json_translation FROM translations WHERE german_text = ?", (text,))
        result = self.trans_cursor.fetchone()
        return json.loads(result[0]) if result else None

    def save_translation_to_db(self, text: str, translation_data: List[Dict]):
        """Save translation to the database."""
        try:
            self.trans_cursor.execute(
                "INSERT INTO translations (german_text, json_translation) VALUES (?, ?)",
                (text, json.dumps(translation_data))
            )
            self.trans_conn.commit()
        except sqlite3.IntegrityError:
            # Handle case where translation already exists
            logger.debug(f"Translation already exists for: {text[:30]}...")

    def update_word_frequency(self, word: str) -> bool:
        """Update word frequency and return True if it should be included."""
        self.trans_cursor.execute(
            "SELECT count FROM word_frequency WHERE word = ?", (word,))
        result = self.trans_cursor.fetchone()

        if result:
            word_count = result[0]
            if word_count < 3:  # Include word if seen less than 3 times
                self.trans_cursor.execute(
                    "UPDATE word_frequency SET count = count + 1 WHERE word = ?", (word,))
                self.trans_conn.commit()
                return True
            return False
        else:
            # Word seen for the first time
            self.trans_cursor.execute(
                "INSERT INTO word_frequency (word, count) VALUES (?, 1)", (word,))
            self.trans_conn.commit()
            return True

    def close(self):
        """Close database connections."""
        if self.trans_conn:
            self.trans_conn.close()
        if self.wordlist_conn:
            self.wordlist_conn.close()
        logger.info("Database connections closed")