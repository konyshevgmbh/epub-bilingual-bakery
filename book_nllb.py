"""
E-book Translator
A tool for translating e-books with keyword extraction and translation memory.

This module handles translation of EPUB files from German to Russian,
with intelligent segment handling and vocabulary tracking.
"""

import argparse
import json
import logging
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import ebooklib
import nltk
import torch
from bs4 import BeautifulSoup, NavigableString
from ebooklib import epub
from keybert import KeyBERT
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ebook_translator")


@dataclass
class TranslationConfig:
    """Configuration for the translation process."""
    input_file: str
    output_file: str
    src_lang: str = "deu_Latn"
    tgt_lang: str = "rus_Cyrl"
    translation_db_path: str = "translations.sqlite"
    wordlist_db_path: str = "wordlist.sqlite"
    model_name: str = "facebook/nllb-200-1.3B"
    min_segment_words: int = 5
    max_segment_words: int = 15
    keyword_threshold: float = 0.2
    max_keyword_count: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    translate_limit: int = 0  # 0 means translate all text
    log_level: str = "INFO"
    reset_word_frequency: bool = True


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


class TextProcessor:
    """Handles text processing, segmentation and cleaning."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text of special characters."""
        text = text.replace('\u00a0', ' ')
        text = text.replace('\u2013', '-')
        return text.strip()
    
    @staticmethod
    def segment_text(text: str, min_words: int = 5, max_words: int = 15) -> List[str]:
        """
        Break text into meaningful segments of min_words to max_words.
        Prioritizes splitting at sentence boundaries, then at commas and other punctuation.
        """
        # First, split into sentences
        sentences = sent_tokenize(text)
        segments = []
        
        for sentence in sentences:
            # Count words in the sentence
            word_count = len(sentence.split())
            
            # If sentence is within the desired word range, keep it as is
            if min_words <= word_count <= max_words:
                segments.append(sentence)
                continue
                
            # If sentence is too short, include as is
            if word_count < min_words:
                segments.append(sentence)
                continue
                
            # If sentence is too long, split at punctuation
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
                        # If segment doesn't have min_words, use word-by-word splitting
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
        
        # Return only segments that meet criteria
        return [
            segment for segment in segments 
            if segment and (len(segment.split()) >= min_words or len(segment) > 10)
        ]


class TranslationEngine:
    """Handles translation and keyword extraction."""

    def __init__(self, config: TranslationConfig, db_manager: DBManager):
        """Initialize the translation engine."""
        self.config = config
        self.db_manager = db_manager
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')
        
        # Load models
        logger.info(f"Loading translation models on {config.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, 
            src_lang=config.src_lang, 
            tgt_lang=config.tgt_lang
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name).to(config.device)
        self.kw_model = KeyBERT()
        logger.info("Models loaded successfully")

    def translate_text(self, text: str) -> str:
        """Translate text using the loaded model."""
        cleaned_text = TextProcessor.clean_text(text)
        inputs = self.tokenizer(cleaned_text, return_tensors="pt").to(self.config.device)
        
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.config.tgt_lang),
            max_length=500
        )
        
        translated_text = self.tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True)[0]
        return translated_text

    def translate_keyword(self, word: str) -> str:
        """Translate an individual word or phrase."""
        inputs = self.tokenizer(word, return_tensors="pt").to(self.config.device)
        
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.config.tgt_lang),
            max_length=50
        )
        
        translated_word = self.tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True)[0]
        return translated_word

    def extract_keywords(self, text: str) -> List[Tuple[str, float]]:
        """Extract keywords from text."""
        if len(text.split()) < 5:
            return []  # Skip short texts

        keywords = self.kw_model.extract_keywords(text, top_n=self.config.max_keyword_count)

        updated_keywords = []
        for keyword, score in keywords:
            if len(keyword) < 3 or keyword.isdigit():
                continue
                
            updated_keyword = self.db_manager.get_db_word(keyword)
            if updated_keyword == '-':
                continue
                
            updated_keywords.append((updated_keyword, score))

        return updated_keywords

    def get_translation(self, text: str) -> List[Dict]:
        """Get translation for a segment of text."""
        cleaned_text = TextProcessor.clean_text(text)
        
        # Skip short or number-only text
        if cleaned_text.isdigit() or len(cleaned_text) < 3:
            return [{"ge": cleaned_text, "ru": cleaned_text, "w": []}]

        # Check if translation exists in database
        existing_translation = self.db_manager.get_translation_from_db(cleaned_text)
        if existing_translation:
            return existing_translation

        try:
            # Translate the text
            russian_text = self.translate_text(cleaned_text)

            # Extract keywords
            keywords = self.extract_keywords(cleaned_text)

            # Create translation data structure
            translation_data = self._create_translation_data(cleaned_text, russian_text, keywords)

            # Save to database
            self.db_manager.save_translation_to_db(cleaned_text, translation_data)

            return translation_data

        except Exception as e:
            logger.error(f"Translation error: {e}")
            return []

    def _create_translation_data(self, source_text: str, target_text: str, 
                                keywords: List[Tuple[str, float]]) -> List[Dict]:
        """Create structured translation data with keywords."""
        word_list = []
        
        for keyword, score in keywords:
            if score > self.config.keyword_threshold:
                translated_keyword = self.translate_keyword(keyword)
                word_list.append({keyword: translated_keyword})

        entry = {
            "ge": source_text,
            "ru": target_text,
            "w": word_list
        }
        
        return [entry]


class HTMLFormatter:
    """Formats translations into HTML."""

    def __init__(self, db_manager: DBManager):
        """Initialize the formatter."""
        self.db_manager = db_manager

    def _filter_words(self, word_objects: List[Dict]) -> List[Dict]:
        """Filter words based on frequency."""
        filtered_words = []

        for word_obj in word_objects:
            if isinstance(word_obj, str):
                continue
                
            for german_word, translation in word_obj.items():
                # Check if word should be included based on frequency
                if self.db_manager.update_word_frequency(german_word):
                    filtered_words.append(word_obj)

        return filtered_words

    def format_translation(self, json_translation: List[Dict]) -> Optional[BeautifulSoup]:
        """Format translation as HTML."""
        if not json_translation:
            return None

        soup = BeautifulSoup("", "html.parser")

        for entry in json_translation:
            ge = entry.get("ge", "").strip()
            ru = entry.get("ru", "").strip()
            w = entry.get("w", [])

            if not ge or not ru or len(ge) < 3 or ge.isdigit():
                if ge:  # If it's a number or very short, just add it as plain text
                    soup.append(" " + ge + " ")
                continue

            filtered_w = self._filter_words(w)

            # If source and target are identical or target is empty
            if ge == ru or not ru:
                soup.append(ge)
                continue

            # Format the translation
            bold_ge = soup.new_tag("b")
            bold_ge.string = ge

            italic_ru = soup.new_tag("i")
            italic_ru.string = ru

            translation_part = soup.new_tag("span")
            translation_part.append(bold_ge)
            translation_part.append(f" ( ")
            translation_part.append(italic_ru)

            # Add word translations if available
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


class EpubProcessor:
    """Processes EPUB files for translation."""

    def __init__(self, config: TranslationConfig, translation_engine: TranslationEngine, 
                 formatter: HTMLFormatter):
        """Initialize the EPUB processor."""
        self.config = config
        self.translation_engine = translation_engine
        self.formatter = formatter
        self.translated_count = 0

    def process_file(self):
        """Process an EPUB file, translating its content."""
        logger.info(f"Processing file {self.config.input_file}...")
        
        # Ensure input file exists
        input_path = Path(self.config.input_file)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return
            
        # Read the EPUB
        book = epub.read_epub(str(input_path))
        
        # Process each document in the EPUB
        for item in book.items:
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                self._process_document(item)
                
                # Check if translation limit has been reached
                if (self.config.translate_limit > 0 and 
                    self.translated_count >= self.config.translate_limit):
                    logger.info(f"Translation limit of {self.config.translate_limit} reached")
                    break
        
        # Save the translated EPUB
        epub.write_epub(self.config.output_file, book)
        logger.info(f"EPUB successfully translated and saved to {self.config.output_file}")
        logger.info(f"Total text blocks translated: {self.translated_count}")

    def _process_document(self, item):
        """Process a single document within the EPUB."""
        logger.info("Processing document...")
        soup = BeautifulSoup(item.get_content(), 'html.parser')

        # Process each paragraph
        for tag in soup.find_all(['p']):
            # Extract text from paragraph
            full_text = tag.get_text()
            
            if not full_text.strip():
                continue
            
            # Create segments from the paragraph text
            segments = TextProcessor.segment_text(
                full_text, 
                min_words=self.config.min_segment_words, 
                max_words=self.config.max_segment_words
            )
            
            # Create a new paragraph to replace the old one
            new_p = soup.new_tag('p')
            
            # Process each segment
            for segment in segments:
                # Check if translation limit reached
                if (self.config.translate_limit > 0 and 
                    self.translated_count >= self.config.translate_limit):
                    break
                
                logger.info(f"Translation ({self.translated_count + 1}): {segment[:50]}...")
                self.translated_count += 1
                
                # Get and format the translation
                json_translation = self.translation_engine.get_translation(segment)
                translated_segment = self.formatter.format_translation(json_translation)
                
                if translated_segment:
                    new_p.append(translated_segment)
                    # Add space between segments
                    new_p.append(" ")
            
            # Replace original paragraph with translated one
            tag.replace_with(new_p)

        # Update the document content
        item.set_content(str(soup).encode('utf-8'))


def setup_argument_parser():
    """Set up and return the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='E-book Translator - Translate EPUB files with vocabulary tracking',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('input_file', help='Path to input EPUB file')
    parser.add_argument('output_file', help='Path to save translated EPUB file')
    
    # Optional arguments
    parser.add_argument('--src-lang', default='deu_Latn', help='Source language code')
    parser.add_argument('--tgt-lang', default='rus_Cyrl', help='Target language code')
    parser.add_argument('--translation-db', help='Path to translation database')
    parser.add_argument('--wordlist-db', default='wordlist.sqlite', 
                        help='Path to wordlist database')
    parser.add_argument('--model', default='facebook/nllb-200-1.3B', 
                        help='Translation model name')
    parser.add_argument('--min-words', type=int, default=5, 
                        help='Minimum words in a segment')
    parser.add_argument('--max-words', type=int, default=15, 
                        help='Maximum words in a segment')
    parser.add_argument('--keyword-threshold', type=float, default=0.2, 
                        help='Minimum score for keywords')
    parser.add_argument('--max-keywords', type=int, default=10, 
                        help='Maximum number of keywords per segment')
    parser.add_argument('--device', choices=['cuda', 'cpu'], 
                        default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Computation device')
    parser.add_argument('--limit', type=int, default=0, 
                        help='Limit number of translations (0 for unlimited)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        default='INFO', help='Logging level')
    parser.add_argument('--no-reset-frequency', action='store_true', 
                        help="Don't reset word frequency counter")
    
    return parser


def configure_logging(log_level):
    """Configure logging with the specified level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logger.setLevel(numeric_level)
    
    # Update handlers to use the new level
    for handler in logger.handlers:
        handler.setLevel(numeric_level)


def config_from_args(args):
    """Create a TranslationConfig object from command-line arguments."""
    translation_db_path = args.translation_db if args.translation_db else Path(args.input_file).with_suffix('.sqlite')
    return TranslationConfig(
        input_file=args.input_file,
        output_file=args.output_file,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        translation_db_path=translation_db_path,  
        wordlist_db_path=args.wordlist_db,
        model_name=args.model,
        min_segment_words=args.min_words,
        max_segment_words=args.max_words,
        keyword_threshold=args.keyword_threshold,
        max_keyword_count=args.max_keywords,
        device=args.device,
        translate_limit=args.limit,
        log_level=args.log_level,
        reset_word_frequency=not args.no_reset_frequency
    )


def main():
    """Main function to run the translator."""
    # Parse command-line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Create configuration
    config = config_from_args(args)
    
    # Configure logging
    configure_logging(config.log_level)
    
    # Initialize components
    db_manager = DBManager(config.translation_db_path, config.wordlist_db_path)
    
    try:
        # Reset word frequency counter if configured
        if config.reset_word_frequency:
            db_manager.reset_word_frequency()
        
        # Initialize translation engine
        translation_engine = TranslationEngine(config, db_manager)
        
        # Initialize HTML formatter
        formatter = HTMLFormatter(db_manager)
        
        # Process the EPUB file
        processor = EpubProcessor(config, translation_engine, formatter)
        processor.process_file()
        
    except KeyboardInterrupt:
        logger.info("Translation interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Error during translation: {e}", exc_info=True)
        return 1
        
    finally:
        # Close database connections
        db_manager.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())