import json
import logging
from typing import List, Dict, Tuple
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from keybert import KeyBERT
from text_processor import TextProcessor

logger = logging.getLogger("ebook_translator")

class TranslationEngine:
    """Handles translation and keyword extraction."""

    def __init__(self, config, db_manager):
        """Initialize the translation engine."""
        self.config = config
        self.db_manager = db_manager
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
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