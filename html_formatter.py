from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger("ebook_translator")

class HTMLFormatter:
    """Formats translations into HTML."""

    def __init__(self, db_manager):
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