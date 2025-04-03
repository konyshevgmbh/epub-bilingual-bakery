import logging
from pathlib import Path
from bs4 import BeautifulSoup
import ebooklib

logger = logging.getLogger("ebook_translator")

class EpubProcessor:
    """Processes EPUB files for translation."""

    def __init__(self, config, translation_engine, formatter):
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