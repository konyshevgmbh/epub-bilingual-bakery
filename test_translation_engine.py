import unittest
from unittest.mock import MagicMock
from translation_engine import TranslationEngine
from translation_config import TranslationConfig

class TestTranslationEngine(unittest.TestCase):
    def setUp(self):
        config = TranslationConfig(
            input_file="input.epub",
            output_file="output.epub",
            model_name="facebook/nllb-200-1.3B"
        )
        self.db_manager = MagicMock()
        self.translation_engine = TranslationEngine(config, self.db_manager)

    def test_translate_text(self):
        self.translation_engine.translate_text = MagicMock(return_value="translated text")
        result = self.translation_engine.translate_text("test text")
        self.assertEqual(result, "translated text")

    def test_extract_keywords(self):
        self.db_manager.get_db_word = MagicMock(side_effect=lambda x: x)
        result = self.translation_engine.extract_keywords("This is a test text.")
        self.assertIsInstance(result, list)

if __name__ == '__main__':
    unittest.main()