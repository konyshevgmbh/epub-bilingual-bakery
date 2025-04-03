import unittest
from translation_config import TranslationConfig

class TestTranslationConfig(unittest.TestCase):
    def test_translation_config_initialization(self):
        config = TranslationConfig(
            input_file="input.epub",
            output_file="output.epub"
        )
        self.assertEqual(config.input_file, "input.epub")
        self.assertEqual(config.output_file, "output.epub")
        self.assertEqual(config.src_lang, "deu_Latn")
        self.assertEqual(config.tgt_lang, "rus_Cyrl")

if __name__ == '__main__':
    unittest.main()