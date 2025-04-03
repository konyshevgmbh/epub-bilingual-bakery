import unittest
from unittest.mock import MagicMock
from html_formatter import HTMLFormatter

class TestHTMLFormatter(unittest.TestCase):
    def setUp(self):
        self.db_manager = MagicMock()
        self.html_formatter = HTMLFormatter(self.db_manager)

    def test_format_translation(self):
        json_translation = [{"ge": "test", "ru": "тест", "w": []}]
        result = self.html_formatter.format_translation(json_translation)
        self.assertIsNotNone(result)
        self.assertIn("test", str(result))
        self.assertIn("тест", str(result))

    def test_filter_words(self):
        self.db_manager.update_word_frequency = MagicMock(return_value=True)
        word_objects = [{"test": "тест"}]
        result = self.html_formatter._filter_words(word_objects)
        self.assertEqual(len(result), 1)

if __name__ == '__main__':
    unittest.main()