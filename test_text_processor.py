import unittest
from text_processor import TextProcessor

class TestTextProcessor(unittest.TestCase):
    def test_clean_text(self):
        text = "Hello, \u00a0World!\u2013"
        cleaned = TextProcessor.clean_text(text)
        self.assertEqual(cleaned, "Hello, World!-")

    def test_segment_text(self):
        text = "This is a test. This is only a test."
        segments = TextProcessor.segment_text(text, min_words=3, max_words=5)
        self.assertEqual(len(segments), 3)
        self.assertIn("This is a test.", segments)

if __name__ == '__main__':
    unittest.main()