import unittest
from unittest.mock import MagicMock
from db_manager import DBManager

class TestDBManager(unittest.TestCase):
    def setUp(self):
        self.db_manager = DBManager("test_translations.sqlite", "test_wordlist.sqlite")

    def test_save_translation_to_db(self):
        self.db_manager.save_translation_to_db("test", [{"ge": "test", "ru": "тест"}])
        result = self.db_manager.get_translation_from_db("test")
        self.assertIsNotNone(result)
        self.assertEqual(result[0]["ge"], "test")
        self.assertEqual(result[0]["ru"], "тест")

    def test_get_db_word(self):
        self.db_manager.wordlist_cursor.execute("INSERT INTO words (key, word) VALUES (?, ?)", ("test_key", "test_word"))
        self.db_manager.wordlist_conn.commit()
        result = self.db_manager.get_db_word("test_key")
        self.assertEqual(result, "test_word")

if __name__ == '__main__':
    unittest.main()