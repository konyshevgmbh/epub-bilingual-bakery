from dataclasses import dataclass

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