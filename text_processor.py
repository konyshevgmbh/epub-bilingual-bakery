import re
from typing import List
from nltk.tokenize import sent_tokenize

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