from nltk.stem.snowball import EnglishStemmer
from gensim.parsing.preprocessing import remove_stopwords
from nltk import word_tokenize
import nltk  # Import NLTK (Natural Language Toolkit) for natural language processing tasks
from typing import Union

# Download the required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Define Lemmatizer
lemma = EnglishStemmer()

def preprocess_text(text: Union[str, float]) -> str:
    if isinstance(text, float):
        # Return an empty string for missing or NaN values
        return ""

    # Remove URLs
    text = ' '.join(word for word in text.split() if not word.startswith('http'))
    text = ' '.join(word for word in text.split() if not word.startswith('www'))

    # Remove special characters and punctuation
    text = ''.join(char for char in text if char.isalnum() or char.isspace())

    # Remove mentions (@username)
    text = ' '.join(word for word in text.split() if not word.startswith('@'))

    # Remove hashtags (#technology)
    text = ' '.join(word[1:] if word.startswith('#') else word for word in text.split())

    # Removing stopwords and lemmatization
   # text = remove_stopwords(text.lower())
    text = word_tokenize(text)
    text = ' '.join([lemma.stem(word) for word in text])

    return text
