# Importing relevant packages

import pickle
from nltk.stem.snowball import EnglishStemmer
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
import nltk  # Import NLTK (Natural Language Toolkit) for natural language processing tasks



def preprocess_text(text):
    # Instantiating our lemmatizer
    lemma = EnglishStemmer()
    # Remove URLs
    text = ' '.join(word for word in text.split() if not word.startswith('http'))
    text = ' '.join(word for word in text.split() if not word.startswith('www'))

    # Remove special characters and punctuation
    text = ''.join(char for char in text if char.isalnum() or char.isspace())

    # Remove mentions (@username)
    text = ' '.join(word for word in text.split() if not word.startswith('@'))

    # Remove hashtags (#technology)
    text = ' '.join(word[1:] if word.startswith('#') else word for word in text.split())

    # Removing stopwords
    ## NB: Remember to convert the text into thier lowercase form so that for example "I" will be exactly the same as "i"
    text = remove_stopwords(text.lower())

    # Tokenization
    text = word_tokenize(text)

    #lemmatization
    text = ' '.join([lemma.stem(word) for word in text])

    # Splitting the text back
    text = word_tokenize(text)
    return text
