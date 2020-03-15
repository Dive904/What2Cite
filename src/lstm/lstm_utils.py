import re
import numpy as np

from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def texts_to_sequence(vectorizer, sentences, index_dict):
    text_sequence = []
    for sentence in sentences:
        tmp_sequence = []
        bigrams = list(ngrams(sentence.split(), 2))
        for bigram in bigrams:
            b = bigram[0] + " " + bigram[1]
            if index_dict.get(b) is not None:
                index_to_add = vectorizer.vocabulary_.get(b)
            else:
                index_to_add = vectorizer.vocabulary_.get(bigram[0])
            if index_to_add is not None:
                tmp_sequence.append(index_to_add)
        text_sequence.append(tmp_sequence)

    return np.asarray(text_sequence)


def preprocess_text(s):
    s = s.lower()

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', s)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    s_lemmatized = []
    for w in sentence.split():
        s_lemmatized.append(lemmatizer.lemmatize(w))

    s = ' '.join(s_lemmatized)

    word_tokens = word_tokenize(s)

    filtered_sentence = [w for w in word_tokens if w not in stop_words]
    filtered_sentence = ' '.join(filtered_sentence)

    return filtered_sentence
