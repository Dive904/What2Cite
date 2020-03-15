import re
import numpy as np

from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def create_count_vectorizer(sentences, index_dict):
    for i in range(len(sentences)):
        sentences[i] = preprocess_text(sentences[i])

    texts_to_fit = []
    for t in sentences:
        bigrams = list(ngrams(t.split(), 2))
        for b in bigrams:
            bi = b[0] + " " + b[1]
            if index_dict.get(bi) is not None:
                texts_to_fit.append(bi)

    vectorizer = CountVectorizer(ngram_range=(1, 2))
    vectorizer.fit_transform(texts_to_fit)

    return vectorizer


def clean_vectorizer_vocabulary(vectorizer, index_dict):
    to_remove = []
    for key in vectorizer.vocabulary_.keys():
        if index_dict.get(key) is None:
            to_remove.append(key)

    for r in to_remove:
        del vectorizer.vocabulary_[r]

    return vectorizer


def max_len_sequence(sequences):
    return max([len(l) for l in sequences])


def pad_sequences(sequences, max_len):
    for i in range(len(sequences)):
        if len(sequences[i]) < max_len:
            sequences[i] = sequences[i] + [0] * (max_len - len(sequences[i]))
        elif len(sequences[i]) > max_len:
            del sequences[i][max_len:]

    return sequences


def texts_to_sequence(vectorizer, sentences, index_dict):
    text_sequence = []
    for sentence in sentences:
        tmp_sequence = []
        bigrams = list(ngrams(sentence.split(), 2))
        for bigram in bigrams:
            b = bigram[0] + " " + bigram[1]
            index_to_add = None
            if index_dict.get(b) is not None:
                index_to_add = vectorizer.vocabulary_.get(b)
            elif index_dict.get(bigram[0]) is not None:
                index_to_add = vectorizer.vocabulary_.get(bigram[0])
            if index_to_add is not None:
                tmp_sequence.append(index_to_add)
        text_sequence.append(tmp_sequence)

    return text_sequence


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
