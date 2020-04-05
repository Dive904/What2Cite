import re
from numpy import asarray

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(s):
    """
    This is a function that is used to preprocess a text for an LSTM classification
    :param s: the text
    :return: the preprocessed text
    """
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


def get_embedding_dict(glove_path):
    embeddings_dictionary = dict()

    embedding_file = open(glove_path, encoding="utf8")
    for line in embedding_file:
        records = line.split()
        word = records[0]
        try:
            vector_dimensions = asarray(records[1:], dtype='float32')
            embeddings_dictionary[word] = vector_dimensions
        except ValueError:
            print(" - Value Error", end="")
    embedding_file.close()

    return embeddings_dictionary

