import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
porter = PorterStemmer()


def preprocess_text(s):
    s_stemmed = []
    for w in s.split():
        s_stemmed.append(porter.stem(w))

    s = ' '.join(s_stemmed)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', s)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    word_tokens = word_tokenize(s)

    filtered_sentence = [w for w in word_tokens if w not in stop_words]
    filtered_sentence = ' '.join(filtered_sentence)

    return filtered_sentence
