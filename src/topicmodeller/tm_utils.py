from nltk.stem import WordNetLemmatizer
from langdetect import detect
import os
import io

from src.fileutils import file_abstract


def extract_abstract(dir_path, start=None, end=None):
    files = os.listdir(dir_path)
    ris = []
    if start is None and end is None:
        for file in files:
            ris += file_abstract.txt_only_abstract_reader(dir_path + file)
    else:
        for file in files[start:end]:
            ris += file_abstract.txt_only_abstract_reader(dir_path + file)

    return ris


def preprocess_abstract(abstracts):
    additional_stopwords = ["paper", "method", "large", "model", "proposed", "study", "based", "using", "approach"]
    ris = []
    lem = WordNetLemmatizer()
    for abstract in abstracts:
        lang = "x"
        try:
            lang = detect(abstract)
        except:
            lang = "x"

        if lang == "en":
            tmp = []
            a = abstract.split()
            for word in a:
                w = word.lower()  # Convert to lowercase
                w = lem.lemmatize(w)  # Lemmatize word
                if w not in additional_stopwords:
                    tmp.append(w)
            ris.append(" ".join(tmp))
    return ris


def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


def print_topics_in_file(model, count_vectorizer, n_top_words, filename):
    words = count_vectorizer.get_feature_names()
    with io.open(filename, "a+", encoding="utf-8") as f:
        for topic_idx, topic in enumerate(model.components_):
            f.write("\nTopic #%d: " % topic_idx)
            f.write(" ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
