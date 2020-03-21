from nltk.stem import WordNetLemmatizer
import os
import io
import re
import operator
import numpy as np

from src.fileutils import file_abstract
from src.lstm import lstm_utils


def extract_only_abstract(dir_path, start=None, end=None):
    files = os.listdir(dir_path)
    ris = []

    if start is None:
        start = 0
    if end is None:
        end = len(files) - 1

    for file in files[start:end]:
        ris += file_abstract.txt_only_abstract_reader(dir_path + file)

    return ris


def extract_only_citations(dir_path, start=None, end=None):
    files = os.listdir(dir_path)
    ris = []

    if start is None:
        start = 0
    if end is None:
        end = len(files) - 1

    for file in files[start:end]:
        ris += file_abstract.txt_only_citations_reader(dir_path + file)

    return ris


def extract_paper_info(dir_path, start=None, end=None):
    files = os.listdir(dir_path)
    ris = []

    if start is None:
        start = 0
    if end is None:
        end = len(files) - 1

    for file in files[start:end]:
        ris += file_abstract.txt_dataset_reader(dir_path + file)

    return ris


def preprocess_abstract(abstracts):
    additional_stopwords = ["paper", "method", "large", "model", "proposed", "study",
                            "based", "using", "approach", "data", "result", "ha", "wa"]
    ris = []

    if isinstance(abstracts, list):
        for a in abstracts:
            x = lstm_utils.preprocess_text(a)
            y = []
            for w in x.split():
                if w not in additional_stopwords:
                    y.append(w)
            x = " ".join(y)
            ris.append(x)
    else:
        x = lstm_utils.preprocess_text(abstracts)
        y = []
        for w in x.split():
            if w not in additional_stopwords:
                y.append(w)
        ris = " ".join(y)

    return ris


def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


def print_topics_in_file(model, count_vectorizer, n_top_words, filename, mode="a+"):
    words = count_vectorizer.get_feature_names()
    with io.open(filename, mode, encoding="utf-8") as f:
        for topic_idx, topic in enumerate(model.components_):
            f.write("\nTopic #%d: " % topic_idx)
            f.write(" ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))


def find_n_maximum(items, n):
    indexed = list(enumerate(items))
    sorted_list = sorted(indexed, key=operator.itemgetter(1), reverse=True)
    top_3 = []
    for i in range(n):
        top_3.append(sorted_list[i][0])

    return top_3


def show_topics(vectorizer, lda, n_words):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))

    return topic_keywords
