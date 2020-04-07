from nltk.stem import WordNetLemmatizer
import os
import io
import re
import operator
import numpy as np

from src.fileutils import file_abstract


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


def extract_paper_info(dir_path, start=None, end=None, exception=None):
    files = os.listdir(dir_path)
    ris = []

    if start is None:
        start = 0
    if end is None:
        end = len(files)

    for file in files[start:end]:
        abstracts = file_abstract.txt_dataset_reader(dir_path + file)
        if exception is None:
            ris += abstracts
        else:
            for a in abstracts:
                if a["id"] not in exception:
                    ris.append(a)

    return ris


def preprocess_abstract(abstract):
    additional_stopwords = ["paper", "method", "large", "model", "proposed", "study",
                            "based", "using", "approach", "data", "result", "ha", "wa"]
    lem = WordNetLemmatizer()

    abstract = re.sub(r'[^\w\s]', '', abstract)  # Remove punctuation
    a = abstract.split()
    ris = []
    for word in a:
        w = word.lower()  # Convert to lowercase
        w = lem.lemmatize(w)  # Lemmatize word
        if w not in additional_stopwords:
            ris.append(w)
    ris = " ".join(ris)

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


def find_n_maximum(items, n, skip=False):
    """
    This function is used to find the n max elem in a list
    :param items: list of items
    :param n: maximum number
    :param skip: if True, return every elem in the n maximum, not only the first n elem of a list
    :return: two list, the first containing the index of the elem and the second the list of the elem
    """
    if not skip:
        indexed = list(enumerate(items))
        sorted_list = sorted(indexed, key=operator.itemgetter(1), reverse=True)
        top_n_index = []
        top_n_elem = []
        for i in range(n):
            top_n_index.append(sorted_list[i][0])
            top_n_elem.append(sorted_list[i][1])
    else:
        without_duplicates = list(dict.fromkeys(items))
        top_index, top_elem = find_n_maximum(without_duplicates, n)
        top_n_index = []
        top_n_elem = []
        for i in range(len(items)):
            if items[i] in top_elem:
                top_n_elem.append(items[i])
                top_n_index.append(i)

    return top_n_index, top_n_elem


def get_topic_keywords(vectorizer, lda, n_words):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))

    return topic_keywords


def convert_df_to_list(df, number_words):
    final_list = []
    placeholder = "(Word;Perc)_"
    for index, row in df.iterrows():
        tmp_list = []
        for i in range(number_words):
            first = row[placeholder + str(i)].split(", ")[0]
            second = row[placeholder + str(i)].split(", ")[1]
            first = str(first[2:-1])
            second = float(second[:-1])
            tmp_list.append((first, second))
        final_list.append(tmp_list)

    return final_list


def get_total_energy(input_list):
    ris = 0
    for x in input_list:
        ris += x[1]

    return ris
