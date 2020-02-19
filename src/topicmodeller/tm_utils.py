from nltk.stem import WordNetLemmatizer
import os
import io

from src.fileutils import file_abstract


# change dataset if you want to reduce LDA input data
def extract_abstract(dir_path, abstract_batch):
    files = os.listdir(dir_path)
    ris = []
    for file in files[0:abstract_batch]:  # change file number here
        ris += file_abstract.txt_only_abstract_reader(dir_path + file)

    return ris


def preprocess_abstract(abstracts):
    ris = []
    lem = WordNetLemmatizer()
    for abstract in abstracts:
        tmp = []
        a = abstract.split()
        for word in a:
            w = word.lower()  # Convert to lowercase
            w = lem.lemmatize(w)  # Lemmatize word
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

