import csv
import re


from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


def extract_topic_from_dataset(topic_dataset):
    return topic_dataset.split(" - ")[0]


def extract_dict_dataset(filepath):
    res = []
    with open(filepath, mode="r", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            tmp = {"text": row["text"], "label": row["label"]}
            res.append(tmp)

    return res


def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    tmp = []
    lem = WordNetLemmatizer()
    text = text.split()
    for word in text:
        w = word.lower()  # Convert to lowercase
        w = lem.lemmatize(w)  # Lemmatize word
        tmp.append(w)
    text_preprocessed = " ".join(tmp)

    return text_preprocessed


def decode_article(reverse_word_index, text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()
