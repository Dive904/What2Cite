from keras_preprocessing.text import Tokenizer
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from nltk.util import ngrams

from src.lstm import lstm_utils

import json

file_pathname = "C:\\Users\\Davide\\Desktop\\word2vec-unigram-bigrams-\\" \
                "word2vec_bi_gram\\word2vec_bi_gram\\word2vec_bi_gram.syn0.npy"
json_pathname = "C:\\Users\\Davide\\Desktop\\word2vec-unigram-bigrams-\\word2vec_bi_gram\\" \
                "word2vec_bi_gram\\word2vec_bi_gram.vocab.json"

text = ["A new fine-grained parallel programming model - Thin Kernel model is brought forth. "
        "In this model, the partitions of the parallel tasks are separated from the computational "
        "kernel of the problem. The Thin Kernel model produces parallel tasks dynamically at runtime "
        "when the tasks are being scheduled which makes the task assignment method in the Thin Kernel "
        "model real and distributed. The Thin Kernel model uses dynamic class loading and on-demand code "
        "transport. The collaboration of these technologies paves the way for large-scale parallel computing "
        "over wide-area networks."]

new_text = ["The analysis of the error of stereo measurements by triangulation is revisited from three points of "
            "view: geometrical, statistical and visual quality. When the target is visible by a set of "
            "distributed cameras in the workspace, there are multiple combinations of camera pairs adequate "
            "to be considered for the location, by triangulation, of the target position. Three-camera placements are "
            "analysed evaluating their precision in a short-medium distance. The work presented analyses which "
            "combination of stereo measurements gives the best results, and proposes a method for the automatic "
            "selection of the most adequate cameras pair."]


with open(json_pathname) as json_file:
    json_data = json.load(json_file)

for i in range(len(text)):
    text[i] = lstm_utils.preprocess_text(text[i])

for i in range(len(new_text)):
    new_text[i] = lstm_utils.preprocess_text(new_text[i])

vectorizer = CountVectorizer(ngram_range=(1, 2))
vectorizer.fit_transform(text)

text_sequence = []
for sentence in text:
    tmp_sequence = []
    bigrams = list(ngrams(sentence.split(), 2))
    for bigram in bigrams:
        b = bigram[0] + " " + bigram[1]
        a = 0
        index_to_add = None
        if json_data.get(b) is not None:
            index_to_add = vectorizer.vocabulary_.get(b)
        elif json_data.get(bigram[0]) is not None:
            index_to_add = vectorizer.vocabulary_.get(bigram[0])
        if index_to_add is not None:
            tmp_sequence.append(index_to_add)
            a = 0
    text_sequence.append(tmp_sequence)

print(text_sequence)

text_sequence = []
for sentence in new_text:
    tmp_sequence = []
    bigrams = list(ngrams(sentence.split(), 2))
    for bigram in bigrams:
        b = bigram[0] + " " + bigram[1]
        index_to_add = -1
        if json_data.get(b) is not None:
            index_to_add = vectorizer.vocabulary_.get(b)
        else:
            index_to_add = vectorizer.vocabulary_.get(bigram[0])
        if index_to_add is not None:
            tmp_sequence.append(index_to_add)
    text_sequence.append(tmp_sequence)

print(text_sequence)
