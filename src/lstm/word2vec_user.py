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

texts = ["A new fine-grained parallel programming model - Thin Kernel model is brought forth "
         "In this model, the partitions of the parallel tasks are separated from the computational "
         "kernel of the problem. The Thin Kernel model produces parallel tasks dynamically at runtime "
         "when the tasks are being scheduled which makes the task assignment method in the Thin Kernel "
         "transport. The collaboration of these technologies paves the way for large-scale parallel computing "
         "over wide-area networks."]

with open(json_pathname) as json_file:
    json_data = json.load(json_file)

for i in range(len(texts)):
    texts[i] = lstm_utils.preprocess_text(texts[i])

texts_to_fit = []
a = 0
for t in texts:
    bigrams = list(ngrams(t.split(), 2))
    for b in bigrams:
        bi = b[0] + " " + b[1]
        if json_data.get(bi) is not None:
            texts_to_fit.append(bi)

print(texts_to_fit)

vectorizer = CountVectorizer(ngram_range=(1, 2))
vectorizer.fit_transform(texts_to_fit)

print(vectorizer.vocabulary_)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)

X_train = tokenizer.texts_to_sequences(texts)
print(X_train)

X_train = lstm_utils.texts_to_sequence(vectorizer, texts, json_data)
print(X_train)
