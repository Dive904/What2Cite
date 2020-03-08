# https://stackabuse.com/python-for-nlp-multi-label-text-classification-with-keras/ <-- tutorial

from numpy import array
from numpy import asarray
from numpy import zeros

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt


def preprocess_text(s):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', s)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


print("INFO: Extracting Training dataset", end="... ")
abstracts_training = pd.read_csv("../../output/lstmdataset/trainingdataset_multilabel.csv")
print("Done ✓", end="\n\n")

print("INFO: Preprocessing Training dataset", end="... ")
X_train = []
sentences = list(abstracts_training["text"])
for sen in sentences:
    X_train.append(preprocess_text(sen))
print("Done ✓", end="\n\n")

abstracts_train_labels = abstracts_training[[str(i) for i in range(40)]]

print("INFO: Extracting Test dataset", end="... ")
abstracts_test = pd.read_csv("../../output/lstmdataset/testdataset_multilabel.csv")
print("Done ✓", end="\n\n")

print("INFO: Preprocessing Test dataset", end="... ")
X_test = []
sentences = list(abstracts_test["text"])
for sen in sentences:
    X_test.append(preprocess_text(sen))
print("Done ✓", end="\n\n")

abstracts_test_labels = abstracts_test[[str(i) for i in range(40)]]

print("INFO: Extracting Validation dataset", end="... ")
abstracts_validation = pd.read_csv("../../output/lstmdataset/valdataset_multilabel.csv")
print("Done ✓", end="\n\n")

print("INFO: Preprocessing Validation dataset", end="... ")
X_val = []
sentences = list(abstracts_validation["text"])
for sen in sentences:
    X_val.append(preprocess_text(sen))
print("Done ✓", end="\n\n")

abstracts_val_labels = abstracts_validation[[str(i) for i in range(40)]]

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_val = tokenizer.texts_to_sequences(X_val)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 200

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)

embeddings_dictionary = dict()

glove_file = open('../../input/glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

deep_inputs = Input(shape=(maxlen,))
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(deep_inputs)
LSTM_Layer_1 = LSTM(128)(embedding_layer)
dense_layer_1 = Dense(6, activation='sigmoid')(LSTM_Layer_1)
model = Model(inputs=deep_inputs, outputs=dense_layer_1)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())
history = model.fit(X_train, abstracts_train_labels, batch_size=128, epochs=5, verbose=1,
                    validation_data=(X_val, abstracts_val_labels))
