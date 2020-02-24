from numpy import array
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
from numpy import array
from numpy import asarray
from numpy import zeros

from src.lstm import lstm_utils
import tensorflow

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

print("INFO: Reading csv", end="... ")
abstracts = pd.read_csv("../../output/lstmdataset/data_multilabel_reduced.csv")
print("Done ✓")

abstracts_labels = abstracts[[str(i) for i in range(55)]]

print("INFO: Preprocessing", end="... ")
X = []
sentences = list(abstracts["text"])
for sen in sentences:
    X.append(lstm_utils.preprocess_text(sen))
print("Done ✓")

y = abstracts_labels.values

print("INFO: Splitting into training and test", end="... ")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("Done ✓")


print("INFO: Fitting tokenizer", end="... ")
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
print("Done ✓")

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 200

print("INFO: Padding", end="... ")
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
print("Done ✓")

embeddings_dictionary = dict()

glove_file = open("../../input/glove.6B.100d.txt", encoding="utf8")

print("INFO: Embedding", end="... ")
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
print("Done ✓")

print("INFO: Creating model", end="... ")
deep_inputs = Input(shape=(maxlen,))
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(deep_inputs)
LSTM_Layer_1 = LSTM(128)(embedding_layer)
dense_layer_1 = Dense(55, activation='sigmoid')(LSTM_Layer_1)
model = Model(inputs=deep_inputs, outputs=dense_layer_1)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print("Done ✓")

print(model.summary())

print("INFO: Fitting model", end="... ")
history = model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=1, validation_split=0.2)
print("Done ✓")

print("INFO: Evaluating model", end="... ")
score = model.evaluate(X_test, y_test, verbose=1)
print("Done ✓")

print("Test Score:", score[0])
print("Test Accuracy:", score[1])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
