# https://stackabuse.com/python-for-nlp-multi-label-text-classification-with-keras/ <-- tutorial


from numpy import asarray
from numpy import zeros

import keras.layers as kl

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer

import pandas as pd
import pickle

from src.lstm import lstm_utils


print("INFO: Extracting Training dataset", end="... ")
abstracts_training = pd.read_csv("../../output/lstmdataset/trainingdataset_multilabel.csv")
print("Done ✓")

print("INFO: Preprocessing Training dataset", end="... ")
X_train = []
sentences = list(abstracts_training["text"])
for sen in sentences:
    X_train.append(lstm_utils.preprocess_text(sen))
print("Done ✓", end="\n\n")

abstracts_train_labels = abstracts_training[[str(i) for i in range(40)]]

print("INFO: Extracting Test dataset", end="... ")
abstracts_test = pd.read_csv("../../output/lstmdataset/testdataset_multilabel.csv")
print("Done ✓")

print("INFO: Preprocessing Test dataset", end="... ")
X_test = []
sentences = list(abstracts_test["text"])
for sen in sentences:
    X_test.append(lstm_utils.preprocess_text(sen))
print("Done ✓", end="\n\n")

abstracts_test_labels = abstracts_test[[str(i) for i in range(40)]]

print("INFO: Extracting Validation dataset", end="... ")
abstracts_validation = pd.read_csv("../../output/lstmdataset/valdataset_multilabel.csv")
print("Done ✓")

print("INFO: Preprocessing Validation dataset", end="... ")
X_val = []
sentences = list(abstracts_validation["text"])
for sen in sentences:
    X_val.append(lstm_utils.preprocess_text(sen))
print("Done ✓", end="\n\n")

abstracts_val_labels = abstracts_validation[[str(i) for i in range(40)]]

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

print("INFO: Tokenizing sequences", end="... ")
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_val = tokenizer.texts_to_sequences(X_val)
print("Done ✓")

vocab_size = len(tokenizer.word_index) + 1

maxlen = 200

print("INFO: Padding sequences", end="... ")
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)
print("Done ✓")

embeddings_dictionary = dict()

glove_file = open('../../input/glove.6B.100d.txt', encoding="utf8")

print("INFO: Embedding words", end="... ")
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

model = Sequential()
model.add(kl.Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False))
model.add(kl.LSTM(400))
model.add(kl.Dense(40, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())
history = model.fit(X_train, abstracts_train_labels, batch_size=128, epochs=40, verbose=1,
                    validation_data=(X_val, abstracts_val_labels))

"""
model_json = model.to_json()
with open("../../output/models/new_lstm.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("../../output/models/new_lstm_weights.h5")
"""

model.save("../../output/models/lstm.h5")
with open("../../output/models/tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

score = model.evaluate(X_test, abstracts_test_labels, verbose=1)

print("Test Loss:", score[0])
print("Test Accuracy:", score[1])
