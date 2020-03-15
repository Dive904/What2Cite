# https://stackabuse.com/python-for-nlp-multi-label-text-classification-with-keras/ <-- tutorial
from keras import regularizers
from numpy import asarray
from numpy import zeros

import keras.layers as kl

from keras.models import Sequential

import matplotlib.pyplot as plt
import pandas as pd
import pickle
import json

from sklearn.feature_extraction.text import CountVectorizer

from src.lstm import lstm_utils

file_pathname = "C:\\Users\\Davide\\Desktop\\word2vec-unigram-bigrams-\\" \
                "word2vec_bi_gram\\word2vec_bi_gram\\word2vec_bi_gram.syn0.npy"
json_pathname = "C:\\Users\\Davide\\Desktop\\word2vec-unigram-bigrams-\\word2vec_bi_gram\\" \
                "word2vec_bi_gram\\word2vec_bi_gram.vocab.json"

with open(json_pathname) as json_file:
    json_data = json.load(json_file)

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

vectorizer = CountVectorizer(ngram_range=(1, 2))
vectorizer.fit_transform(X_train + X_val)

print("INFO: Tokenizing sequences", end="... ")
X_train = lstm_utils.texts_to_sequence(vectorizer, X_train, json_data)
X_test = lstm_utils.texts_to_sequence(vectorizer, X_test, json_data)
X_val = lstm_utils.texts_to_sequence(vectorizer, X_val, json_data)
print("Done ✓")

vocab_size = len(vectorizer.vocabulary_.keys()) + 1

"""
maxlen = 200

print("INFO: Padding sequences", end="... ")
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)
print("Done ✓")
"""

embeddings_dictionary = dict()

embedding_col_number = 300
embedding_file = open('../../input/glove.6B.300d.txt', encoding="utf8")

print("INFO: Embedding words", end="... ")
for line in embedding_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
embedding_file.close()

embedding_matrix = zeros((vocab_size, embedding_col_number))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
print("\nDone ✓")

# 63% acc
model = Sequential()
model.add(kl.Embedding(vocab_size, embedding_col_number, weights=[embedding_matrix], trainable=False))
model.add(kl.Dropout(0.5))
# model.add(kl.Conv1D(filters=500, kernel_size=10, strides=3, padding="same", activation="sigmoid"))
# model.add(kl.MaxPool1D(pool_size=10, padding="same"))
model.add(kl.Bidirectional(kl.LSTM(500, activation='tanh')))
# model.add(kl.LSTM(500, activation='tanh'))
model.add(kl.Dropout(0.4))
model.add(kl.Dense(40, activation='softmax',
                   activity_regularizer=regularizers.l1(0.01),
                   kernel_regularizer=regularizers.l2(0.01)))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())
history = model.fit(X_train, abstracts_train_labels, batch_size=128, epochs=15, verbose=1,
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

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
