# https://stackabuse.com/python-for-nlp-multi-label-text-classification-with-keras/ <-- tutorial
from keras import regularizers
from numpy import asarray
from numpy import zeros
import keras

import keras.layers as kl

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer

import matplotlib.pyplot as plt
import pandas as pd
import pickle

from src.lstm import lstm_utils

n_epoch = 20
bacth_size = 128
n_words_tokenizer = 35000
maxlen = 200

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

tokenizer = Tokenizer(num_words=n_words_tokenizer)
tokenizer.fit_on_texts(X_train + X_val)

print("INFO: Tokenizing sequences", end="... ")
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_val = tokenizer.texts_to_sequences(X_val)
print("Done ✓")

vocab_size = len(tokenizer.word_index) + 1

print("INFO: Padding sequences", end="... ")
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)
print("Done ✓")

embeddings_dictionary = dict()

embedding_col_number = 300
embedding_file = open('../../input/glove.840B.300d.txt', encoding="utf8")

print("INFO: Embedding words", end="...")
for line in embedding_file:
    records = line.split()
    word = records[0]
    try:
        vector_dimensions = asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    except ValueError:
        print(" - Value Error", end="")
embedding_file.close()

embedding_matrix = zeros((vocab_size, embedding_col_number))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
print(" Done ✓")

# 73% acc
model = Sequential()
model.add(kl.Embedding(vocab_size, embedding_col_number, weights=[embedding_matrix], trainable=False))
model.add(kl.Dropout(0.5))
model.add(kl.Bidirectional(kl.LSTM(550, activation='tanh')))
model.add(kl.Dropout(0.4))
model.add(kl.Dense(40, activation='softmax',
                   activity_regularizer=regularizers.l1(0.01),
                   kernel_regularizer=regularizers.l2(0.01)))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

checkpoint = keras.callbacks.ModelCheckpoint('../../output/modeltrain/lstm{epoch:08d}.h5', period=1)

history = model.fit(X_train, abstracts_train_labels, batch_size=bacth_size, epochs=n_epoch, verbose=1,
                    validation_data=(X_val, abstracts_val_labels),
                    callbacks=[checkpoint])

score = model.evaluate(X_test, abstracts_test_labels, verbose=1)

print("Test Loss:", score[0])
print("Test Accuracy:", score[1])

with open("../../output/modeltrain/tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

model.save("../../output/modeltrain/lstmfinal.h5")

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
