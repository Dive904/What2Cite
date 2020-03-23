import keras

from keras.preprocessing.sequence import pad_sequences
from tensorflow_core.python.keras.models import load_model

import matplotlib.pyplot as plt
import pandas as pd
import pickle

from src.lstm import lstm_utils

n_epoch = 6
bacth_size = 128
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

with open('../../output/modeltrain/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

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

model = load_model("../../output/modeltrain/train20/lstmfinal.h5")

checkpoint = keras.callbacks.ModelCheckpoint('../../output/modeltrain/lstm{epoch:08d}.h5', period=1)

history = model.fit(X_train, abstracts_train_labels, batch_size=bacth_size, epochs=n_epoch, verbose=1,
                    validation_data=(X_val, abstracts_val_labels),
                    callbacks=[checkpoint])

score = model.evaluate(X_test, abstracts_test_labels, verbose=1)

print("Test Loss:", score[0])
print("Test Accuracy:", score[1])

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
