from keras.preprocessing.sequence import pad_sequences
from tensorflow_core.python.keras.models import load_model

import pandas as pd
import pickle

from src.lstm import lstm_utils

maxlen = 200

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

with open('../../output/modeltrain/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

print("INFO: Tokenizing sequences", end="... ")
X_test = tokenizer.texts_to_sequences(X_test)
print("Done ✓")

print("INFO: Padding sequences", end="... ")
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
print("Done ✓")

print("*** 20/15 ***")
model = load_model("../../output/modeltrain/train20/lstm00000015.h5")
score = model.evaluate(X_test, abstracts_test_labels, verbose=1)
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])

print("*** 20/19 ***")
model = load_model("../../output/modeltrain/train20/lstm00000019.h5")
score = model.evaluate(X_test, abstracts_test_labels, verbose=1)
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])

print("*** 02 ***")
model = load_model("../../output/modeltrain/lstm00000002.h5")
score = model.evaluate(X_test, abstracts_test_labels, verbose=1)
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])
