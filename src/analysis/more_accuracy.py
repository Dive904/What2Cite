import pickle
import gc

import pandas as pd
from keras_preprocessing.sequence import pad_sequences

from tensorflow_core.python.keras.models import load_model

from src.lstm import lstm_utils
from src.pipelineapplication import utils

lstm_model = "../../output/official/lstm.h5"
tokenizer_model = "../../output/official/tokenizer.pickle"
testset_path = "../../output/lstmdataset/testdataset_multilabel.csv"
validationset_path = "../../output/lstmdataset/trainingdataset_multilabel.csv"
trainingset_path = "../../output/lstmdataset/valdataset_multilabel.csv"
padding = ",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1"
T = 0.4
N = 3

print("*** TEST SET ***")
print("INFO: Extracting Test dataset", end="... ")
abstracts_test = pd.read_csv(testset_path)
print("Done ✓")

print("INFO: Preprocessing Test dataset", end="... ")
X_test = []
sentences = list(abstracts_test["text"])

with open(testset_path, "r", encoding="utf-8") as fp:
    lines = fp.readlines()
    lines = lines[1:]
    lines = list(map(lambda x: x[-len(padding):-1], lines))
    lines = list(map(lambda x: x.split(","), lines))
    lines = list(map(lambda x: list(map(lambda y: int(y), x)), lines))
    labels = lines

for sen in sentences:
    X_test.append(lstm_utils.preprocess_text(sen))
print("Done ✓", end="\n\n")

model = load_model(lstm_model)  # load model from single file

with open(tokenizer_model, 'rb') as handle:  # get the tokenizer
    tokenizer = pickle.load(handle)

labels = list(map(lambda x: x.index(1), labels))

print("INFO: Tokenizing sequences", end="... ")
seq = tokenizer.texts_to_sequences(X_test)
seq = pad_sequences(seq, padding='post', maxlen=200)
print("Done ✓")

print("INFO: Predicting", end="... ")
yhat = model.predict(seq)  # make predictions
print("Done ✓")

predictions = []
for y in yhat:
    pr = list(enumerate(y))
    pr.sort(key=lambda x: x[1], reverse=True)
    predictions.append(pr[:N])

cont = 0
for i in range(len(predictions)):
    p = predictions[i]
    predicted_class = [x[0] for x in p]
    if labels[i] in predicted_class:
        cont += 1

acc_1 = cont / len(predictions)

predictions = list(map(lambda x: utils.get_valid_predictions(x, T), yhat))

cont = 0
for i in range(len(predictions)):
    p = predictions[i]
    predicted_class = [x[0] for x in p]
    if labels[i] in predicted_class:
        cont += 1

acc_2 = cont / len(predictions)

print("Accuracy with first " + str(N) + " elements: " + str(acc_1))
print("Accuracy with threshold " + str(acc_2))
print()

abstracts_test = X_test = None
gc.collect()

print("*** TRAINING SET ***")
print("INFO: Extracting Training dataset", end="... ")
abstracts_training = pd.read_csv(trainingset_path)
print("Done ✓")

print("INFO: Preprocessing Training dataset", end="... ")
X_training = []
sentences = list(abstracts_training["text"])

with open(trainingset_path, "r", encoding="utf-8") as fp:
    lines = fp.readlines()
    lines = lines[1:]
    lines = list(map(lambda x: x[-len(padding):-1], lines))
    lines = list(map(lambda x: x.split(","), lines))
    lines = list(map(lambda x: list(map(lambda y: int(y), x)), lines))
    labels = lines

for sen in sentences:
    X_training.append(lstm_utils.preprocess_text(sen))
print("Done ✓", end="\n\n")

model = load_model(lstm_model)  # load model from single file

with open(tokenizer_model, 'rb') as handle:  # get the tokenizer
    tokenizer = pickle.load(handle)

labels = list(map(lambda x: x.index(1), labels))

print("INFO: Tokenizing sequences", end="... ")
seq = tokenizer.texts_to_sequences(X_training)
seq = pad_sequences(seq, padding='post', maxlen=200)
print("Done ✓")

print("INFO: Predicting", end="... ")
yhat = model.predict(seq)  # make predictions
print("Done ✓")

predictions = []
for y in yhat:
    pr = list(enumerate(y))
    pr.sort(key=lambda x: x[1], reverse=True)
    predictions.append(pr[:N])

cont = 0
for i in range(len(predictions)):
    p = predictions[i]
    predicted_class = [x[0] for x in p]
    if labels[i] in predicted_class:
        cont += 1

acc_1 = cont / len(predictions)

predictions = list(map(lambda x: utils.get_valid_predictions(x, T), yhat))

cont = 0
for i in range(len(predictions)):
    p = predictions[i]
    predicted_class = [x[0] for x in p]
    if labels[i] in predicted_class:
        cont += 1

acc_2 = cont / len(predictions)

print("Accuracy with first " + str(N) + " elements: " + str(acc_1))
print("Accuracy with threshold " + str(acc_2))
print()

abstracts_training = X_training = None
gc.collect()

print("*** VALIDATION SET ***")
print("INFO: Extracting Validation dataset", end="... ")
abstracts_validation = pd.read_csv(validationset_path)
print("Done ✓")

print("INFO: Preprocessing Validation dataset", end="... ")
X_validation = []
sentences = list(abstracts_validation["text"])

with open(validationset_path, "r", encoding="utf-8") as fp:
    lines = fp.readlines()
    lines = lines[1:]
    lines = list(map(lambda x: x[-len(padding):-1], lines))
    lines = list(map(lambda x: x.split(","), lines))
    lines = list(map(lambda x: list(map(lambda y: int(y), x)), lines))
    labels = lines

for sen in sentences:
    X_validation.append(lstm_utils.preprocess_text(sen))
print("Done ✓", end="\n\n")

model = load_model(lstm_model)  # load model from single file

with open(tokenizer_model, 'rb') as handle:  # get the tokenizer
    tokenizer = pickle.load(handle)

labels = list(map(lambda x: x.index(1), labels))

print("INFO: Tokenizing sequences", end="... ")
seq = tokenizer.texts_to_sequences(X_validation)
seq = pad_sequences(seq, padding='post', maxlen=200)
print("Done ✓")

print("INFO: Predicting", end="... ")
yhat = model.predict(seq)  # make predictions
print("Done ✓")

predictions = []
for y in yhat:
    pr = list(enumerate(y))
    pr.sort(key=lambda x: x[1], reverse=True)
    predictions.append(pr[:N])

cont = 0
for i in range(len(predictions)):
    p = predictions[i]
    predicted_class = [x[0] for x in p]
    if labels[i] in predicted_class:
        cont += 1

acc_1 = cont / len(predictions)

predictions = list(map(lambda x: utils.get_valid_predictions(x, T), yhat))

cont = 0
for i in range(len(predictions)):
    p = predictions[i]
    predicted_class = [x[0] for x in p]
    if labels[i] in predicted_class:
        cont += 1

acc_2 = cont / len(predictions)

print("Accuracy with first " + str(N) + " elements: " + str(acc_1))
print("Accuracy with threshold " + str(acc_2))
