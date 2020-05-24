import pickle

import pandas as pd
from keras_preprocessing.sequence import pad_sequences

from tensorflow_core.python.keras.models import load_model

from src.lstm import lstm_utils
from src.pipelineapplication import utils

lstm_model = "../../output/official/lstm.h5"
tokenizer_model = "../../output/official/tokenizer.pickle"
testset_path = "../../output/lstmdataset/testdataset_multilabel.csv"
padding = ",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1"
T = 0.4

print("INFO: Extracting Test dataset", end="... ")
abstracts_test = pd.read_csv(testset_path)
print("Done ✓")

print("INFO: Preprocessing Test dataset", end="... ")
X_test = []
sentences = list(abstracts_test["text"])
abstracts_test_labels = abstracts_test[[str(i) for i in range(40)]]

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

X_test = X_test[:100]
abstracts_test_labels = abstracts_test_labels[:100]

print("INFO: Tokenizing sequences", end="... ")
seq = tokenizer.texts_to_sequences(X_test)
seq = pad_sequences(seq, padding='post', maxlen=200)
print("Done ✓")

print("INFO: Predicting", end="... ")
yhat = model.predict(seq)  # make predictions
predictions = list(map(lambda x: utils.get_valid_predictions(x, T), yhat))
print(len(predictions))
for p in predictions:
    print(p)
print("Done ✓")
