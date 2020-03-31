import pickle
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from tensorflow_core.python.keras.models import load_model
from src.pipelineapplication import utils

from src.lstm import lstm_utils

lstm_model = "../../output/official/lstm.h5"
tokenizer_model = "../../output/official/tokenizer.pickle"

text = utils.get_abstracts_to_analyze()
text = list(map(lambda x: x[0], text))

text[0] = lstm_utils.preprocess_text(text[0])

with open(tokenizer_model, 'rb') as handle:
    tokenizer = pickle.load(handle)

seq = tokenizer.texts_to_sequences(text)
seq = pad_sequences(seq, padding='post', maxlen=200)

# load model from single file
model = load_model(lstm_model)

# make predictions
yhat = model.predict(seq)
list_couple = []
for y in yhat:
    topic = np.argmax(y)
    prob = np.round(y[topic], 3)
    list_couple.append((topic, prob))

for c in list_couple:
    print(c)
