import pickle
from keras.preprocessing.sequence import pad_sequences
from tensorflow_core.python.keras.models import load_model

from src.lstm import lstm_utils

text = ["With the spread of semantic technologies more and more companies manage their own knowledge graphs (KG), "
        "applying them, among other tasks, to text analysis. However, the proprietary KGs are by design "
        "domain specific and do not include all the different possible meanings of the words used in a corpus. "
        "In order to enable the usage of these KGs for automatic text annotations, we introduce a robust method "
        "for discriminating word senses using sense indicators found in the KG: types, synonyms and/or hypernyms. "
        "The method uses collocations to induce word senses and to discriminate the sense included in the KG "
        "from the other senses, without the need for information about the latter, or the need for manual effort. "
        "On the two datasets created specially for this task the method outperforms the baseline and shows "
        "accuracy above 80%."]

text[0] = lstm_utils.preprocess_text(text[0])

with open('../../output/official/tokenizer_73acc_glove840.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

seq = tokenizer.texts_to_sequences(text)
seq = pad_sequences(seq, padding='post', maxlen=200)

# load model from single file
model = load_model("../../output/official/lstm_73acc_glove840.h5")
# make predictions
yhat = model.predict_classes(seq)
print(yhat)
