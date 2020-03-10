import pickle
from tensorflow_core.python.keras.models import load_model

from src.lstm import lstm_utils

text = ["Inspection of defects in civil infrastructure has been a constant field of research. In the majority of inspections, a technician is responsible to go physically to the field in order to detect and measure defects. Through the measurement results, engineers are able to perform the Structural Health Monitoring (SHM) of a measured structure. In this paper, a fully architecture of an autonomous system is proposed with the goal to automate the SHM task. The proposed system uses an autonomous robot, database and the proposed architecture to integrate all sub-systems for the automation of the SHM. Experimental results validate the technical feasibility of the proposed system."]

text[0] = lstm_utils.preprocess_text(text[0])

with open('../../output/models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

seq = tokenizer.texts_to_sequences(text)

# load model from single file
model = load_model("../../output/models/lstm.h5")
# make predictions
yhat = model.predict(seq)
print(yhat)
