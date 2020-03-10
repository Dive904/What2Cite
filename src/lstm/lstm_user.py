from keras.engine.saving import model_from_json
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

import numpy as np
from tensorflow_core.python.keras.models import load_model

from src.lstm import lstm_utils

json_file = open('../../output/models/lstm.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("../../output/models/lstm_weights.h5")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

text = ['Overlap of footprints of light emitting diodes (LEDs) increases the positioning accuracy of wearable '
        'LED indoor positioning systems (IPS) but such an approach assumes that the footprint boundaries are '
        'defined. In this work, we develop a mathematical model for defining the footprint boundaries of an '
        'LED in terms of a threshold angle instead of the conventional half or full angle. To show the effect '
        'of the threshold angle, we compare how overlaps and receiver tilts affect the performance of an '
        'LED-based IPS when the optical boundary is defined at the threshold angle and at the full angle. '
        'Using experimental measurements, simulations, and theoretical analysis, the effect of the defined '
        'threshold angle is estimated. The results show that the positional time when using the newly defined '
        'threshold angle is 12 times shorter than the time when the full angle is used. When the effect of tilt '
        'is considered, the threshold angle time is 22 times shorter than the full angle positioning time. '
        'Regarding accuracy, it is shown in this work that a positioning error as low as 230 mm can be '
        'obtained. Consequently, while the IPS gives a very low positioning error, a defined threshold angle '
        'reduces delays in an overlap-based LED IPS.']

text[0] = lstm_utils.preprocess_text(text[0])

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(text)
seq = tokenizer.texts_to_sequences(text)
maxlen = 200
padded = pad_sequences(seq, maxlen=maxlen)

# load model from single file
model = load_model("../../output/models/new_lstm_final_model.h5")
# make predictions
yhat = model.predict(padded)
print(yhat[0])
print(np.argmax(yhat[0]))
