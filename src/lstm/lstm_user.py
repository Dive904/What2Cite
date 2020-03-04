from keras.engine.saving import model_from_json
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

import numpy as np

from src.datasetcreator import utils

json_file = open('../../output/models/lstm.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("../../output/models/lstm_weights.h5")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

text = ["This article looks at some of the challenges related to the deployment of the Internet Of Things (IoT), "
        "specifically to ascertain that security becomes an integral part of the technology rather than a bolted-on "
        "wrapper of limited efficacy. IoT security (IoTSec) is needed at all ‘layers’ of the IoT environment and may "
        "be specific to the IoT ‘layer’ in question."]

text[0] = utils.preprocess_text(text[0])

tokenizer = Tokenizer(num_words=5000)
seq = tokenizer.texts_to_sequences(text)
maxlen = 200
padded = pad_sequences(seq, maxlen=maxlen)
pred = loaded_model.predict(padded)
labels = [str(i) for i in range(55)]
print(pred, labels[np.argmax(pred)])
