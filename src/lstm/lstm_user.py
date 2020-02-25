from keras.engine.saving import model_from_json
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

import numpy as np

from src.lstm import lstm_utils

json_file = open('../../output/models/lstm.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("../../output/models/lstm_weights.h5")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

text = ["The Joint Video Exploration Team (JVET) recently launched the standardization of next-generation "
        "video coding named Versatile Video Coding (VVC) in which the Adaptive Multiple Transforms (AMT) "
        "is adopted as the primary residual coding transform solution. AMT introduces multiple transforms "
        "selected from the DST/DCT families and achieves noticeable coding gains. However, the set of transforms "
        "are calculated using direct matrix multiplication which induces higher run-time complexity and limits the "
        "application for practical video codec. In this paper, a fast DST-VII/DCT-VIII algorithm based on partial "
        "butterfly with dual implementation support is proposed, which aims at achieving reduced operation counts and "
        "run-time cost meanwhile yield almost the same coding performance. The proposed method has been implemented "
        "on top of the VTM-1.1 and experiments have been conducted using Common Test Conditions (CTC) to "
        "validate the efficacy. The experimental results show that the proposed methods, in the state-of-the-art "
        "codec, can provide an average of 7%, 5% and 8% overall decoding time savings under All Intra (AI), "
        "Random Access (RA) and Low Delay B (LDB) configuration, respectively yet still maintains "
        "coding performance."]

text[0] = lstm_utils.preprocess_text(text[0])

tokenizer = Tokenizer(num_words=5000)
seq = tokenizer.texts_to_sequences(text)
maxlen = 200
padded = pad_sequences(seq, maxlen=maxlen)
pred = loaded_model.predict(padded)
labels = [str(i) for i in range(55)]
print(pred, labels[np.argmax(pred)])
