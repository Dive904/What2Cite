# https://towardsdatascience.com/multi-class-text-classification-with-lstm-using-tensorflow-2-0-d88627c10a35
# https://machinelearningmastery.com/diagnose-overfitting-underfitting-lstm-models/ <-- diagnose Over/Under fitting

import csv
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

from src.lstmdatasetcreator import utils

additional_stopwords = ["paper", "method", "large", "model", "proposed", "study", "based", "using", "approach", "also"]
STOPWORDS = set(stopwords.words('english')).union(set(additional_stopwords))

vocab_size = 6000
embedding_dim = 64
max_length = 1000
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8

training_articles = []
training_labels = []

print("INFO: Extracting Training dataset", end="... ")
with open("../../output/lstmdataset/trainingdataset.csv", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        training_labels.append(row[1])
        article = row[0].lower()
        for word in STOPWORDS:
            token = ' ' + word + ' '
            article = article.replace(token, ' ')
            article = article.replace(' ', ' ')
        training_articles.append(article)
print("Done ✓", end="\n\n")

print(len(training_labels))
print(len(training_articles))

validation_articles = []
validation_labels = []

print("INFO: Extracting Validation dataset", end="... ")
with open("../../output/lstmdataset/valdataset.csv", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        validation_labels.append(row[1])
        article = row[0].lower()
        for word in STOPWORDS:
            token = ' ' + word + ' '
            article = article.replace(token, ' ')
            article = article.replace(' ', ' ')
        validation_articles.append(article)
print("Done ✓", end="\n\n")

print(len(training_articles))
print(len(training_labels))
print(len(validation_articles))
print(len(validation_labels))

print("INFO: Fitting Tokenizer", end="... ")
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_articles)
word_index = tokenizer.word_index
print(dict(list(word_index.items())[0:10]))
print("Done ✓", end="\n\n")

train_sequences = tokenizer.texts_to_sequences(training_articles)
print(train_sequences[10])

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(len(train_sequences[0]))
print(len(train_padded[0]))

print(len(train_sequences[1]))
print(len(train_padded[1]))

print(len(train_sequences[10]))
print(len(train_padded[10]))

validation_sequences = tokenizer.texts_to_sequences(validation_articles)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(len(validation_sequences))
print(validation_padded.shape)

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(training_labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(training_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))
print(training_label_seq[0])
print(training_label_seq[1])
print(training_label_seq[2])
print(training_label_seq.shape)

print(validation_label_seq[0])
print(validation_label_seq[1])
print(validation_label_seq[2])
print(validation_label_seq.shape)

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

print(utils.decode_article(reverse_word_index, train_padded[10]))
print('---')
print(training_articles[10])

model = tf.keras.Sequential([
    # Add an Embedding layer expecting input vocab of size 5000,
    # and output embedding dimension of size 64 we set at the top
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    # use ReLU in place of tanh function since they are very good alternatives of each other.
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    # Add a Dense layer with 6 units and softmax activation.
    # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
    tf.keras.layers.Dense(41, activation='softmax')
])
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

num_epochs = 12
history = model.fit(train_padded, training_label_seq,
                    epochs=num_epochs,
                    validation_data=(validation_padded, validation_label_seq),
                    verbose=2)

model_json = model.to_json()
with open("../../output/models/new_lstm.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("../../output/models/new_lstm_weights.h5")

utils.plot_graphs(history, "accuracy")
utils.plot_graphs(history, "loss")
