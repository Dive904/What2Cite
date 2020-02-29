import csv
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

additional_stopwords = ["paper", "method", "large", "model", "proposed", "study", "based", "using", "approach", "also"]
STOPWORDS = set(stopwords.words('english')).union(set(additional_stopwords))

print(tf.__version__)
vocab_size = 5000
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8

articles = []
labels = []

with open("../../output/lstmdataset/data_reduced.csv", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        labels.append(row[1])
        article = row[0].lower()
        for word in STOPWORDS:
            token = ' ' + word + ' '
            article = article.replace(token, ' ')
            article = article.replace(' ', ' ')
        articles.append(article)

print(len(labels))
print(len(articles))

train_size = int(len(articles) * training_portion)

train_articles = articles[0: train_size]
train_labels = labels[0: train_size]

validation_articles = articles[train_size:]
validation_labels = labels[train_size:]

print(train_size)
print(len(train_articles))
print(len(train_labels))
print(len(validation_articles))
print(len(validation_labels))

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_articles)
word_index = tokenizer.word_index
print(dict(list(word_index.items())[0:10]))

train_sequences = tokenizer.texts_to_sequences(train_articles)
print(train_sequences[10])
