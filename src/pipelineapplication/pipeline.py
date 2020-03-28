# This script is used to implement the pipeline for only specific abstracts given in input in a list

import pickle
import numpy as np

from keras_preprocessing.sequence import pad_sequences
from tensorflow_core.python.keras.models import load_model

from src.fileutils import file_abstract
from src.pipelineapplication import utils
from src.lstm import lstm_utils

missig_citation_path = "../../output/official/missing_citations.txt"
cit_labelled_analyzed_pickle_path = "../../output/official/cit_labelled_with_final_topic_pickle.pickle"
lstm_model = "../../output/official/lstm.h5"
tokenizer_model = "../../output/official/tokenizer.pickle"
cit_labelled_path = "../../output/official/topics_cits_labelled_pickle.pickle"
P = 1

with open(cit_labelled_path, 'rb') as handle:  # take the list of CitTopic score
    # in this part, we read the cit topics labelled with other information.
    # We don't need that. So, we'll do a map removing all noise, getting only the paper id for all CitTopic
    cit_topics = pickle.load(handle)
    cit_topics = list(map(lambda x: list(map(lambda y: y[0], x)), cit_topics))

dataset = file_abstract.txt_lstm_dataset_reader("../../output/official/final.txt")

with open(cit_labelled_analyzed_pickle_path, 'rb') as handle:  # take the list of CitTopic score
    cit_topic_labelled = pickle.load(handle)

abstracts = utils.get_abstracts_to_analyze()  # get the abstract to analyze

abstracts_prep = list(map(lambda x: lstm_utils.preprocess_text(x[0]), abstracts))  # prepare text for classification

with open(tokenizer_model, 'rb') as handle:  # get the tokenizer
    tokenizer = pickle.load(handle)

seq = tokenizer.texts_to_sequences(abstracts_prep)
seq = pad_sequences(seq, padding='post', maxlen=200)

model = load_model(lstm_model)  # load model from single file
yhat = model.predict(seq)  # make predictions
predicted_topic = []
for y in yhat:
    # at the end of the loop, predicted_topic will contain a list of couples where the first element is the predicted
    # topic and the second element is the probability of the predicted topic
    p = np.argmax(y)
    predicted_topic.append((p, np.round(y[p], 3)))

# normalizing the CitTopic scores
cit_topic_normalized = np.round(list(map(lambda x: utils.normalize_scores_on_cittopics(x, P), cit_topic_labelled)), 3)

reference_cit_topics = []
for p in predicted_topic:
    # at the end of the loop, reference_cit_topics will contain a list of couple, where the first element is the
    # classified topic for the paper, and the second element is reference CitTopic that classified paper
    topic = p[0]  # get the topic
    classification_score = p[1]
    score_on_predicted_topic = []
    for c in cit_topic_normalized:
        topic_score = c[topic]  # get the score for that specific topic
        k = topic_score * classification_score  # do the multiply
        score_on_predicted_topic.append(k)  # append this to the list
    reference_cit_topics.append((topic, int(np.argmax(score_on_predicted_topic))))  # get the reference CitTopic

data_out_citations = []
for data in dataset:  # looking for all the out citations of the abstract
    # at the end of this loop, the data_out_citations will contain a list of records where the first element
    # is the paper id of the classified abstract, the second element is the paper title and the third element
    # is the list of out citations of that paper
    for a in abstracts:
        if data["id"] == a[1]:
            data_out_citations.append((a[1], data["title"], data["outCitations"]))

# now, data_out_citations and reference_cit_topics have the same len, this because an element on the i-th position of
# data_out_citations are the data of an input analyzed paper and in reference_cit_topics on position i, there is
# the reference CitTopic of that analyzed input paper

with open(missig_citation_path, "w") as out_file:
    for i in range(len(data_out_citations)):
        paper_id = data_out_citations[i][0]
        paper_title = data_out_citations[i][1]
        paper_out_citations = data_out_citations[i][2]
        classified_topic = reference_cit_topics[i][0]
        reference_cit_topic_index = reference_cit_topics[i][1]
        reference_cit_topic_list = cit_topics[reference_cit_topic_index]
        out_file.write("Paper ID: " + paper_id + "\n")
        out_file.write("Paper Title: " + paper_title + "\n")
        out_file.write("Paper Topic (Classified): " + str(classified_topic) + "\n")
        out_file.write("REFERENCE CitTopic Index: " + str(reference_cit_topic_index) + "\n")

        missing = utils.compute_missing_citations(reference_cit_topic_list, paper_out_citations)

        out_file.write("MISSING CITATIONS: " + str(missing) + "\n")
        out_file.write("---" + "\n")

