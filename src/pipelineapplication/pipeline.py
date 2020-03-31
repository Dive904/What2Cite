# This script is used to implement the pipeline for only specific abstracts given in input in a list

import pickle
import numpy as np
import gc

from keras_preprocessing.sequence import pad_sequences
from tensorflow_core.python.keras.models import load_model

from src.fileutils import file_abstract
from src.pipelineapplication import utils
from src.lstm import lstm_utils
from src.topicmodeller import tm_utils

# input
cit_labelled_analyzed_pickle_path = "../../output/official/cit_labelled_with_final_topic_pickle.pickle"
lstm_model = "../../output/official/lstm.h5"
tokenizer_model = "../../output/official/tokenizer.pickle"
cit_labelled_path = "../../output/official/topics_cits_labelled_pickle.pickle"
semanticdatasetextracted_path = "C:\\Users\\Davide\\Desktop\\semanticdatasetextracted\\"
batch_number = 90
P = 1
Pt = 3

# output
missig_citation_path = "../../output/official/missing_citations.txt"

with open(cit_labelled_path, 'rb') as handle:  # take the list of CitTopic score
    # in this part, we read the cit topics labelled with other information.
    # We don't need that. So, we'll do a map removing all noise, getting only the paper id for all CitTopic
    cit_topics = pickle.load(handle)
    cit_topics = list(map(lambda x: list(map(lambda y: y[0], x)), cit_topics))

print("INFO: Extracting dataset", end="... ")
dataset = file_abstract.txt_lstm_dataset_reader("../../output/official/final.txt")
print("Done ✓")

with open(cit_labelled_analyzed_pickle_path, 'rb') as handle:  # take the list of CitTopic score
    cit_topic_labelled = pickle.load(handle)

abstracts = utils.get_abstracts_to_analyze()  # get the abstract to analyze

abstracts_prep = list(map(lambda x: lstm_utils.preprocess_text(x[0]), abstracts))  # prepare text for classification

with open(tokenizer_model, 'rb') as handle:  # get the tokenizer
    tokenizer = pickle.load(handle)

print("INFO: Tokenizing sequences", end="... ")
seq = tokenizer.texts_to_sequences(abstracts_prep)
seq = pad_sequences(seq, padding='post', maxlen=200)
print("Done ✓")

print("INFO: Getting predictions", end="... ")
model = load_model(lstm_model)  # load model from single file
yhat = model.predict(seq)  # make predictions
predicted_topic = []
for y in yhat:
    # at the end of the loop, predicted_topic will contain a list of couples where the first element is the predicted
    # topic and the second element is the probability of the predicted topic
    p = np.argmax(y)
    predicted_topic.append((p, np.round(y[p], 3)))
print("Done ✓")

reference_cit_topics = []
for i in range(len(predicted_topic)):
    # at the end of the loop, reference_cit_topics will contain a list of couple, where the first element is the
    # classified topic for the paper, and the second element is reference CitTopic of that classified paper
    topic = predicted_topic[i][0]  # get the topic
    classification_score = predicted_topic[i][1]
    score_on_predicted_topic = []
    tmp = []
    for i in range(len(cit_topic_labelled)):
        topic_score = cit_topic_labelled[i][topic]  # get the score for that specific topic
        if topic_score > Pt:  # if score is higher than a given input
            tmp.append(i)
    # append a couple with first the abstract id and second the reference CitTopic index
    reference_cit_topics.append((topic, tmp))

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

to_write = []
for i in range(len(data_out_citations)):
    reference_cit_topic_index = reference_cit_topics[i][1]
    reference_cit_topic_list = []
    for ii in reference_cit_topic_index:
        reference_cit_topic_list.append(cit_topics[ii])
    paper_out_citations = data_out_citations[i][2]

    info_to_write = {
        "paper_id": data_out_citations[i][0],
        "paper_title": data_out_citations[i][1],
        "paper_out_citations": data_out_citations[i][2],
        "classified_topic": reference_cit_topics[i][0],
        "reference_cit_topic_index": reference_cit_topic_index,
        "reference_cit_topic_list": reference_cit_topic_list
    }

    missings = []
    for cittopic in reference_cit_topic_list:
        missing = utils.compute_missing_citations(cittopic, paper_out_citations)
        missings.append(missing)

    missing_title_dict = []
    for missing in missings:
        # with this section, we're looking for paper titles in missing citations, only in the first part of the dataset
        # because, for memory reasons, the two parts of dataset cannot be hold at the same, so we must split
        # the computing
        missing_title = [(paper_id, None) for paper_id in missing]
        for data in dataset:
            # we do in this way, for optimal reason: the dataset is very large, in this way, we read the
            # dataset only once
            try:
                h = missing.index(data["id"])
                missing_title[h] = (data["id"], data["title"])
            except ValueError:
                h = None
        missing_title_dict.append(missing_title)

    info_to_write["missing_title"] = missing_title_dict
    to_write.append(info_to_write)

print("INFO: Cleaning up and reading", end="... ")
# now, we clean up memory from first part of dataset, and read the second part of it
dataset = None
gc.collect()
paper_info = tm_utils.extract_paper_info("C:\\Users\\Davide\\Desktop\\semanticdatasetextracted\\", start=batch_number)
print("Done ✓")

# continue with looking up missing title with the rest of dataset
print("INFO: Analyzing", end="... ")
for elem in to_write:
    missing_titles = elem["missing_title"]  # get list of missing title
    for missing_title in missing_titles:
        missing = list(map(lambda m: m[0], missing_title))  # remove second element
        missing_title_none = list(filter(lambda m: m[1] is None, missing_title))  # getting couples with None
        missing_title_none = list(map(lambda m: m[0], missing_title_none))  # getting only the paper id
        for p in paper_info:  # scannig the second part of the dataset
            if p["id"] in missing_title_none:  # if an id is in the missed title with none
                try:
                    h = missing.index(p["id"])  # get the index of that id in the missing list
                    missing_title[h] = (p["id"], p["title"])  # update the missing list with new information
                except ValueError:  # index() method can raise this exception, when the element is not found
                    h = None
    elem["missing_title"] = missing_titles
print("Done ✓")

paper_info = None
gc.collect()

print("INFO: Writing output file", end="... ")
with open(missig_citation_path, "w") as out_file:
    for elem in to_write:
        out_file.write("Paper ID: " + str(elem["paper_id"]) + "\n")
        out_file.write("Paper Title: " + str(elem["paper_title"]) + "\n")
        out_file.write("Paper Topic (Classified): " + str(elem["classified_topic"]) + "\n")
        out_file.write("Found " + str(len(elem["reference_cit_topic_index"])) + " reference CitTopics")
        for i in range(len(elem["reference_cit_topic_index"])):
            out_file.write("REFERENCE CitTopic Index: " + str(elem["reference_cit_topic_index"][i]) + "\n")
            cit_topic_length = len(cit_topics[elem["reference_cit_topic_index"][i]])
            missing_len = len(elem["missing_title"][i])
            out_file.write("MISSING CITATIONS WITH TITLES (" + str(missing_len) + " out of "
                           + str(cit_topic_length) + "): " + "\n")
            for x in elem["missing_title"][i]:
                out_file.write(str(x) + "\n")
            out_file.write("---" + "\n")
    out_file.write("***" + "\n")
print("Done ✓")
