# This script is used to analyze the CitTopic, classifying all the papers in the CitTopic and getting the TopK
import pickle
import numpy as np
import gc

from time import process_time

from keras.engine.saving import load_model
from keras_preprocessing.sequence import pad_sequences

from src.fileutils import file_abstract
from src.citations import cit_utils
from src.neuralnetwork import lstm_utils
from src.pipelineapplication import utils
from src.topicmodeller import tm_utils

# input
close_dataset = "../../output/closedataset/closedataset.txt"
topic_citations_filename = "../../output/official/topics_cits.txt"
lstm_model = "../../output/official/neuralnetwork.h5"
tokenizer_model = "../../output/official/tokenizer.pickle"
topic_number = 40
t_for_true_prediction = 0.4  # probability threshold to consider a prediction as valid
abstract_document_topic_filename = "../../output/official/abstract_document_topic.csv"
K = 3

# output
cit_structure_pickle_path = "../../output/closedataset/cit_structure_pickle.pickle"

print("INFO: Reading Citation Topic file and initializing structure", end="... ")
cit_structure = {}
cit_topic = cit_utils.read_topics_cit_file(topic_citations_filename)
for cit in cit_topic:
    for c in cit:
        cit_structure[c] = [0 for x in range(topic_number)]
print("Done ✓")

print("INFO: Reading Abstract-Document-Topic Matrix", end="... ")
abstract_document_topic_matrix = cit_utils.get_abstract_document_topic_matrix(abstract_document_topic_filename)
print("Done ✓")

print("INFO: Reading close dataset and initializing topic", end="... ")
paper_info = file_abstract.txt_dataset_reader(close_dataset)
paper_info = paper_info[:500000]
for i in range(len(paper_info)):
    paper_info[i]["topic"] = None
print("Done ✓")

print("INFO: Reading Tokenizer and Neural Network", end="... ")
with open(tokenizer_model, 'rb') as handle:
    tokenizer = pickle.load(handle)

model = load_model(lstm_model)
print("Done ✓")

print("INFO: Analyzing first part", end="... ")
for i in range(len(paper_info)):
    paper_list = abstract_document_topic_matrix.get(paper_info[i]["id"])
    if paper_list is not None:
        topic = np.argmax(paper_list)
        paper_info[i]["topic"] = topic
print("Done ✓")

abstract_document_topic_matrix = None
gc.collect()

print("INFO: Analyzing second part", end="... ")
count = 0
for i in range(len(paper_info)):
    count += 1
    if count % 500 == 0:
        print("Counted: " + str(count))
    if paper_info[i]["topic"] is None:
        text = paper_info[i]["paperAbstract"]
        text = lstm_utils.preprocess_text(text)
        seq = tokenizer.texts_to_sequences([text])
        seq = pad_sequences(seq, padding='post', maxlen=200)
        topic = model.predict_classes(seq)[0]
        paper_info[i]["topic"] = topic
print("Done ✓")

print("INFO: Filling structure", end="... ")
for paper in paper_info:
    out_citations = paper["outCitations"]
    for out in out_citations:
        score_list = cit_structure.get(out)
        if score_list is not None:
            score_list[paper["topic"]] += 1
            cit_structure[out] = score_list
print("Done ✓")

print("INFO: Writing on file", end="... ")
with open(cit_structure_pickle_path, "wb") as handle_file:
    pickle.dump(cit_structure, handle_file, protocol=pickle.HIGHEST_PROTOCOL)
print("Done ✓")
