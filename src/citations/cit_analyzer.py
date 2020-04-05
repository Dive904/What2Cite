# This script is used to analyze the CitTopic, classifying all the papers in the CitTopic and getting the TopK

import pickle
import gc
import numpy as np

from keras.engine.saving import load_model
from keras_preprocessing.sequence import pad_sequences

from src.lstm import lstm_utils
from src.citations import cit_utils
from src.topicmodeller import tm_utils

# input
batch_number = 90
k = 3
topic_citations_filename = "../../output/official/topics_cits.txt"
topic_filename = "../../output/official/topics.txt"
abstract_document_topic_filename = "../../output/official/abstract_document_topic.csv"
lstm_model = "../../output/official/lstm.h5"
tokenizer_model = "../../output/official/tokenizer.pickle"
cit_topic_info_pickle_path = "../../output/official/cit_topic_info_pickle.pickle"

# output
topics_cits_labelled_filename = "../../output/official/topics_cits_labelled.txt"
topics_cits_labelled_pickle_filename = "../../output/official/topics_cits_labelled_pickle.pickle"

print("INFO: Reading Abstract-Document-Topic Matrix", end="... ")
abstract_document_topic_matrix = cit_utils.get_abstract_document_topic_matrix(abstract_document_topic_filename)
print("Done ✓")

print("INFO: Reading Citation Topic file", end="... ")
cit_topic = cit_utils.read_topics_cit_file(topic_citations_filename)
t = []
for c in cit_topic:
    tmp = []
    for x in c:
        tmp.append((x, None, 0))
    t.append(tmp)
cit_topic = t
print("Done ✓")

# in this part of code, we scan the CitTopics looking for classified Topic in the LDA (from the abstract document
# topic matrix)
first_step_result = []
for c in cit_topic:
    tmp = []
    for x in c:
        paper_id = x[0]
        paper_list = abstract_document_topic_matrix.get(paper_id)
        if paper_list is not None:
            topK_index, topK_elem = tm_utils.find_n_maximum(paper_list, k)
            second = list(zip(topK_index, topK_elem))
            third = 1
        else:
            second = None
            third = 0
        tmp.append((paper_id, second, third))
    first_step_result.append(tmp)

for t in first_step_result:
    print(t)
print()

print()
count = cit_utils.count_none(first_step_result)
print("******************* None Count: " + str(count))
print()

print("INFO: Cleaning memory", end="... ")
abstract_document_topic_matrix = None
cit_topic = None
t = None
gc.collect()
print("Done ✓")

# from the first scan, it's probably that we have some elements without classified topic. Now we look for other
# information using the rest of dataset
with open(cit_topic_info_pickle_path, 'rb') as handle:  # take the list of CitTopic score
    cit_topic_info = pickle.load(handle)

print("INFO: Reading Tokenizer and Neural Network", end="... ")
with open(tokenizer_model, 'rb') as handle:
    tokenizer = pickle.load(handle)

model = load_model(lstm_model)
print("Done ✓")

print("INFO: Classifing None papers", end="... ")
# with this type of loop, for optimal reasons, we scan the dataset only once
for i in range(len(first_step_result)):
    for j in range(len(first_step_result[i])):
        paper_id = first_step_result[i][j][0]
        topic = first_step_result[i][j][1]
        third = first_step_result[i][j][2]
        if topic is None:
            couple = cit_topic_info.get(paper_id)
            if (couple is not None) and (couple[1] is not None):
                text = couple[1]
                text = lstm_utils.preprocess_text(text)
                seq = tokenizer.texts_to_sequences([text])
                seq = pad_sequences(seq, padding='post', maxlen=200)
                yhat = model.predict(seq)
                yhat = list(yhat[0])
                topK_index, topK_elem = tm_utils.find_n_maximum(yhat, k)
                topK_elem = list(map(lambda y: np.round(y, 3), topK_elem))
                topic = list(zip(topK_index, topK_elem))
                third = 2
                first_step_result[i][j] = (paper_id, topic, third)
print("Done ✓")
second_step_result = first_step_result

print()
for t in second_step_result:
    print(t)
print()
count = cit_utils.count_none(second_step_result)
print("******************* None Count: " + str(count))
print()

new_cit_topic = None
gc.collect()

print("INFO: Writing output file", end="... ")
with open(topics_cits_labelled_filename, "w") as output_file:
    for i in range(len(second_step_result)):
        output_file.write(str(i) + " -> " + str(second_step_result[i]) + "\n")

with open(topics_cits_labelled_pickle_filename, "wb") as handle_file:
    pickle.dump(second_step_result, handle_file, protocol=pickle.HIGHEST_PROTOCOL)
print("Done ✓")
