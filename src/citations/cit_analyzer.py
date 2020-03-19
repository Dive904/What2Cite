from keras.preprocessing.sequence import pad_sequences
from tensorflow_core.python.keras.models import load_model

import pickle
import gc

from src.lstm import lstm_utils
from src.fileutils import file_abstract
from src.citations import cit_utils
from src.topicmodeller import tm_utils

batch_number = 90
topic_citations_filename = "../../output/citations/topics_cits.txt"
topic_filename = "../../output/lstmdataset/topics.txt"
lstm_dataset = "../../output/lstmdataset/final.txt"

print("INFO: Reading first part of dataset", end="... ")
dataset = file_abstract.txt_lstm_dataset_reader(lstm_dataset)
print("Done ✓")

citations = []
print("INFO: Reading Citation Topic", end="... ")
with open(topic_citations_filename, mode="r", encoding="utf-8") as cit_file:
    lines = cit_file.readlines()
    for line in lines:
        if line.startswith("Topic "):
            cit_list = line.split(": ")[1].split()
            citations.append(cit_list)
print("Done ✓")
print(len(citations))
print("INFO: Creating first list couple", end="... ")
cit_topics = []
for cit_list in citations:
    topics = []
    for cit in cit_list:
        topic = cit_utils.get_cit_topic(cit, dataset)
        topics.append((cit, topic))
    cit_topics.append(topics)
print("Done ✓")

print()
for t in cit_topics:
    print(t)
print()
count = cit_utils.count_none(cit_topics)
print("******************* None Count: " + str(count))
print()

print("INFO: Cleaning memory", end="... ")
dataset = None
gc.collect()
print("Done ✓")

print("INFO: Reading rest of dataset", end="... ")
paper_info = tm_utils.extract_paper_info("C:\\Users\\Davide\\Desktop\\semanticdatasetextracted\\",
                                         start=batch_number)
print("Done ✓")

print("INFO: Reading Tokenizer and Neural Network", end="... ")
with open('../../output/models/tokenizer_73acc_glove840.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = load_model("../../output/models/lstm_73acc_glove840.h5")
print("Done ✓")

print("INFO: Classifing None papers", end="... ")
new_cit_topic = []
for cit_list in cit_topics:
    tmp_list = []
    for couple in cit_list:
        id = couple[0]
        topic = couple[1]
        if topic is None:
            text = None
            for p in paper_info:
                if p["id"] == id:
                    text = p["paperAbstract"]
            if text is not None:
                text = lstm_utils.preprocess_text(text)
                seq = tokenizer.texts_to_sequences([text])
                seq = pad_sequences(seq, padding='post', maxlen=200)
                yhat = model.predict_classes(seq)
                topic = str(yhat[0])
        tmp_list.append((id, topic))
    new_cit_topic.append(tmp_list)
print("Done ✓")

print()
for t in new_cit_topic:
    print(t)
print()
count = cit_utils.count_none(new_cit_topic)
print("******************* None Count: " + str(count))
print()

list_to_write = []
for t in new_cit_topic:
    tmp_list = []
    for x in t:
        tmp_list.append(x[1])
    list_to_write.append(tmp_list)

new_cit_topic = None
gc.collect()

print("INFO: Writing output file", end="... ")
with open("../../output/citations/topics_cits_labelled.txt", "w") as output_file:
    for i in range(len(list_to_write)):
        output_file.write(str(i) + " -> " + str(list_to_write[i]) + "\n")
print("Done ✓")
