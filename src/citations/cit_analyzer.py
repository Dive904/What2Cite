from src.fileutils import file_abstract
from src.citations import cit_utils

topic_citations_filename = "../../output/citations/topics_cits.txt"
topic_filename = "../../output/lstmdataset/topics.txt"
lstm_dataset = "../../output/lstmdataset/final.txt"

citations = []
dataset = file_abstract.txt_lstm_dataset_reader(lstm_dataset)

with open(topic_citations_filename, mode="r", encoding="utf-8") as cit_file:
    lines = cit_file.readlines()
    for line in lines:
        if line.startswith("Topic "):
            cit_list = line.split(": ")[1].split()
            citations.append(cit_list)

cit_topics = []

for cit_list in citations:
    topics = []
    for cit in cit_list:
        topics.append(cit_utils.get_cit_topic(cit, dataset))
    cit_topics.append(topics)

for t in cit_topics:
    print(t)
