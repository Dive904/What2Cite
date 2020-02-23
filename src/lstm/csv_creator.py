import csv

from src.fileutils import file_abstract
from src.lstm import lstm_utils

dataset = file_abstract.txt_lstm_dataset_reader("../../output/lstmdataset/final.txt")
inputrows = []

for data in dataset:
    topic = lstm_utils.extract_topic_from_dataset(data["topic"])
    tmp_row = [data["paperAbstract"], topic]
    inputrows.append(tmp_row)


with open('../../output/lstmdataset/data.csv', 'w', newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["text", "label"])
    for d in inputrows:
        writer.writerow(d)
