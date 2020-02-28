import csv

from src.fileutils import file_abstract
from src.lstm import lstm_utils

print("INFO: Extracting dataset", end="... ")
dataset = file_abstract.txt_lstm_dataset_reader("../../output/lstmdataset/final.txt")
inputrows = []
print("Done ✓")

label_list = [0 for i in range(55)]

print("INFO: Creating rows", end="... ")
for data in dataset:
    tmp_list = list(label_list)
    topic = lstm_utils.extract_topic_from_dataset(data["topic"])
    tmp_list[int(topic)] = 1
    tmp_row = [data["paperAbstract"]] + tmp_list
    inputrows.append(tmp_row)
print("Done ✓")

print("INFO: Writing csv", end="... ")
label_list = [str(i) for i in range(55)]
with open('../../output/lstmdataset/data_multilabel.csv', 'w', newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["text"] + label_list)
    for d in inputrows:
        writer.writerow(d)
print("Done ✓")
