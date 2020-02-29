import csv

from src.fileutils import file_abstract
from src.lstm import lstm_utils

print("INFO: Extracting dataset", end="... ")
dataset = file_abstract.txt_lstm_dataset_reader("../../output/lstmdataset/final.txt")
inputrows = []
print("Done ✓")

print("INFO: Creating rows", end="... ")
for data in dataset:
    topic = lstm_utils.extract_topic_from_dataset(data["topic"])
    tmp_row = [data["paperAbstract"], topic]
    inputrows.append(tmp_row)
print("Done ✓")

rows = 1000
print("INFO: Writing csv", end="... ")
with open('../../output/lstmdataset/data_reduced.csv', 'w', newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["text", "label"])
    for d in inputrows[:rows]:
        writer.writerow(d)
print("Done ✓")
