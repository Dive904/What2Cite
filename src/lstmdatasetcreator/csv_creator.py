import csv

from src.fileutils import file_abstract
from src.lstmdatasetcreator import utils

print("INFO: Extracting dataset", end="... ")
dataset = file_abstract.txt_lstm_dataset_reader("../../output/lstmdataset/final.txt")
inputrows = []
print("Done ✓")

print("INFO: Creating rows", end="... ")
for data in dataset:
    topic = utils.extract_topic_from_dataset(data["topic"])
    tmp_row = [data["paperAbstract"], topic]
    inputrows.append(tmp_row)
print("Done ✓")

print("INFO: Writing csv", end="... ")
with open('../../output/lstmdataset/data.csv', 'w', newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["text", "label"])
    for d in inputrows:
        writer.writerow(d)
print("Done ✓")
