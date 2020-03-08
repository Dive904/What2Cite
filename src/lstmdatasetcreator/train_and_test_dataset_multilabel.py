import csv

from src.lstmdatasetcreator import utils
from src.fileutils import file_abstract


# threshold must not be higher than 18500
threshold = 2000
val_dataset_length_perc = 10  # per topic
training_dataset_length_perc = 60  # per topic
test_dataset_length_perc = 30  # per topic

number_topic = 40
result_dataset = {}

print("INFO: Extracting dataset", end="... ")
dataset = file_abstract.txt_lstm_dataset_reader("../../output/lstmdataset/final.txt")
print("Done ✓")

labels = [str(i) for i in range(number_topic)]

for l in labels:
    result_dataset[l] = []

print("INFO: Getting abstracts", end="... ")
for data in dataset:
    topic = utils.extract_topic_from_dataset(data["topic"])
    if len(result_dataset[topic]) < threshold:
        result_dataset[topic].append(data["paperAbstract"])
print("Done ✓")

val_dataset_length = int((threshold * val_dataset_length_perc) / 100)
training_dataset_length = int((threshold * training_dataset_length_perc) / 100)
test_dataset_length = int((threshold * test_dataset_length_perc) / 100)

print()
print("Training dataset topic length: " + str(training_dataset_length))
print("Test dataset topic length: " + str(test_dataset_length))
print("Val dataset topic length: " + str(val_dataset_length))
print()

training_dataset = {}
test_dataset = {}
val_dataset = {}

for l in labels:
    training_dataset[l] = []
    test_dataset[l] = []
    val_dataset[l] = []

print("INFO: Creating datasets", end="... ")
for topic in result_dataset.keys():
    for i in range(training_dataset_length):
        training_dataset[topic].append(result_dataset[topic][0])
        del result_dataset[topic][0]

    for i in range(test_dataset_length):
        test_dataset[topic].append(result_dataset[topic][0])
        del result_dataset[topic][0]

    for i in range(val_dataset_length):
        val_dataset[topic].append(result_dataset[topic][0])
        del result_dataset[topic][0]
print("Done ✓")

print("INFO: Writing training file", end="... ")
with open("../../output/lstmdataset/trainingdataset_multilabel.csv", "w", newline='', encoding="utf-8") as train_file:
    writer = csv.writer(train_file)
    writer.writerow(["text"] + labels)
    for topic in labels:
        abstracts = training_dataset[topic]
        for i in range(len(abstracts)):
            t = [0 for i in range(40)]
            t[int(topic)] = 1
            writer.writerow([abstracts[i]] + t)
print("Done ✓")

print("INFO: Writing test file", end="... ")
with open("../../output/lstmdataset/testdataset_multilabel.csv", "w", newline='', encoding="utf-8") as test_file:
    writer = csv.writer(test_file)
    writer.writerow(["text", "label"])
    for topic in labels:
        abstracts = test_dataset[topic]
        for i in range(len(abstracts)):
            t = [0 for i in range(40)]
            t[int(topic)] = 1
            writer.writerow([abstracts[i]] + t)
print("Done ✓")

print("INFO: Writing validation file", end="... ")
with open("../../output/lstmdataset/valdataset_multilabel.csv", "w", newline='', encoding="utf-8") as val_file:
    writer = csv.writer(val_file)
    writer.writerow(["text", "label"])
    for topic in labels:
        abstracts = val_dataset[topic]
        for i in range(len(abstracts)):
            t = [0 for i in range(40)]
            t[int(topic)] = 1
            writer.writerow([abstracts[i]] + t)
print("Done ✓")
