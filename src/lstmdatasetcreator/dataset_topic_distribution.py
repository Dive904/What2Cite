from src.fileutils import file_abstract
from src.lstmdatasetcreator import utils

import numpy as np
import matplotlib.pyplot as plt

print("INFO: Extracting dataset", end="... ")
dataset = file_abstract.txt_lstm_dataset_reader("../../output/lstmdataset/final.txt")
print("Done ✓")

number_topic = 40

labels = [str(i) for i in range(number_topic)]

dic = {}

for l in labels:
    dic[l] = 0

print("INFO: Analyzing dataset", end="... ")
for data in dataset:
    dic[utils.extract_topic_from_dataset(data["topic"])] += 1

labels = []
values = []
for key in dic.keys():
    labels.append(key)
    values.append(dic[key])
print("Done ✓")

print("INFO: Creating plot", end="... ")
plt.rcdefaults()
y_pos = np.arange(len(labels))
plt.barh(y_pos, values, align='center', alpha=0.5)
plt.yticks(y_pos, labels)
plt.xlabel('Abstracts')
plt.title('Distribution of Topics')

plt.savefig(str(number_topic) + "_topic_distribution.jpg")
print("Done ✓")
