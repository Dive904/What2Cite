import pickle
import numpy as np

cit_labelled_path = "../../output/official/topics_cits_labelled_pickle.pickle"
cit_labelled_analyzed_path = "../../output/official/cit_labelled_with_final_topic.txt"

w = [1, 0.75, 0.50]

with open(cit_labelled_path, 'rb') as handle:
    cit_topic_labelled = pickle.load(handle)

first_step_result = []

print("INFO: Analyzing", end="... ")
for c in cit_topic_labelled:
    topics = [1 for i in range(40)]
    for record in c:
        if record[1] is None:
            topics = list(map(lambda x: x + 0.15, topics))
        else:
            topic_list = record[1]
            for i in range(len(topic_list)):
                topic = topic_list[i][0]
                topics[topic] += w[i]
    first_step_result.append(topics)
print("Done ✓")

first_step_result = list(map(lambda x: np.round(x, 2), first_step_result))
second_step_result = []

for l in first_step_result:
    maximum = max(l)
    pos = []
    for i in range(len(l)):
        if l[i] == maximum:
            pos.append(i)

    second_step_result.append((list(l), pos))

with open(cit_labelled_analyzed_path, "w") as out_file:
    for i in range(len(second_step_result)):
        out_file.write(str(i) + " -> " + str(second_step_result[i][0]) + " -> " + str(second_step_result[i][1]) + "\n")
