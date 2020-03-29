# This script is used to analyze the score of every topic in the CitTopics

import pickle
import numpy as np

# input
cit_labelled_path = "../../output/official/topics_cits_labelled_pickle.pickle"
w = [1, 0.75, 0.50]  # score of every position

# output
cit_labelled_analyzed_path = "../../output/official/cit_labelled_with_final_topic.txt"
cit_labelled_analyzed_pickle_path = "../../output/official/cit_labelled_with_final_topic_pickle.pickle"

with open(cit_labelled_path, 'rb') as handle:  # take the list of CitTopic score
    cit_topic_labelled = pickle.load(handle)

first_step_result = []

print("INFO: Analyzing", end="... ")
for c in cit_topic_labelled:  # put the scores in every topic
    topics = [1 for i in range(40)]
    for record in c:
        if record[1] is None:  # if there is a None object, every topic gets 0.15 points
            topics = list(map(lambda x: x + 0.15, topics))
        else:
            topic_list = record[1]
            for i in range(len(topic_list)):
                topic = topic_list[i][0]
                topics[topic] += w[i]
    first_step_result.append(topics)
print("Done ✓")

first_step_result = list(map(lambda x: np.round(x, 2), first_step_result))  # rounding scores
second_step_result = []
list_to_pickle = []

# this part is used to get the maximal topic for a CitTopic, maybe this will not necessary anymore...
for l in first_step_result:
    maximum = max(l)
    pos = []
    for i in range(len(l)):
        if l[i] == maximum:
            pos.append(i)

    second_step_result.append((list(l), pos))
    list_to_pickle.append(list(l))

with open(cit_labelled_analyzed_path, "w") as out_file:  # writing results on disk
    for i in range(len(second_step_result)):
        out_file.write(str(i) + " -> " + str(second_step_result[i][0]) + " -> " + str(second_step_result[i][1]) + "\n")

with open(cit_labelled_analyzed_pickle_path, "wb") as handle_file:  # saving list to use in pipeline application
    pickle.dump(list_to_pickle, handle_file, protocol=pickle.HIGHEST_PROTOCOL)
