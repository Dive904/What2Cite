import pickle

from src.topicmodeller import tm_utils
from src.citations import cit_utils


# input
semanticdatasetextracted_all_filename = "C:\\Users\\Davide\\Desktop\\semanticdatasetextracted_all\\"
topics_cit_filename = "../../output/official/topics_cits.txt"
partition_read_dataset = [(None, 90), (90, None)]

# output
cit_topic_info_pickle_path = "../../output/official/cit_topic_info_pickle.pickle"

print("INFO: Initializing dictionary", end="... ")
cit_topic_info = {}
cit_topics = cit_utils.read_topics_cit_file(topics_cit_filename)
for cit in cit_topics:
    for c in cit:
        cit_topic_info[c] = (None, None)
cit_topic_info_keys = cit_topic_info.keys()
print("Done ✓")

for partition in partition_read_dataset:
    print("INFO: Reading part of dataset", end="... ")
    paper_info = tm_utils.extract_paper_info(semanticdatasetextracted_all_filename,
                                             start=partition[0],
                                             end=partition[1])
    print("Done ✓")

    print("INFO: Analyzing", end="... ")
    for paper in paper_info:
        if paper["id"] in cit_topic_info_keys:
            cit_topic_info[paper["id"]] = (paper["title"], paper["paperAbstract"])
    print("Done ✓")

print("INFO: Writing on file", end="... ")
with open(cit_topic_info_pickle_path, "wb") as handle_file:  # saving list to use in pipeline application
    pickle.dump(cit_topic_info, handle_file, protocol=pickle.HIGHEST_PROTOCOL)
print("Done ✓")
