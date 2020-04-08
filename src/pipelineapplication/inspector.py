import pickle

from src.pipelineapplication import utils

cit_labelled_path = "../../output/official/topics_cits_labelled_pickle.pickle"
cit_topic_info_pickle_path = "../../output/official/cit_topic_info_pickle.pickle"

with open(cit_topic_info_pickle_path, 'rb') as handle:  # take the list of CitTopic score
    cit_topic_info = pickle.load(handle)

with open(cit_labelled_path, 'rb') as handle:  # take the list of CitTopic score
    # in this part, we read the cit topics labelled with other information.
    # We don't need that. So, we'll do a map removing all noise, getting only the paper id for all CitTopic
    cit_topics = pickle.load(handle)
    cit_topics = list(map(lambda x: list(map(lambda y: y[0], x)), cit_topics))

abstracts = utils.get_abstracts_to_analyze()  # get the abstract to analyze

for a in abstracts:
    citations = a["outCitations"]
    for cit in cit_topics:
        for c in cit:
            if c in citations:
                print(c)
    print("---")
