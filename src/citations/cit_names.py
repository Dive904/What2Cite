import pickle

cit_topic_info_pickle_path = "../../output/official/cit_topic_info_pickle.pickle"
citopic = "82266f6103bade9005ec555ed06ba20b5210ff22 0c8413ab8de0c1b8f2e86402b8d737d94371610f 0a2586e0a5f8bb4e35aa0763a6b8bca428af6bd2 cd5a26b89f0799db1cbc1dff5607cb6815739fe7 daa63f57c3fbe994c4356f8d986a22e696e776d2 2e2089ae76fe914706e6fa90081a79c8fe01611e 93bc65d2842b8cc5f3cf72ebc5b8f75daeacea35"

with open(cit_topic_info_pickle_path, 'rb') as handle:  # take the list of CitTopic score
    cit_topic_info = pickle.load(handle)

citopic = citopic.split()
print(citopic)

for c in citopic:
    print(c + " -> " + str(cit_topic_info.get(c)[0]))
