from joblib import load
from src.topicmodeller import tm_utils

batch_number = 1
number_words = 10

print("INFO: Extracting " + str(batch_number) + " batch abstract", end="... ")
paper_info = tm_utils.extract_paper_info("C:\\Users\\Davide\\Desktop\\semanticdatasetextracted\\",
                                         end=batch_number)
abstracts_extracted = tm_utils.extract_only_abstract("C:\\Users\\Davide\\Desktop\\semanticdatasetextracted\\", end=batch_number)
print("Done ✓")
print("INFO: Preprocessing asbtract", end="... ")
abstracts_extracted = tm_utils.preprocess_abstract(abstracts_extracted)
print("Done ✓", end="\n\n")

lda = load("../../output/officialmodels/lda.jlb")
countvect = load("../../output/officialmodels/countvect.jlb")

tm_utils.print_topics(lda, countvect, number_words)

count_data = countvect.fit_transform(abstracts_extracted)
doc_topic = lda.transform(count_data)

for n in range(doc_topic.shape[0]):
    topic_most_pr = doc_topic[n].argsort()
    print(topic_most_pr)
    '''
    x = paper_info[n]
    x["topic"] = topic_most_pr
    paper_info[n] = x
    print(doc_topic[n])
    '''

words = countvect.get_feature_names()
topics = []
for topic_idx, topic in enumerate(lda.components_):
    strng = ""
    strng += " ".join([words[i] for i in topic.argsort()[:-number_words - 1:-1]])
    topics.append(strng)

print(topics)
