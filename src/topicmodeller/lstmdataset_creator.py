from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from joblib import dump

import io
import pandas as pd
import numpy as np
import gc

from src.topicmodeller import tm_utils

batch_number = 90  # change this for test
number_topic = 40  # change this for test
number_words = 7

print("INFO: Extracting " + str(batch_number) + " batch abstract and preprocessing", end="... ")
paper_info = tm_utils.extract_paper_info("C:\\Users\\Davide\\Desktop\\semanticdatasetextracted\\", end=batch_number)
abstracts_extracted = []
for data in paper_info:
    abstracts_extracted.append(tm_utils.preprocess_abstract(data["paperAbstract"]))
print("Done ✓")

print("INFO: Fitting count vectorizer", end="... ")
vectorizer = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
data_vectorized = vectorizer.fit_transform(abstracts_extracted)
print("Done ✓")

print("INFO: Cleaning memory", end="... ")
abstracts_extracted = None
gc.collect()
print("Done ✓")

print("INFO: Computing LDA", end="... ")
lda_model = LDA(n_components=number_topic, n_jobs=-1)
lda_model.fit(data_vectorized)
print("Done ✓")

print("INFO: Dumping LDA and CountVectorizer", end="... ")
dump(lda_model, "../../output/officialmodels/lda_abstract.jlb")
dump(vectorizer, "../../output/officialmodels/countvect_abstract.jlb")
print("Done ✓")

print("INFO: Transforming LDA", end="... ")
lda_output = lda_model.transform(data_vectorized)
print("Done ✓")

print("INFO: Making dataframe", end="... ")
topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]
docnames = [paper_info[i]["id"] for i in range(len(paper_info))]
df_document_topic = pd.DataFrame(np.round(lda_output, 3), columns=topicnames, index=docnames)
df_document_topic.to_csv("../../output/doctopic/abstract.csv")
print("Done ✓")

print("INFO: Getting documents dominant topic", end="... ")
filename = "../../output/lstmdataset/topics.txt"
tm_utils.print_topics_in_file(lda_model, vectorizer, number_words, filename, "w")
for n in range(lda_output.shape[0]):
    items = lda_output[n]
    max_indexes = tm_utils.find_n_maximum(items, 1)
    x = paper_info[n]
    x["topic"] = max_indexes[0]
    paper_info[n] = x
print("Done ✓")

print("INFO: Writing output file", end="... ")
with io.open("../../output/lstmdataset/final.txt", "w", encoding="utf-8") as f:
    for elem in paper_info:
        f.write("*** ID: " + elem["id"] + "\n")
        f.write("*** TITLE: " + elem["title"] + "\n")
        f.write("*** ABSTRACT: " + elem["paperAbstract"] + "\n")
        f.write("*** OUTCITATIONS: ")
        for i in range(len(elem["outCitations"])):
            f.write(elem["outCitations"][i])
            if i != len(elem["outCitations"]) - 1:
                f.write(", ")
        f.write("\n")
        f.write("*** TOPIC: " + str(elem["topic"]) + "\n")
        f.write("---" + "\n")
print("Done ✓")
