from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from joblib import dump

import gc
import pandas as pd
import numpy as np

from src.topicmodeller import tm_utils

end_batch_number = 1
number_topics = 40
number_words = 30

print("INFO: Extracting " + str(end_batch_number) + " batch citations and removing empty citations", end="... ")
paper_info = tm_utils.extract_paper_info("C:\\Users\\Davide\\Desktop\\semanticdatasetextracted\\",
                                         end=end_batch_number)
citations = []
paper_info = paper_info[:500]
for data in paper_info:
    out_cit = data["outCitations"]
    if len(out_cit) != 0:
        citations.append(" ".join(out_cit))
print("Done ✓")

count_vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')

# Fit and transform the processed titles
print("INFO: Fitting count vectorizer", end="... ")
count_data = count_vectorizer.fit_transform(citations)
print("Done ✓")

print("INFO: Cleaning memory", end="... ")
citations_extracted = None
new_citations = None
gc.collect()
print("Done ✓")

# Create and fit the LDA model
print("INFO: Computing LDA", end="... ")
lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data)  # Print the topics found by the LDA model
print("Done ✓")

print("INFO: Saving models", end="... ")
dump(lda, "../../output/models/lda_cit_topic.jlb")
dump(count_vectorizer, "../../output/models/countvect_cit_topic.jlb")
print("Done ✓")

lda_output = lda.transform(count_data)

# column names
topicnames = ["Topic" + str(i) for i in range(lda.n_components)]

# index names
docnames = [paper_info[i]["id"] for i in range(len(paper_info))]

# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda_output, 3), columns=topicnames, index=docnames)

df_document_topic.to_csv("../../output/doctopic/abstract.csv")

# Topic-Keyword Matrix
df_topic_keywords = pd.DataFrame(lda.components_)

# Assign Column and Index
df_topic_keywords.columns = count_vectorizer.get_feature_names()
df_topic_keywords.index = topicnames

topic_keywords = tm_utils.get_topic_keywords(count_vectorizer, lda, number_words)
topic_keywords_perc = []

for i in range(len(topic_keywords)):
    tmp = []
    for word in topic_keywords[i]:
        perc = df_topic_keywords[word]["Topic" + str(i)]
        tmp.append((word, perc))
    topic_keywords_perc.append(tmp)

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords_perc)
df_topic_keywords.columns = ['(Word;Perc)_' + str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic ' + str(i) for i in range(df_topic_keywords.shape[0])]

df_topic_keywords.to_csv("../../output/doctopic/cit_topic_keywords.csv")


