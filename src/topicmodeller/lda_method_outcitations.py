from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from joblib import dump

import gc
import pandas as pd

from src.topicmodeller import tm_utils

# input
end_batch_number = 22  # almost 500k papers
number_topics = 750
number_words = 50

# output
cit_topic_keywords_filename = "../../output/doctopic/cit_topic_keywords.csv"

print("INFO: Extracting " + str(end_batch_number) + " batch citations and removing empty citations", end="... ")
paper_info = tm_utils.extract_paper_info("C:\\Users\\Davide\\Desktop\\semanticdatasetextracted\\",
                                         end=end_batch_number)
citations = []
for data in paper_info:
    out_cit = data["outCitations"]
    if len(out_cit) != 0:
        citations.append(" ".join(out_cit))
print("Done ✓")

count_vectorizer = CountVectorizer(max_df=0.8, min_df=10)

print("INFO: Fitting count vectorizer", end="... ")
count_data = count_vectorizer.fit_transform(citations)
print("Done ✓")

print("INFO: Cleaning memory", end="... ")
citations = None
paper_info = None
gc.collect()
print("Done ✓")

# Create and fit the LDA model
print("INFO: Computing LDA", end="... ")
lda = LDA(n_components=number_topics, n_jobs=1)
lda.fit(count_data)
print("Done ✓")

print("INFO: Saving models", end="... ")
dump(lda, "../../output/models/lda_cit_topic.jlb")
dump(count_vectorizer, "../../output/models/countvect_cit_topic.jlb")
print("Done ✓")

lda_output = lda.transform(count_data)

# column names
topicnames = ["Topic" + str(i) for i in range(lda.n_components)]

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

df_topic_keywords.to_csv(cit_topic_keywords_filename)
