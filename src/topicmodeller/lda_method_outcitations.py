from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from joblib import dump

import gc

from src.topicmodeller import tm_utils

out_filename = "../../output/citations/topics_cits.txt"
end_batch_number = 1
number_topics = 40
number_words = 10

print("INFO: Extracting citations", end="... ")
citations_extracted = tm_utils.extract_only_citations("C:\\Users\\Davide\\Desktop\\semanticdatasetextracted\\",
                                                      end=end_batch_number)
print("Done ✓")

new_citations = []

print("INFO: Removing empty citations", end="... ")
for cit in citations_extracted:
    if len(cit) != 0:
        new_citations.append(cit)
print("Done ✓")

# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')

# Fit and transform the processed titles
print("INFO: Fitting count vectorizer", end="... ")
count_data = count_vectorizer.fit_transform(new_citations)
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

print("INFO: Writing topics on file", end="... ")
tm_utils.print_topics(lda, count_vectorizer, number_words)
print("Done ✓")

print("INFO: Saving models", end="... ")
dump(lda, "../../output/models/lda_cit_topic.jlb")
dump(count_vectorizer, "../../output/models/countvect_cit_topic.jlb")
print("Done ✓")
