from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

from src.topicmodeller import tm_utils

out_filename = "../../output/citations/topics_cits.txt"
batch_number = 1
number_topics = 40
number_words = 10

print("INFO: Extracting citations", end="... ")
citations_extracted = tm_utils.extract_only_citations("C:\\Users\\Davide\\Desktop\\semanticdatasetextracted\\",
                                                      end=batch_number)
print("Done ✓")

new_citations = []

print("INFO: Removing empty citations", end="... ")
for cit in citations_extracted:
    if len(cit) != 0:
        new_citations.append(cit)
print("Done ✓")

# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')

# Fit and transform the processed titles
print("INFO: Fitting count vectorizer", end="... ")
count_data = count_vectorizer.fit_transform(new_citations)
print("Done ✓")

# Create and fit the LDA model
print("INFO: Computing LDA", end="... ")
lda = LDA(n_components=number_topics, n_jobs=1)
lda.fit(count_data)  # Print the topics found by the LDA model
print("Done ✓")

print("INFO: Writing topics on file", end="... ")
tm_utils.print_topics_in_file(lda, count_vectorizer, number_words, out_filename)
print("Done ✓")
