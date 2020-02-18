from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

from src.topicmodeller import tm_utils

print("INFO: Extracting abstract", end="... ")
abstracts = tm_utils.extract_abstract("C:\\Users\\Davide\\Desktop\\TesiAPP\\subsetLDA\\")
print("Done ✓")

print("INFO: Preprocessing asbtract", end="... ")
abstracts = tm_utils.preprocess_abstract(abstracts)
print("Done ✓", end="\n\n")

# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')

# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(abstracts)

# Tweak the two parameters below
number_topics = 25
number_words = 5

print("-----------------------------------")
print("Number of abstract: " + str(len(abstracts)))
print("Number of topic: " + str(number_topics))
print("Number of words: " + str(number_words))
print("-----------------------------------", end="\n\n")

print("INFO: Waiting for LDA...")
# Create and fit the LDA model
lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data)  # Print the topics found by the LDA model
print("INFO: Topics found via LDA:")
tm_utils.print_topics(lda, count_vectorizer, number_words)
