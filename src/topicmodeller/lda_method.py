from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import io

from src.topicmodeller import tm_utils


def compute(batch_number, number_topics):
    # Tweak the parameter below
    number_words = 5

    print("INFO: Extracting " + str(batch_number) + " batch abstract", end="... ")
    abstracts = tm_utils.extract_abstract("C:\\Users\\Davide\\Desktop\\TesiAPP\\", batch_number)
    print("Done ✓")

    print("INFO: Preprocessing asbtract", end="... ")
    abstracts = tm_utils.preprocess_abstract(abstracts)
    print("Done ✓")

    # Initialise the count vectorizer with the English stop words
    count_vectorizer = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')

    # Fit and transform the processed titles
    print("INFO: Fitting count vectorizer", end="... ")
    count_data = count_vectorizer.fit_transform(abstracts)
    print("Done ✓", end="\n\n")

    print("-----------------------------------")
    print("Number of batch: " + batch_number)
    print("Number of abstract: " + str(len(abstracts)))
    print("Number of topic: " + str(number_topics))
    print("Number of words: " + str(number_words))
    print("-----------------------------------", end="\n\n")

    filename = "../../output/ldatest/" + str(batch_number) + "_" + str(number_topics) + ".txt"
    with io.open(filename, "w", encoding="utf-8") as f:
        f.write("Number of batch: " + batch_number + "\n")
        f.write("Number of abstract: " + str(len(abstracts)) + "\n")
        f.write("Number of topic: " + str(number_topics) + "\n")
        f.write("Number of words: " + str(number_words) + "\n")

    print("INFO: Waiting for LDA...")
    # Create and fit the LDA model
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(count_data)  # Print the topics found by the LDA model
    print("INFO: Topics found via LDA:")
    tm_utils.print_topics_in_file(lda, count_vectorizer, number_words, filename)

    return filename
