from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from joblib import dump
import io

from src.topicmodeller import tm_utils


def compute(number_topics, paper_info, abstracts):
    # Tweak the parameter below
    number_words = 7

    # Initialise the count vectorizer with the English stop words
    count_vectorizer = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')

    # Fit and transform the processed titles
    print("INFO: Fitting count vectorizer", end="... ")
    count_data = count_vectorizer.fit_transform(abstracts)
    print("Done ✓")

    # Create and fit the LDA model
    print("INFO: Computing LDA", end="... ")
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(count_data)  # Print the topics found by the LDA model
    print("Done ✓")

    print("INFO: Dumping LDA", end="... ")
    dump(lda, "../../output/models/lda.jlb")
    print("Done ✓")

    print("INFO: Transforming LDA", end="... ")
    doc_topic = lda.transform(count_data)
    print("Done ✓")
    print("INFO: Getting document association", end="... ")
    filename = "../../output/lstmdataset/topics.txt"
    tm_utils.print_topics_in_file(lda, count_vectorizer, number_words, filename)
    for n in range(doc_topic.shape[0]):
        topic_most_pr = doc_topic[n].argmax()
        x = paper_info[n]
        x["topic"] = topic_most_pr
        paper_info[n] = x
    print("Done ✓")

    words = count_vectorizer.get_feature_names()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        strng = ""
        strng += " ".join([words[i] for i in topic.argsort()[:-number_words - 1:-1]])
        topics.append(strng)

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
            f.write("*** TOPIC: " + str(elem["topic"]) + " - " + topics[int(elem["topic"])] + "\n")
            f.write("---" + "\n")
    print("Done ✓")


batch_number = 65  # change this for test
number_topic = 55  # change this for test

print("INFO: Extracting " + str(batch_number) + " batch abstract", end="... ")
paper_info = tm_utils.extract_paper_info("C:\\Users\\Davide\\Desktop\\semanticdatasetextracted\\",
                                         end=batch_number)
abstracts_extracted = tm_utils.extract_only_abstract("C:\\Users\\Davide\\Desktop\\semanticdatasetextracted\\", end=batch_number)
print("Done ✓")
print("INFO: Preprocessing asbtract", end="... ")
abstracts_extracted = tm_utils.preprocess_abstract(abstracts_extracted)
print("Done ✓", end="\n\n")
compute(number_topic, paper_info, abstracts_extracted)