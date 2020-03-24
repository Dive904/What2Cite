from src.topicmodeller import tm_utils
from joblib import load

import pandas as pd
import numpy as np

batch_number = 90  # change this for test

print("INFO: Extracting " + str(batch_number) + " batch abstract and preprocessing", end="... ")
paper_info = tm_utils.extract_paper_info("C:\\Users\\Davide\\Desktop\\semanticdatasetextracted\\", end=batch_number)
abstracts_extracted = []
for data in paper_info:
    abstracts_extracted.append(tm_utils.preprocess_abstract(data["paperAbstract"]))
print("Done ✓")

lda_model = load("../../output/official/lda_abstract.jlb")
vectorizer = load("../../output/official/countvect_abstract.jlb")

data_vectorized = vectorizer.fit_transform(abstracts_extracted)

lda_output = lda_model.transform(data_vectorized)

print("INFO: Making dataframe", end="... ")
topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]
docnames = [paper_info[i]["id"] for i in range(len(paper_info))]
df_document_topic = pd.DataFrame(np.round(lda_output, 3), columns=topicnames, index=docnames)
df_document_topic.to_csv("../../output/doctopic/abstract_document_topic.csv")
print("Done ✓")

