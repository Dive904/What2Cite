from datetime import datetime
import io

from src.topicmodeller import tm_utils
from src.topicmodeller import lda_method_abstract

# Number of Batch, Number of Topic
# test_index = [(25, 25), (25, 50), (50, 25), (50, 50), (65, 25), (65, 50)]
test_index = [(65, 45), (65, 50), (65, 55)]

batch_number = 0
number_topic = 0
abstracts = []
for index in test_index:
    prev_bn = batch_number
    prev_nt = number_topic

    batch_number = index[0]
    number_topic = index[1]

    if batch_number == prev_bn:
        print("INFO: Batch number is not changed, extracting and preprocessing is not necessary.")
        abstracts_extracted = []
        d_prep = 0
    else:
        print("INFO: Extracting " + str(batch_number) + " batch abstract", end="... ")
        abstracts_extracted = tm_utils.extract_only_abstract("C:\\Users\\Davide\\Desktop\\TesiAPPimproved\\", prev_bn,
                                                             batch_number)
        print("Done ✓")
        print("INFO: Preprocessing asbtract", end="... ")
        start_prep = datetime.now()
        abstracts_extracted = tm_utils.preprocess_abstract(abstracts_extracted)
        end_prep = datetime.now()
        d_prep = end_prep - start_prep
        print("Done ✓")

    abstracts = abstracts + abstracts_extracted
    start_compute = datetime.now()
    filename = lda_method_abstract.compute(batch_number, number_topic, abstracts)
    end_compute = datetime.now()
    d_compute = end_compute - start_compute

    with io.open(filename, "a+", encoding="utf-8") as f:
        f.write("\n\nPreprocessing time (" + str(batch_number - prev_bn) + " more batch): " + str(d_prep))
        f.write("\nLDA computing time: " + str(d_compute))
        if d_prep != 0:
            f.write("\nTotal Time: " + str(d_prep + d_compute))
        else:
            f.write("\nTotal Time: " + str(d_compute))
