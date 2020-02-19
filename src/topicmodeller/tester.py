from datetime import datetime
import io

from src.topicmodeller import tm_utils
from src.topicmodeller import lda_method

# Number of Batch, Number of Topic
test_index = [(10, 25), (10, 50), (25, 25), (25, 50), (30, 25), (30, 50), (45, 25), (55, 25), (55, 50)]

for index in test_index:
    batch_number = index[0]
    number_topic = index[1]
    start = datetime.now()
    start_str = start.strftime("%H:%M:%S")
    print("INFO: Start Time: " + str(start))
    filename = lda_method.compute(batch_number, number_topic)
    end = datetime.now()
    end_str = start.strftime("%H:%M:%S")
    print("INFO: End Time: " + str(end))

    with io.open(filename, "a+", encoding="utf-8") as f:
        f.write("\n" + str(end - start))
