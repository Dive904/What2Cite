# This script is used to analyze the CitTopic, classifying all the papers in the CitTopic and getting the TopK
import pickle
import gc

from keras.engine.saving import load_model
from keras_preprocessing.sequence import pad_sequences

from src.fileutils import file_abstract
from src.citations import cit_utils
from src.lstm import lstm_utils
from src.pipelineapplication import utils

# input
close_dataset = "../../output/closedataset/closedataset.txt"
topic_citations_filename = "../../output/official/topics_cits.txt"
lstm_model = "../../output/official/lstm.h5"
tokenizer_model = "../../output/official/tokenizer.pickle"
topic_number = 40
t_for_true_prediction = 0.4  # probability threshold to consider a prediction as valid

# output
cit_structure_pickle_path = "../../output/closedataset/cit_structure_pickle.pickle"

print("INFO: Reading Citation Topic file and initializing structure", end="... ")
cit_structure = {}
cit_topic = cit_utils.read_topics_cit_file(topic_citations_filename)
for cit in cit_topic:
    for c in cit:
        cit_structure[c] = [0 for x in range(topic_number)]
print("Done ✓")

print("INFO: Reading close dataset", end="... ")
paper_info = file_abstract.txt_dataset_reader(close_dataset)
print("Done ✓")

print("INFO: Reading Tokenizer and Neural Network", end="... ")
with open(tokenizer_model, 'rb') as handle:
    tokenizer = pickle.load(handle)

model = load_model(lstm_model)
print("Done ✓")

print("INFO: Analyzing", end="... ")
for paper in paper_info:
    out_citations = paper["outCitations"]
    topic = None
    for out_cit in out_citations:
        score_topic = cit_structure.get(out_cit)
        if score_topic is not None:
            if topic is None:
                text = paper["paperAbstract"]
                text = lstm_utils.preprocess_text(text)
                seq = tokenizer.texts_to_sequences([text])
                seq = pad_sequences(seq, padding='post', maxlen=200)
                topic = model.predict(seq)
                valid_topic = utils.get_valid_predictions(topic[0], t_for_true_prediction)
                topic = list(map(lambda x: x[0], valid_topic))
            for t in topic:
                score_topic[t] += 1
            cit_structure[out_cit] = score_topic
    gc.collect()
print("Done ✓")

print("INFO: Writing on file", end="... ")
with open(cit_structure_pickle_path, "wb") as handle_file:  # saving list to use in pipeline application
    pickle.dump(cit_structure, handle_file, protocol=pickle.HIGHEST_PROTOCOL)
print("Done ✓")
