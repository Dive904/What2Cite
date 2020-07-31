import pickle
import numpy as np

from keras_preprocessing.sequence import pad_sequences
from tensorflow_core.python.keras.models import load_model

from src.fileutils import file_abstract
from src.pipelineapplication import utils
from src.lstm import lstm_utils

# input
cit_labelled_analyzed_pickle_path = "../../output/official/cit_labelled_with_final_topic_pickle.pickle"
lstm_model = "../../output/official/lstm.h5"
tokenizer_model = "../../output/official/tokenizer.pickle"
cit_labelled_path = "../../output/official/topics_cits_labelled_pickle.pickle"
cit_topic_info_pickle_path = "../../output/official/cit_topic_info_pickle.pickle"
cit_structure_pickle_path = "../../output/official/cit_structure_pickle.pickle"
hittingplot_base_path = "../../output/hittingplots/"
closedataset_path = "../../output/official/closedataset.txt"
Percentile = 10
N_papers = 1500
t_for_true_prediction = 0.4  # probability threshold to consider a prediction as valid
N = 3

# output
hittingplot_total_path = "../../output/hittingplots/total_hitting_plot" + \
                         str(Percentile) + "_" + str(N_papers) + ".png"

with open(cit_topic_info_pickle_path, 'rb') as handle:  # take the list of CitTopic score
    cit_topic_info = pickle.load(handle)

with open(cit_structure_pickle_path, 'rb') as handle:  # take the list of CitTopic score
    cit_structure = pickle.load(handle)
print("Done ✓")

with open(cit_labelled_path, 'rb') as handle:  # take the list of CitTopic score
    # in this part, we read the cit topics labelled with other information.
    # We don't need that. So, we'll do a map removing all noise, getting only the paper id for all CitTopic
    cit_topics = pickle.load(handle)
    cit_topics = list(map(lambda x: list(map(lambda y: y[0], x)), cit_topics))

with open(cit_labelled_analyzed_pickle_path, 'rb') as handle:  # take the list of CitTopic score
    cit_topic_labelled = pickle.load(handle)

with open(tokenizer_model, 'rb') as handle:  # get the tokenizer
    tokenizer = pickle.load(handle)

model = load_model(lstm_model)  # load model from single file

# abstracts = utils.get_abstracts_to_analyze()  # get the abstract to analyze
print("INFO: Reading close dataset and picking random abstracts", end="... ")
abstracts = file_abstract.txt_dataset_reader(closedataset_path)
abstracts = abstracts[500000:]
abstracts = utils.pick_random_abstracts(abstracts, N_papers)
print("Done ✓")

# abstract = [{id = "...", title = "...", "year": "..." outCitations = ["..."]}]

# prepare texts for classification
abstracts_prep = list(map(lambda x: lstm_utils.preprocess_text(x["paperAbstract"]), abstracts))

# abstract = [{id = "...", title = "...", outCitations = ["..."]}]

print("INFO: Tokenizing sequences", end="... ")
seq = tokenizer.texts_to_sequences(abstracts_prep)
seq = pad_sequences(seq, padding='post', maxlen=200)
print("Done ✓")

print("INFO: Getting predictions", end="... ")
yhat = model.predict(seq)  # make predictions
for i in range(len(yhat)):
    # at the end of the loop, every dictionary for the abstracts, will contain an additional element with all the
    # valid predictions. Please note: the list of abstracts and the list of yhat (i.e predictions for every element)
    # have the same length
    valid_predictions = utils.get_valid_predictions(yhat[i], t_for_true_prediction)
    # valid_predictions = utils.get_n_predictions(yhat[i], N)
    abstracts[i]["validPredictions"] = valid_predictions
print("Done ✓")

# abstract = [{id = "...", title = "...", outCitations = ["..."], validPredictions = [("topic", prob)]}]

print("INFO: Analyzing", end="... ")
heights_split = []
for i in range(len(abstracts)):
    # at the end of this loop, every valid prediction will be extended with the index of reference CitTopics
    valid_predictions = abstracts[i]["validPredictions"]
    paper_abstract = abstracts[i]["paperAbstract"]
    out_citations = abstracts[i]["outCitations"]

    # count total possible CitTopics
    total_possible_cit_topics = 0
    for cit in cit_topics:
        for c in cit:
            if c in out_citations:
                total_possible_cit_topics += 1

    # total_possible_cit_topics = total_possible_cit_topics * len(valid_predictions)

    for k in range(len(valid_predictions)):
        topic = valid_predictions[k][0]
        prob = valid_predictions[k][1]
        score_count_list = utils.all_score_function(topic, cit_topics, cit_structure)

        score_count_list = list(enumerate(score_count_list))

        tmp = list(map(lambda x: x[0], score_count_list))

        # Create hitting plots
        title = abstracts[i]["id"] + " - " + str(topic)
        score_count_list.sort(key=lambda x: x[1], reverse=True)
        index_score_count_list = list(map(lambda x: x[0], score_count_list))
        height = [0 for index in index_score_count_list]
        for ii in range(len(index_score_count_list)):
            index = index_score_count_list[ii]
            reference_cit_topic = cit_topics[index]
            t = utils.compute_hit_citations(reference_cit_topic, out_citations)
            # height.append(len(t))
            height[ii] += len(t)

        valid_predictions[k] = (topic, prob, tmp)

    heights_split.append((utils.split_list(height, Percentile), total_possible_cit_topics))
print("Done ✓")
# abstract = [{id = "...", title = "...", outCitations = ["..."],
#                                                   validPredictions = [(topic, prob, [index])]}]

# create aggregate plot
print("INFO: Creating aggregate plot", end="... ")
max_index = int(100 / Percentile)
total = list(np.zeros(max_index))
total_c = 0
for heights in heights_split:
    hh = 0
    while hh < len(heights[0]):
        total[hh] += sum(heights[0][hh])
        hh += 1
    total_c += heights[1]

total = utils.normalize_cit_count(total, total_c)
total_to_plot = list(map(lambda x: int(x), total))
total_to_print = list(map(lambda x: np.round(x, 2), total))
bars = ["P" + str(i + 1) for i in range(max_index)]
utils.make_bar_plot(total_to_plot, bars, "Total Hitting Plot - " + str(Percentile) + "% - " +
                    str(N_papers) + " abstracts - #4",
                    hittingplot_total_path)
print("Done ✓")

print(total_to_print)
