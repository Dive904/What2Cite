# This script is used to implement the pipeline for only specific abstracts given in input in a list

import pickle
import numpy as np

from keras_preprocessing.sequence import pad_sequences
from tensorflow_core.python.keras.models import load_model

from src.fileutils import file_abstract
from src.pipelineapplication import utils
from src.lstm import lstm_utils
from src.textsimilarity import ts_utils
from src.topicmodeller import tm_utils

# input
cit_labelled_analyzed_pickle_path = "../../output/official/cit_labelled_with_final_topic_pickle.pickle"
lstm_model = "../../output/official/lstm.h5"
tokenizer_model = "../../output/official/tokenizer.pickle"
cit_labelled_path = "../../output/official/topics_cits_labelled_pickle.pickle"
cit_topic_info_pickle_path = "../../output/official/cit_topic_info_pickle.pickle"
glove_path = '../../input/glove.840B.300d.txt'
cit_structure_pickle_path = "../../output/official/cit_structure_pickle.pickle"
hittingplot_base_path = "../../output/hittingplots/"
closedataset_path = "../../output/official/closedataset.txt"
Percentile = 10
emb_dim = 300
P = 1
Pt = 0.05
J = 3
N_papers = 1500
t_for_true_prediction = 0.4  # probability threshold to consider a prediction as valid

# output
missig_citation_path = "../../output/official/missing_citations.txt"
abstracts_pickle_path = "../../output/official/abstracts_pickle.pickle"
hittingplot_total_path = "../../output/hittingplots/total_hitting_plot" + str(Percentile) + ".png"

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

for abstract in abstracts:
    abstract["missing"] = []

# abstract = [{id = "...", title = "...", outCitations = ["..."], missing = [], hit = [}]

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
    abstracts[i]["validPredictions"] = valid_predictions
print("Done ✓")

# abstract = [{id = "...", title = "...", outCitations = ["..."], validPredictions = [("topic", prob)], missing = []}]

# TODO: we should work in this loop to find a good method for CitTopic selection
print("INFO: Analyzing", end="... ")
# TODO: disable this line if you don't want to allow the document similarity
# embeddings_dictionary = lstm_utils.get_embedding_dict(glove_path)
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

    total_possible_cit_topics = total_possible_cit_topics * len(valid_predictions)

    # TODO: disable this line if you don't want to allow the document similarity
    # paper_abstract_emb = ts_utils.compute_embedding_vector(paper_abstract, embeddings_dictionary)

    for k in range(len(valid_predictions)):
        topic = valid_predictions[k][0]
        prob = valid_predictions[k][1]
        score_count_list = utils.all_score_function(topic, cit_topics, cit_structure)
        """
        score_count_list = []
        for cit_topic in cit_topics:
            score_count = 0
            for cit in cit_topic:
                score_list = cit_structure.get(cit)
                score_count += score_list[topic]
            score_count_list.append(score_count)
        """
        score_count_list = list(enumerate(score_count_list))


        # choice of CitTopic
        '''
        score_mean = np.mean(score_count_list)
        score_count_list = list(filter(lambda x: x[1] > score_mean, score_count_list))
        '''
        tmp = list(map(lambda x: x[0], score_count_list))

        # Create hitting plots
        title = abstracts[i]["id"] + " - " + str(topic)
        score_count_list.sort(key=lambda x: x[1])
        index_score_count_list = list(map(lambda x: x[0], score_count_list))
        height = []
        for index in index_score_count_list:
            reference_cit_topic = cit_topics[index]
            t = utils.compute_hit_citations(reference_cit_topic, out_citations)
            height.append(len(t))
        bars = list(map(lambda x: str(x), index_score_count_list))
        path = hittingplot_base_path + abstracts[i]["id"] + "#" + str(topic) + ".png"
        # utils.make_bar_plot(height, bars, title, path)

        heights_split.append((utils.split_list(height, Percentile), total_possible_cit_topics))
        # TODO: if you don't want to allow the document similarity, disable this fraction of code
        '''
        tmp1 = []
        for index in tmp:
            cit_topic = cit_topics[index]
            total = 0
            for c in cit_topic:
                abstract = cit_topic_info.get(c)[1]
                if abstract is not None:
                    abstract = lstm_utils.preprocess_text(abstract)
                    abstract = ts_utils.compute_embedding_vector(abstract, embeddings_dictionary)
                    cos = ts_utils.compute_cosine_similarity(paper_abstract_emb, abstract, emb_dim)
                    total += cos
            tmp1.append(total)

        avg = np.mean(tmp1)
        tmp2 = []
        for w in range(len(tmp1)):
            if tmp1[w] > avg:
                tmp2.append(tmp[w])
        tmp = tmp2
        # end of fraction code of document similarity
        '''
        valid_predictions[k] = (topic, prob, tmp)

# create aggregate plot
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
total = list(map(lambda x: int(x), total))
bars = [str(i + 1) for i in range(max_index)]
utils.make_bar_plot(total, bars, "Total Hitting Plot - " + str(Percentile) + "% - " + str(N_papers) + " abstracts - #4",
                    hittingplot_total_path)

# abstract = [{id = "...", title = "...", outCitations = ["..."],
#                                                   validPredictions = [(topic, prob, [index])], missing = []}]

for i in range(len(abstracts)):
    valid_predictions = abstracts[i]["validPredictions"]
    out_citations = abstracts[i]["outCitations"]
    for valid in valid_predictions:
        topic = valid[0]
        reference_cit_topic_index = valid[2]
        missing = []
        for index in reference_cit_topic_index:
            reference_cit_topic = cit_topics[index]
            t = utils.compute_missing_citations(reference_cit_topic, out_citations)
            n = [None for g in range(len(t))]
            t = list(zip(t, n))
            tt = list(zip(utils.compute_hit_citations(reference_cit_topic, out_citations), n))
            abstracts[i]["missing"].append((topic, index, t, tt))
print("Done ✓")

# abstract = [{id = "...", title = "...", abstract = "...", outCitations = ["..."],
#                                                   validPredictions = [(topic, prob, [index])],
#                                                   missing = [(topic, ref_index, [id_missing, None], [id_hit, None]]}]

for i in range(len(abstracts)):
    all_l = []
    for j in range(len(abstracts[i]["missing"])):
        t = []
        f = abstracts[i]["missing"][j][0]
        s = abstracts[i]["missing"][j][1]
        for k in range(len(abstracts[i]["missing"][j][2])):
            first = abstracts[i]["missing"][j][2][k][0]
            second = abstracts[i]["missing"][j][2][k][1]
            t.append((first, cit_topic_info.get(first)[0]))
        th = []
        for k in range(len(abstracts[i]["missing"][j][3])):
            first = abstracts[i]["missing"][j][3][k][0]
            second = abstracts[i]["missing"][j][3][k][1]
            th.append((first, cit_topic_info.get(first)[0]))
        all_l.append((f, s, t, th))
    abstracts[i]["missing"] = all_l

# abstract = [{id = "...", title = "...", abstract = "...", outCitations = ["..."],
#                                                   validPredictions = [(topic, prob, [index])],
#                                                  missing = [(topic, ref_index, [id_missing, title], [id_hit, title]]}]

print("INFO: Writing output file", end="... ")
with open(abstracts_pickle_path, "wb") as handle_file:
    pickle.dump(abstracts, handle_file, protocol=pickle.HIGHEST_PROTOCOL)

with open(missig_citation_path, "w", encoding="utf-8") as out_file:
    for elem in abstracts:
        out_file.write("Paper ID: " + str(elem["id"]) + "\n")
        out_file.write("Paper Title: " + str(elem["title"]) + "\n")
        out_file.write("Paper year: " + str(elem["year"]) + "\n")
        missing = elem["missing"]
        out_file.write("Possible Citation Topics found: " + str(len(missing)) + "\n")
        for m in missing:
            topic = m[0]
            reference_cit_topic_index = m[1]
            missing_couple = m[2]
            hit_couple = m[3]
            cit_topic_len = len(cit_topics[reference_cit_topic_index])
            missing_couple_len = len(missing_couple)
            if cit_topic_len != missing_couple_len:
                out_file.write("Paper Topic (Classified): " + str(topic) + "\n")
                out_file.write("Reference CitTopic index: " + str(reference_cit_topic_index) + "\n")
                out_file.write("Missing citations: " + str(missing_couple_len) + " out of " + str(cit_topic_len) + "\n")
                for couple in missing_couple:
                    out_file.write(str(couple[0]) + " - " + str(couple[1]) + "\n")
                out_file.write("Hit citations: " + str(len(hit_couple)) + " out of " + str(cit_topic_len) + "\n")
                for couple in hit_couple:
                    out_file.write(str(couple[0]) + " - " + str(couple[1]) + "\n")

        out_file.write("*****" + "\n\n")
print("Done ✓")
