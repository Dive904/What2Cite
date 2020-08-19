import pickle

from keras.engine.saving import load_model
from keras_preprocessing.sequence import pad_sequences

from src.neuralnetwork import lstm_utils
from src.pipelineapplication import utils

text = "The continuous evolution of research has led to an exponential growth of the  scientific literature. " \
       "This leads to a difficulty for researchers to document themselves in the most appropriate way, looking for " \
       "treatises and papers that can be useful for their research. In this paper, we propose a novel model for " \
       "capturing meaningful and labeled relations between articles based on both topics and latent citation " \
       "dependencies."

lstm_model = "../../output/official/neuralnetwork.h5"
tokenizer_model = "../../output/official/tokenizer.pickle"
cit_structure_pickle_path = "../../output/official/cit_structure_pickle.pickle"
cit_labelled_path = "../../output/official/topics_cits_labelled_pickle.pickle"
cit_topic_info_pickle_path = "../../output/official/cit_topic_info_pickle.pickle"
N = 10

text = lstm_utils.preprocess_text(text)

with open(tokenizer_model, 'rb') as handle:
    tokenizer = pickle.load(handle)

seq = tokenizer.texts_to_sequences([text])
seq = pad_sequences(seq, padding='post', maxlen=200)

# load model from single file
model = load_model(lstm_model)

yhat = model.predict_classes(seq)
topic = yhat[0]

with open(cit_structure_pickle_path, 'rb') as handle:  # take the list of CitTopic score
    cit_structure = pickle.load(handle)

with open(cit_topic_info_pickle_path, 'rb') as handle:  # take the list of CitTopic score
    cit_topic_info = pickle.load(handle)

with open(cit_labelled_path, 'rb') as handle:  # take the list of CitTopic score
    # in this part, we read the cit topics labelled with other information.
    # We don't need that. So, we'll do a map removing all noise, getting only the paper id for all CitTopic
    cit_topics = pickle.load(handle)
    cit_topics = list(map(lambda x: list(map(lambda y: y[0], x)), cit_topics))

score_count_list = utils.all_score_function(topic, cit_topics, cit_structure)

score_count_list = list(enumerate(score_count_list))

score_count_list.sort(key=lambda x: x[1])
index_score_count_list = list(map(lambda x: x[0], score_count_list))

index_score_count_list = index_score_count_list[:N]

print("TOPIC: " + str(topic))
for i in index_score_count_list:
    print("+++++     " + str(i) + "     +++++")
    cit = cit_topics[i]
    for c in cit:
        title = cit_topic_info.get(c)
        if title is not None:
            title = title[0]
        print(title)





