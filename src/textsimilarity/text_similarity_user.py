from src.neuralnetwork import lstm_utils
from src.pipelineapplication import utils
from src.textsimilarity import ts_utils

emb_dim = 300
glove_path = '../../input/glove.6B.300d.txt'

abstracts = utils.get_abstracts_to_analyze()
abstracts = list(map(lambda x: lstm_utils.preprocess_text(x["abstract"]), abstracts))
abstract1 = abstracts[0]
abstract2 = abstracts[1]

embeddings_dictionary = lstm_utils.get_embedding_dict(glove_path)

abstract1_emb = ts_utils.compute_embedding_vector(abstract1, embeddings_dictionary)
abstract2_emb = ts_utils.compute_embedding_vector(abstract2, embeddings_dictionary)

cos = ts_utils.compute_cosine_similarity(abstract1_emb, abstract2_emb, emb_dim)
print(cos)
