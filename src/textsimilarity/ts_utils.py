import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


def compute_cosine_similarity(abstract1, abstract2, emb_dim):
    total = 0
    for a1 in abstract1:
        a1_res = a1.reshape(1, emb_dim)
        for a2 in abstract2:
            a2_res = a2.reshape(1, emb_dim)
            cos = cosine_similarity(a1_res, a2_res)
            total += cos[0][0]

    return total


def compute_embedding_vector(abstract, emb_dic):
    res = []
    for word in abstract.split():
        emb_vector = emb_dic.get(word)
        if emb_vector is not None:
            res.append(emb_vector)

    return np.asarray(res)
