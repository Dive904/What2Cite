import pickle

cit_structure_pickle_path = "../../output/official/cit_structure_pickle.pickle"
citopic = "82266f6103bade9005ec555ed06ba20b5210ff22 0c8413ab8de0c1b8f2e86402b8d737d94371610f " \
          "0a2586e0a5f8bb4e35aa0763a6b8bca428af6bd2 cd5a26b89f0799db1cbc1dff5607cb6815739fe7 " \
          "daa63f57c3fbe994c4356f8d986a22e696e776d2 2e2089ae76fe914706e6fa90081a79c8fe01611e " \
          "93bc65d2842b8cc5f3cf72ebc5b8f75daeacea35"

with open(cit_structure_pickle_path, 'rb') as handle:  # take the list of CitTopic score
    cit_structure = pickle.load(handle)

citopic = citopic.split()

S = []

for c in citopic:
    scores = cit_structure[c]
    print(scores)
    S.append(scores)

res = []

i = 0
while i < len(S[0]):
    su = 0
    for l in S:
        su += l[i]
    res.append(su)
    i += 1

res = list(enumerate(res))
res.sort(key=lambda x: x[1], reverse=True)
print(res)
