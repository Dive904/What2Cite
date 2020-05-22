import pickle
import matplotlib.pyplot as plt

cit_labelled_path = "../../output/official/topics_cits_labelled_pickle.pickle"

with open(cit_labelled_path, 'rb') as handle:  # take the list of CitTopic score
    # in this part, we read the cit topics labelled with other information.
    # We don't need that. So, we'll do a map removing all noise, getting only the paper id for all CitTopic
    cit_topics = pickle.load(handle)
    cit_topics = list(map(lambda x: list(map(lambda y: y[0], x)), cit_topics))

cit_topics = list(map(lambda x: len(x), cit_topics))
cit_topics.sort()
res = {}

for c in cit_topics:
    count = res.get(c)
    if count is not None:
        count += 1
    else:
        count = 1
    res[c] = count

print(res)

X = list(res.keys())
Y = []

for x in X:
    Y.append(res[x])

print(X)
print(Y)

plt.bar(list(map(lambda x: str(x), X)), Y)
plt.xlabel("Length")
plt.ylabel("Number of CiTopic with length X")
plt.show()
