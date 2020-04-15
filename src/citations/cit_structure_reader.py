# input
import pickle

cit_structure_pickle_path = "../../output/closedataset/cit_structure_pickle.pickle"
with open(cit_structure_pickle_path, 'rb') as handle:  # take the list of CitTopic score
    cit_structure = pickle.load(handle)
print("Done ✓")

print(cit_structure)
