# import string


def apply_filter(d):
    if "paperAbstract" in d.keys() and "title" in d.keys() and "fieldsOfStudy" in d.keys() and "sources" in d.keys():
        if len(d["paperAbstract"]) > 0 and "DBLP" in d["sources"] and "Computer Science" in d["fieldsOfStudy"]:
            return True

    return False


"""
def check_if_abstract_is_english(abstract):
    abstract = abstract.translate(str.maketrans("", "", string.punctuation))
    abstract = abstract.split()
    new_abstract = []
    for word in abstract:
        if not check_has_upper(word):
            new_abstract.append(word)

    if len(new_abstract) == 0:
        return False


def check_has_upper(word):
    for w in word:
        if "A" <= w <= "Z":
            return True
    return False
"""