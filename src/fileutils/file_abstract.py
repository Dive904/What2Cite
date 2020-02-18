# Library with file utils

import io


def txt_abstract_creator(filename, list_dic):
    """
    Use this method to create a txt file with paper:
    TITLE
    ABSTRACT
    ---
    :param filename: filename
    :param list_dic: list of json dictionary (or a dictionary with title and abstract)
    """
    with io.open(filename + ".txt", "w", encoding="utf-8") as f:
        for d in list_dic:
            f.write("*** TITLE: " + d["title"] + "\n")
            f.write("*** ABSTRACT: " + d["paperAbstract"] + "\n")
            f.write("---" + "\n")


def txt_only_abstract_reader(filepath):
    result = []
    d = {}

    state = "e"
    with io.open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("*** TITLE: "):
                state = "t"
                d["title"] = line.split("*** TITLE: ")[1][:-1]
            elif line.startswith("*** ABSTRACT: "):
                state = "a"
                d["paperAbstract"] = line.split("*** ABSTRACT: ")[1][:-1]
            elif line.startswith("---"):
                result.append(d)
                state = "e"
                d = {}
            else:
                if state == "t":
                    d["title"] += line[:-1]
                elif state == "a":
                    d["paperAbstract"] += line[:-1]

    return result
