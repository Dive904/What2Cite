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
