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
            f.write("*** ID: " + d["id"] + "\n")
            f.write("*** TITLE: " + d["title"] + "\n")
            f.write("*** ABSTRACT: " + d["paperAbstract"] + "\n")
            f.write("---" + "\n")


def txt_abstract_reader(filepath):
    """
    Use this method to read an abstract file
    :param filepath: filepath
    :return: a list where every element is a python dictionary with title and abstract
    """
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


def txt_only_abstract_reader(filepath):
    """
    Use this method to read ONLY an abstract from file
    :param filepath:
    :return: a list of abstract
    """
    result = []
    state = "e"
    tmp_abstract = ""
    with io.open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            if not line.startswith(" \n"):
                if line.startswith("*** TITLE: "):
                    state = "t"
                elif line.startswith("*** ABSTRACT: "):
                    state = "a"
                    tmp_abstract = line.split("*** ABSTRACT: ")[1][:-1]
                elif line.startswith("---"):
                    result.append(tmp_abstract)
                    state = "e"
                    tmp_abstract = ""
                else:
                    if state == "a":
                        tmp_abstract += " " + line[:-1]

    return result
