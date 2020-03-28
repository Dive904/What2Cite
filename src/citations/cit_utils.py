def read_topics_cit_file(filepath):
    """
    This function is used to read CitTopic file
    :param filepath: path of the file
    :return: a list of all the CitTopics
    """
    res = []
    with open(filepath, "r", encoding="utf-8") as input_file:
        lines = input_file.readlines()
        for line in lines:
            res.append(line.split(" -> ")[1].split())

    return res


def count_none(input_list):
    """
    This script is used to count nones in a CitTopic labelled (after the classification of the papers in the CitTopic)
    :param input_list: CitTopic list
    :return: number of None in the input_list
    """
    ris = 0
    for l in input_list:
        for x in l:
            if x[1] is None:
                ris += 1

    return ris


def get_abstract_document_topic_matrix(filename):
    """
    This script is used to read the Abstract-Document-Topic Matrix
    :param filename: path of the matrix file
    :return: a dictionary where every key is a paper id and every associated element is a list with values
    """
    ris = {}
    with open(filename, "r", encoding="utf-8") as input_file:
        lines = input_file.readlines()[1:]

    for line in lines:
        line_splitted = line[:-1].split(",")
        ris[line_splitted[0]] = [float(x) for x in line_splitted[1:]]

    return ris
