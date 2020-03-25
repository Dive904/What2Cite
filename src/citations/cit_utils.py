from src.lstmdatasetcreator import utils


def read_topics_cit_file(filepath):
    res = []
    with open(filepath, "r", encoding="utf-8") as input_file:
        lines = input_file.readlines()
        for line in lines:
            res.append(line.split(" -> ")[1].split())

    return res


def get_cit_topic(id_in, dataset):
    for data in dataset:
        if data["id"] == id_in:
            return utils.extract_topic_from_dataset(data["topic"])

    return None


def get_total_energy(input_list):
    ris = 0
    for x in input_list:
        ris += x[1]

    return ris


def count_none(input_list):
    ris = 0
    for l in input_list:
        for x in l:
            if x[1] is None:
                ris += 1

    return ris


def get_abstract_document_topic_matrix(filename):
    ris = {}
    with open(filename, "r", encoding="utf-8") as input_file:
        lines = input_file.readlines()[1:]

    for line in lines:
        line_splitted = line[:-1].split(",")
        ris[line_splitted[0]] = [float(x) for x in line_splitted[1:]]

    return ris
