from src.lstmdatasetcreator import utils


def get_cit_topic(id, dataset):
    for data in dataset:
        if data["id"] == id:
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
