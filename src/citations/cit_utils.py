from src.lstmdatasetcreator import utils


def get_cit_topic(id, dataset):
    for data in dataset:
        if data["id"] == id:
            return utils.extract_topic_from_dataset(data["topic"])

    return None


def count_none(list):
    ris = 0
    for l in list:
        for x in l:
            if x[1] is None:
                ris += 1

    return ris
