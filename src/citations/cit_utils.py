from src.datasetcreator import utils


def get_cit_topic(id, dataset):
    for data in dataset:
        if data["id"] == id:
            return utils.extract_topic_from_dataset(data["topic"])

    return None
