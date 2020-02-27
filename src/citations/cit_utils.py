from src.lstm import lstm_utils


def get_cit_topic(id, dataset):
    for data in dataset:
        if data["id"] == id:
            return lstm_utils.extract_topic_from_dataset(data["topic"])

    return None
