def get_abstracts_to_analyze():
    """
    This is only a simple function that helps to keep the abstract to analyze
    :return: list of couple where the first element is the paper abstract and the second element is the paper id
    """
    text = [("In this paper, we present a unique two-stage classifier system for identifying normal mammograms. "
             "We present methods that extract features from breast regions characterizing normal and cancerous "
             "tissue. A subset of the features is used to construct a classifier. This classifier is then used to "
             "classify each mammogram as normal or abnormal. We designed a unique two-stage cascading classifier "
             "system. A binary decision tree classifier was used in the first stage. Cost constraints can be set to "
             "correctly classify cancerous regions. The regions classified as abnormal in the first-stage were used "
             "as input to the second-stage classifier, a linear classifier. We will show that the overall performance "
             "of our two-stage cascading classifier is better than a single classifier. Results of full-field normal "
             "human reader.", "d9fdb21a2cd87d7b7a2765c1c279b9a58f8f24d7")  # T25
            ]

    return text


def normalize_scores_on_cittopics(cittopic, p):
    """
    This function is used to normalize scores in a specific CitTopic
    :param cittopic: the CitTopic score list
    :param p: max score to assigned to a topic
    :return: the normalized CitTopic score list
    """
    max_value = p * len(cittopic)

    return list(map(lambda x: x / max_value, cittopic))


def compute_missing_citations(cittopics, outcitations):
    """
    This function is used to compute the missing citations for a paper
    :param cittopics: the CitTopic
    :param outcitations: the list of records including outcitations
    :return: a list of missing citations
    """
    missing = []
    for c in cittopics:
        if c not in outcitations:
            missing.append(c)

    return missing
