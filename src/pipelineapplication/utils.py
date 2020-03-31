def get_abstracts_to_analyze():
    """
    This is only a simple function that helps to keep the abstract to analyze
    :return: list of couple where the first element is the paper abstract and the second element is the paper id
    """
    text = [("We present Cohesion, a novel approach to Desktop Grid Computing. A major design goal of Cohesion is "
             "to enable advanced parallel programming models and application specific frameworks. We focus on "
             "methods for irregularly structured task-parallel problems, which require fully dynamic problem "
             "decomposition. Cohesion overcomes limitations of classical Desktop Grid platforms by employing "
             "peer-to-peer principles and a flexible system architecture based on a microkernel approach. "
             "Arbitrary modules can be dynamically loaded to replace default functionality, resulting in a platform "
             "that can easily adapt to application specific requirements. We discuss two representative example "
             "applications and report on the results of performance experiments that especially consider the high "
             "volatility of resources prevailing in a Desktop Grid.", "36af398b1a6f1f4bf68b04d41b59b490f3b2824e")
            ]

    return text


def normalize_scores_on_cittopics(cittopic, p):
    """
    This function is used to normalize scores in a specific CitTopic
    :param cittopic: the CitTopic score list
    :param p: max score to assigned to a topic
    :return: the normalized CitTopic score list
    """
    max_value = p * len(cittopic) + 1

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
