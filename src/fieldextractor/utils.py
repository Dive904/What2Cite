from langdetect import detect


def apply_filter(d, fos):
    """
    This function is used to apply all the useful filers
    :param d: jason as a dictionary
    :param fos: field of study
    :return: the result of the filer
    """
    if "paperAbstract" in d.keys() and "title" in d.keys() and "fieldsOfStudy" in d.keys() and "sources" in d.keys():
        return len(d["paperAbstract"]) > 0 and "DBLP" in d["sources"] and fos in d["fieldsOfStudy"]

    return False


def check_if_abstract_is_english(abstract):
    """
    This function is used to check if an abstract is English
    :param abstract:
    :return:
    """
    try:
        lang = detect(abstract)
    except:
        lang = "x"

    return lang == "en"
