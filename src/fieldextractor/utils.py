from langdetect import detect


def apply_filter(d):
    if "paperAbstract" in d.keys() and "title" in d.keys() and "fieldsOfStudy" in d.keys() and "sources" in d.keys():
        if len(d["paperAbstract"]) > 0 and "DBLP" in d["sources"] and "Computer Science" in d["fieldsOfStudy"]:
            return True

    return False


def check_if_abstract_is_english(abstract):
    lang = "x"
    try:
        lang = detect(abstract)
    except:
        lang = "x"

    return lang == "en"
